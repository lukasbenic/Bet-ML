import copy
from collections import deque
import joblib
import numpy as np
import os
from pprint import pprint
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
import torch
import yaml

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    Ridge,
)
from sklearn.svm import SVR
from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
    mean_absolute_error as mae,
    mean_squared_error as mse,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.samplers import TPESampler
from ensemble_regressor import EnsembleRegressor

from onedrive import Onedrive
from utils.constants import regression_models


def objective(trial, x_train_df, y_train_df, model_name):
    if model_name == "Ridge":
        alpha = trial.suggest_float("alpha", 1e-5, 1e2)
        model = Ridge(alpha=alpha)
    elif model_name == "KNeighborsRegressor":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    elif model_name == "Lasso":
        alpha = trial.suggest_float("alpha", 1e-5, 1e2)
        model = Lasso(alpha=alpha)
    elif model_name == "ElasticNet":
        alpha = trial.suggest_float("alpha", 1e-5, 1e2)
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elif model_name == "BayesianRidge":
        alpha_1 = trial.suggest_float("alpha_1", 1e-9, 1e-4)
        alpha_2 = trial.suggest_float("alpha_2", 1e-9, 1e-4)
        lambda_1 = trial.suggest_float("lambda_1", 1e-9, 1e-4)
        lambda_2 = trial.suggest_float("lambda_2", 1e-9, 1e-4)
        model = BayesianRidge(
            alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2
        )
    elif model_name == "RandomForestRegressor":
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    elif model_name == "GradientBoostingRegressor":
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
        )
    elif model_name == "SVR":
        C = trial.suggest_float("C", 1e-4, 1e4)
        epsilon = trial.suggest_float("epsilon", 1e-4, 1e1)
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        model = SVR(C=C, epsilon=epsilon, kernel=kernel)
    else:
        raise ValueError(f"Invalid model name '{model_name}'")

    # Perform cross-validation with n_jobs=1 to avoid inefficiencies
    scores = cross_val_score(
        model,
        x_train_df,
        y_train_df,
        cv=3,
        scoring=make_scorer(r2_score),
        n_jobs=1,
    )
    return scores.mean()


# Train the best model using best parameters
def create_best_model(best_params, model_name, x_train_df, y_train_df):
    if model_name == "Ridge":
        model = Ridge(**best_params)
    elif model_name == "KNeighborsRegressor":
        model = KNeighborsRegressor(**best_params)
    elif model_name == "Lasso":
        model = Lasso(**best_params)
    elif model_name == "ElasticNet":
        model = ElasticNet(**best_params)
    elif model_name == "BayesianRidge":
        model = BayesianRidge(**best_params)
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(**best_params)
    elif model_name == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(**best_params)
    elif model_name == "SVR":
        model = SVR(**best_params)
    elif model_name == "Ensemble":
        model_files = os.listdir("models")
        ensemble_models = [
            joblib.load(f"models/{model}")
            for model in model_files
            if len(model.split("_")) == 1
        ]
        print("Models used for Ensemble")
        [print(model) for model in model_files]
        model = EnsembleRegressor(ensemble_models)
    else:
        raise ValueError(f"Invalid model name '{model_name}'")

    # Train the best model with the best parameters
    model.fit(x_train_df, y_train_df)
    return model


def rms(y_pred, y):
    rms = np.sqrt(np.mean((y - y_pred) ** 2))
    return rms


def update_tracker(tracker: dict, metrics: dict):
    for key, item in tracker.items():
        name_lst = key.split("_")
        m_key = "_".join(name_lst[1:])

        if key == "race_counter":
            continue

        # update non-list items in tracker
        if not isinstance(item, list):
            tracker[key] += metrics[m_key]

        # update the list items
        if isinstance(item, list):
            if "actual" in key:
                item.append(tracker["total_profit"])
            elif "expected" in key:
                item.append(tracker["total_m_c_margin"] + tracker["total_m_i_margin"])
            elif "green" in key:
                item.append(tracker["total_green_margin"])

    tracker["race_counter"] += 1
    return tracker


def process_run_results(results: dict, tracker: dict):
    def print_add(x: int, y: int, msg: str):
        x += y
        print(x, msg)

    prev_key = ""
    for key, item in tracker.items():
        if isinstance(item, list):
            if "actual" in key:
                item.append(tracker["total_profit"])
            if "expected" in key:
                item.append(tracker["total_m_c_marg"] + tracker["total_m_i_marg"])

            if "green" in key:
                item.append(tracker["total_green_margin"])
        else:
            name_lst = key.split("_")

            result_key = "_".join(name_lst[1:])

            if result_key == "m_c_marg" or result_key == "m_i_marg":
                print(f"{key}: {item}")
                continue

            if result_key == "counter":
                print(f"Race: {tracker[key]}")
                continue

            result = results[result_key]

            if "total" in name_lst:
                print_add(item, result, f"{key}: ")

            print(
                f"{key}: {item}"
            ) if result_key == "m_c_marg" or result_key == "m_i_marg" or result_key == "counter" else print(
                f"{result_key}: ", result
            )

            if not result_key in prev_key:
                print("---")

        prev_key = key

    if tracker["race_counter"] % 10 == 0:
        # plt.plot(range(race_counter), actual_profit_plotter, label="backtest", color="b")
        plt.plot(
            range(tracker["race_counter"]),
            tracker["expected_profit_plotter"],
            label="expected",
            color="y",
        )
        plt.plot(
            range(tracker["race_counter"]),
            tracker["green_plotter"],
            label="greened_profit",
            color="g",
        )
        plt.axhline(y=0.5, color="r", linestyle="-")
        plt.xlabel("Number of Races")
        plt.ylabel("Profit")

        plt.legend()
        plt.draw()
        print("")

    tracker["race_counter"] += 1


# def rms(y, y_pred):
#     rms = np.sqrt(np.mean((y - y_pred) ** 2))
#     return rms


def get_simulation_plot(
    tracker: dict,
    strategy_name: str,
    fig_size=(10, 6),
    style="darkgrid",
    palette="dark",
):
    sns.set_style(style)
    sns.set_palette(palette)

    profit_tracker, val_tracker = {}, {}
    for key, value in tracker.items():
        if key in ["actual_profit_plotter", "expected_profit_plotter", "green_plotter"]:
            print("here")
            profit_tracker[key] = value
        else:
            val_tracker[key] = value

    # Create a DataFrame for the val_tracker dictionary
    df = pd.DataFrame.from_dict(val_tracker, orient="index")

    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=profit_tracker, ax=ax)
    ax.set(
        title=f"Simulation Results: {strategy_name}",
        xlabel="Number of Races",
        ylabel="Profit",
    )

    print("Total expected profit is ", tracker["expected_profit_plotter"][-1])

    return fig


def train_test_model(
    ticks_df: pd.DataFrame,
    onedrive: Onedrive,
    model_name: str,
    regression=True,
    save=True,
    x_train_path="utils/x_train_df.csv",
    y_train_path="utils/y_train_df.csv",
    model_path="models/",
):
    x_train_df = (
        pd.read_csv(x_train_path, index_col=False)
        if os.path.exists(x_train_path)
        else None
    )
    y_train_df = (
        pd.read_csv(y_train_path, index_col=False)
        if os.path.exists(y_train_path)
        else None
    )
    print(y_train_df)
    mean120_train_df = None

    # Pre-process data
    if x_train_df is None or y_train_df is None:
        print(
            "x_train_df and/or y_train_df not found, commencing fetch and normalization..."
        )
        train_df = onedrive.get_train_df()
        train_df = normalized_transform(train_df, ticks_df)
        print("Finished train data normalization...")

        mean120_actual_train = train_df["mean_120_temp"]
        if not regression:
            train_df = train_df.drop(["mean_120_temp"], axis=1)

        mean120_train_df = train_df["mean_120"]
        bsp_train_df = train_df["bsps"]
        train_df["bsps"] = ((mean120_train_df - bsp_train_df) > 0).astype(int)

        df_majority = train_df[(train_df["bsps"] == 0)]
        df_minority = train_df[(train_df["bsps"] == 1)]

        # downsample majority
        df_majority = df_majority.head(
            len(df_minority)
        )  # because I don't trust the resample

        # Combine majority class with upsampled minority class
        train_df = pd.concat([df_minority, df_majority])
        mean120_train_df = train_df["mean_120"]

        y_train_df = train_df["bsps"]
        if regression:
            y_train_df = train_df["bsps_temp"]
        
        y_train_df.to_csv(y_train_path, index=False)

        x_train_df = train_df.drop(["bsps"], axis=1)
        x_train_df = x_train_df.drop(["bsps_temp"], axis=1)

    clm = x_train_df.columns
    scaler = StandardScaler()

    x_train_df = pd.DataFrame(scaler.fit_transform(x_train_df), columns=clm)
    print(x_train_df)
    y_train_df = y_train_df.values.ravel()
    print(y_train_df)

    mean120_train_df = (
        mean120_train_df if not mean120_train_df is None else x_train_df["mean_120"]
    )

    if not os.path.exists(x_train_path):
        x_train_df.to_csv(x_train_path, index=False)

    base_model = (
        joblib.load(f"{model_path}{model_name}.pkl")
        if os.path.exists(f"{model_path}{model_name}.pkl")
        else None
    )

    if base_model:
        print("Model successfully loaded.")
        return base_model, clm, scaler

    print(f"Commencing model training: {model_name}...")

    try:
        base_model = regression_models[model_name]
    except KeyError:
        raise ValueError(
            f"Invalid model name '{model_name}'. Please choose one of the following: {', '.join(regression_models.keys())}"
        )

    best_params = None
    if not model_name == "Ensemble":
        n_trials = 20 if model_name in ["RandomForestRegressor", "KNeighborsRegressor", "GradientBoostingRegressor"] else 200
        print(y_train_df)
        # Create Optuna study and optimize hyperparameters
        study = optuna.create_study(sampler=TPESampler(seed=42), direction="maximize")
        study.optimize(
            lambda trial: objective(trial, x_train_df, y_train_df, model_name),
            n_trials=n_trials,
        )

        # Get the best model
        best_params = study.best_trial.params

    best_model = create_best_model(best_params, model_name, x_train_df, y_train_df)

    # m.fit(x_train_df, y_train_df.values.ravel())
    _ = joblib.dump(best_model, f"{model_path}{model_name}.pkl")

    print("Model successfully trained")
    test_df = onedrive.get_test_df()
    metrics = test_model(
        ticks_df,
        best_model,
        scaler,
        clm,
        x_train_df,
        y_train_df,
        mean120_train_df,
        test_df,
    )

    if save:
        with open(f"{model_path}/{model_name}_results.yaml", "w") as f:
            yaml.dump(metrics, f)

    return best_model, clm, scaler


def test_model(
    ticks_df,
    model,
    scaler,
    clm,
    x_train_df,
    y_train_df,
    mean120_train_df,
    test_analysis_df,
    regression=True,
):
    test_analysis_df = test_analysis_df.dropna()
    test_analysis_df = test_analysis_df[
        (test_analysis_df["mean_120"] <= 50) & (test_analysis_df["mean_120"] > 1.1)
    ]
    test_analysis_df = test_analysis_df[test_analysis_df["mean_14400"] > 0]
    # below is a slight hack ...
    test_analysis_df = test_analysis_df.drop(
        test_analysis_df[test_analysis_df["std_2700"] > 1].index
    )

    test_analysis_df_y = pd.DataFrame().assign(
        market_id=test_analysis_df["market_id"],
        selection_ids=test_analysis_df["selection_ids"],
        bsps=test_analysis_df["bsps"],
    )
    # Sort out our test the same as before.
    test_df = copy.copy(test_analysis_df)
    test_df = normalized_transform(test_df, ticks_df)

    mean120_actual_test = test_df["mean_120_temp"]
    if not regression:
        test_df = test_df.drop(["mean_120_temp"], axis=1)

    mean120_test_df = test_df["mean_120"]
    bsp_test_df = test_df["bsps"]
    test_df["bsps"] = ((mean120_test_df - bsp_test_df) > 0).astype(int)

    y_test_df = copy.copy(test_df["bsps"])
    if regression:
        y_test_df = test_df["bsps_temp"]
    x_test_df = test_df.drop(["bsps"], axis=1)

    bsp_actual_test = test_df["bsps_temp"]
    x_test_df = x_test_df.drop(["bsps_temp"], axis=1)

    print("TEST ------")
    print(x_test_df)
    x_test_df = pd.DataFrame(scaler.transform(x_test_df), columns=clm)

    # test_analysis_df = test_analysis_df.drop(["bsps"], axis=1)
    y_pred_train = model.predict(x_train_df)
    y_pred_test = model.predict(x_test_df)

    metrics = {
        "mse": {
            "train": mse(y_pred_train, y_train_df),
            "train_mean_120": mse(mean120_train_df, y_train_df),
            "test": mse(y_pred_test, y_test_df),
            "test_mean_120": mse(mean120_test_df, y_test_df),
        },
        "mae": {
            "train": mae(y_pred_train, y_train_df),
            "train_mean_120": mae(mean120_train_df, y_train_df),
            "test": mae(y_pred_test, y_test_df),
            "test_mean_120": mae(mean120_test_df, y_test_df),
        },
        # "rms": {
        #     "train": rms(y_pred_train, y_train_df),
        #     "test_mean_120": rms(mean120_train_df, y_train_df),
        #     "test": rms(y_pred_test, y_test_df),
        #     "test_mean_120": rms(mean120_test_df, y_test_df),
        # },
        "rmse": {
            "train": mse(y_pred_train, y_train_df, squared=False),
            "train_mean_120": mse(mean120_train_df, y_train_df, squared=False),
            "test": mse(y_pred_test, y_test_df, squared=False),
            "test_mean_120": mse(mean120_test_df, y_test_df, squared=False),
        },
        "r2_score": {
            "train": r2_score(y_pred_train, y_train_df),
            "train_mean_120": r2_score(mean120_train_df, y_train_df),
            "test": r2_score(y_pred_test, y_test_df),
            "test_mean_120": r2_score(mean120_test_df, y_test_df),
        },
    }

    return metrics


def normalized_transform(train_df, ticks_df):
    """This takes the train_df and transform it to add ratios, WoM, and then turns everything
    into ticks and then normalizes everything"""
    # lets now try using ticks and total average? so mean_ticks / total_mean_ticks
    train_df = train_df.dropna()
    train_df = train_df[(train_df["mean_120"] > 1.1) & (train_df["mean_120"] <= 50)]
    train_df = train_df[train_df["mean_14400"] > 0]
    train_df = train_df.drop(train_df[train_df["std_2700"] > 1].index)  # slight hack
    # These above must match the test_analysis - kind of annoying I know ...

    # find wom columns
    lay_wom_list = []
    back_wom_list = []
    for column in train_df.columns:
        if "RWoML" in column:
            lay_wom_list.append(column)
        elif "RWoMB" in column:
            back_wom_list.append(column)

    # compute ratios and add them to train_df
    for i in range(len(lay_wom_list)):
        timie = lay_wom_list[i].split("_")[1]
        train_df["WoM_ratio_{}".format(timie)] = (
            train_df[lay_wom_list[i]] / train_df[back_wom_list[i]]
        )
        train_df = train_df.drop([lay_wom_list[i], back_wom_list[i]], axis=1)

    # find mean and volume columns
    mean_list = []
    volume_list = []
    for column in train_df.columns:
        if "mean" in column:
            mean_list.append(column)
        elif "volume" in column:
            volume_list.append(column)

    train_df["total_volume"] = 0
    train_df["sum_mean_volume"] = 0
    # compute ratios and add them to train_df
    for i in range(len(mean_list)):
        timie = lay_wom_list[i].split("_")[1]
        train_df["mean_and_volume_{}".format(timie)] = (
            train_df[mean_list[i]] * train_df[volume_list[i]]
        )
        train_df["total_volume"] += train_df[volume_list[i]]
        train_df["sum_mean_volume"] += train_df["mean_and_volume_{}".format(timie)]
        train_df = train_df.drop(["mean_and_volume_{}".format(timie)], axis=1)
    train_df["total_vwap"] = train_df["sum_mean_volume"] / train_df["total_volume"]
    train_df = train_df.drop(["sum_mean_volume"], axis=1)

    # OK now we have a total average ... we need to turn these into ticks

    total_vwap_ticks = []
    bsps_ticks = []
    mean_dict = {}

    for index, row in train_df.iterrows():
        total_vwap_ticks.append(
            ticks_df.iloc[ticks_df["tick"].sub(row["total_vwap"]).abs().idxmin()][
                "number"
            ]
        )
        bsps_ticks.append(
            ticks_df.iloc[ticks_df["tick"].sub(row["bsps"]).abs().idxmin()]["number"]
        )
    for i in range(len(mean_list)):
        timie = lay_wom_list[i].split("_")[1]
        mean_dict[timie] = []
        train_df["std_{}".format(timie)] = (
            train_df["std_{}".format(timie)] / train_df["mean_{}".format(timie)]
        )
        for index, row in train_df.iterrows():
            mean_dict[timie].append(
                ticks_df.iloc[ticks_df["tick"].sub(row[mean_list[i]]).abs().idxmin()][
                    "number"
                ]
            )

    train_df["total_vwap"] = total_vwap_ticks
    train_df["mean_120_temp"] = train_df["mean_120"]
    for key in mean_dict.keys():
        train_df["mean_{}".format(key)] = mean_dict[key]
        train_df["mean_{}".format(key)] = (
            train_df["mean_{}".format(key)] / train_df["total_vwap"]
        )
        train_df["volume_{}".format(key)] = (
            train_df["volume_{}".format(key)] / train_df["total_volume"]
        )

    try:
        train_df["bsps_temp"] = train_df[
            "bsps"
        ]  # drop this above but needed for margin
        # print(bsps_ticks)
        train_df["bsps"] = bsps_ticks
        train_df["bsps"] = train_df["bsps"] / train_df["total_vwap"]
    except:
        print("no bsps in this df")

    train_df = train_df.drop(["total_volume", "total_vwap"], axis=1)
    train_df = train_df.drop(["Unnamed: 0", "selection_ids", "market_id"], axis=1)

    return train_df


def get_strategy_files(
    onedrive: Onedrive, bsps_path="july_22_bsps", test_folder_path="horses_jul_wins"
):
    test_folder_files = os.listdir(test_folder_path)
    bsp_df = Onedrive.get_bsps(target_folder=bsps_path)
    test_analysis_df = onedrive.get_test_df(target_folder="Analysis_files")
    ticks_df = onedrive.get_folder_contents(
        target_folder="ticks", target_file="ticks.csv"
    )
    market_files = deque(
        [
            os.path.join(test_folder_path, f_name)
            for f_name in test_folder_files
            if float(f_name) in bsp_df["EVENT_ID"].values
        ]
    )
    return bsp_df, test_analysis_df, market_files, ticks_df


def preprocess_test_analysis(test_analysis_df):
    test_analysis_df = test_analysis_df.dropna()
    test_analysis_df = test_analysis_df[
        (test_analysis_df["mean_120"] <= 50) & (test_analysis_df["mean_120"] > 1.1)
    ]
    test_analysis_df = test_analysis_df[test_analysis_df["mean_14400"] > 0]
    test_analysis_df = test_analysis_df.drop(
        test_analysis_df[test_analysis_df["std_2700"] > 1].index
    )

    test_analysis_df_y = pd.DataFrame().assign(
        market_id=test_analysis_df["market_id"],
        selection_ids=test_analysis_df["selection_ids"],
        bsps=test_analysis_df["bsps"],
    )

    return test_analysis_df, test_analysis_df_y
