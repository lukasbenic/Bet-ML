import copy
from collections import deque
from typing import Any, Dict, Tuple, Union
import joblib
import numpy as np
import os
from pprint import pprint
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import torch
from xgboost import XGBRegressor
import yaml
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
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
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from ensemble_regressor import EnsembleRegressor

from onedrive import Onedrive
from utils.constants import KELLY_PERCENT, regression_models


def objective(
    trial: Trial,
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str,
    metric: str = "r2",
) -> float:
    """
    Optuna objective function for hyperparameter optimization of various regression models.

    :param trial: Optuna trial object for the optimization process
    :param X: DataFrame containing the training features
    :param y: DataFrame containing the training target variable
    :param model_name: String representing the name of the model to be optimized
    :param metric: String representing the evaluation metric to be optimized (default: 'neg_mean_squared_error')
    :return: Mean cross-validated score of the model based on the selected metric
    """
    if model_name == "Ridge":
        alpha = trial.suggest_float("alpha", 0.001, 100, log=True)
        model = Ridge(alpha=alpha)
    elif model_name == "XGBRegressor":
        learning_rate = trial.suggest_float("learning_rate", 0.001, 1.0, log=True)
        max_depth = trial.suggest_int("max_depth", 1, 10)
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        model = XGBRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            objective="reg:squarederror",
        )
    elif model_name == "KNeighborsRegressor":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    elif model_name == "Lasso":
        alpha = trial.suggest_float("alpha", 0.001, 100, log=True)
        model = Lasso(alpha=alpha)
    elif model_name == "ElasticNet":
        alpha = trial.suggest_float("alpha", 0.001, 100, log=True)
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
        n_estimators = trial.suggest_int("n_estimators", 100, 2000)
        max_depth = trial.suggest_int("max_depth", 5, 50, log=True)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_int("max_features", 1, len(X.columns))
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
    elif model_name == "GradientBoostingRegressor":
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.2, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        )
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
        )
    elif model_name == "SVR":
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 0.001, 0.1, log=True)
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "rbf", "poly", "sigmoid"]
        )
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        model = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)

    elif model_name == "GaussianProcessRegressor":
        kernel_type = trial.suggest_categorical(
            "kernel",
            ["RBF", "Matern", "RationalQuadratic", "ExpSineSquared", "DotProduct"],
        )
        alpha = trial.suggest_float("alpha", 1e-15, 1e-10, log=True)

        if kernel_type == "RBF":
            length_scale = trial.suggest_float("length_scale", 1e-2, 1e2, log=True)
            kernel = RBF(length_scale=length_scale)
        elif kernel_type == "Matern":
            length_scale = trial.suggest_float("length_scale", 1e-2, 1e2, log=True)
            nu = trial.suggest_float("nu", 0.5, 2.5)
            kernel = Matern(length_scale=length_scale, nu=nu)
        elif kernel_type == "RationalQuadratic":
            length_scale = trial.suggest_float("length_scale", 1e-2, 1e2, log=True)
            alpha = trial.suggest_float("alpha", 1e-2, 1e2, log=True)
            kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha)
        elif kernel_type == "ExpSineSquared":
            length_scale = trial.suggest_float("length_scale", 1e-2, 1e2, log=True)
            periodicity = trial.suggest_float("periodicity", 1e-2, 1e2, log=True)
            kernel = ExpSineSquared(length_scale=length_scale, periodicity=periodicity)
        elif kernel_type == "DotProduct":
            sigma_0 = trial.suggest_float("sigma_0", 1e-2, 1e2, log=True)
            kernel = DotProduct(sigma_0=sigma_0)
        else:
            raise ValueError(f"Invalid kernel type '{kernel_type}'")

        model = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, n_restarts_optimizer=3
        )
    elif model_name == "LinearRegression":
        # Specify hyperparameters for LinearRegression
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

        model = LinearRegression(fit_intercept=fit_intercept)

    elif model_name == "DecisionTreeRegressor":
        # Specify hyperparameters for DecisionTreeRegressor
        max_depth = trial.suggest_int("max_depth", 2, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2", None]
        )

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
    else:
        raise ValueError(f"Invalid model name '{model_name}'")

    # Perform cross-validation with n_jobs=-1 to use all available cores
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=1)
    return np.mean(scores)


def create_best_model(
    best_params: Dict[str, Any],
    model_name: str,
    x_train_df: pd.DataFrame,
    y_train_df: pd.DataFrame,
) -> Union[
    Ridge,
    KNeighborsRegressor,
    Lasso,
    ElasticNet,
    BayesianRidge,
    RandomForestRegressor,
    GradientBoostingRegressor,
    SVR,
    EnsembleRegressor,
]:
    """
    Create and train the best model with the best parameters found by Optuna.

    :param best_params: Dictionary containing the best hyperparameters for the chosen model
    :param model_name: String representing the name of the model to be created
    :param x_train_df: DataFrame containing the training features
    :param y_train_df: DataFrame containing the training target variable
    :return: Trained model with the best parameters
    """
    if model_name == "Ridge":
        model = Ridge(**best_params)
    elif model_name == "KNeighborsRegressor":
        model = KNeighborsRegressor(**best_params)
    elif model_name == "XGBRegressor":
        model = XGBRegressor(**best_params)
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
    elif model_name == "GaussianProcessRegressor":
        model = GaussianProcessRegressor(**best_params)
    elif model_name == "LinearRegression":
        model = LinearRegression(**best_params)
    elif model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(**best_params)
    elif model_name == "Ensemble":
        model_files = os.listdir("models")
        ensemble_models = [
            joblib.load(f"models/{model}")
            for model in model_files[1:]
            if len(model.split("_")) == 1
        ]
        print("Models used for Ensemble")
        [print(model) for model in model_files[1:] if len(model.split("_")) == 1]
        model = EnsembleRegressor(ensemble_models)
    else:
        raise ValueError(f"Invalid model name '{model_name}'")

    # Train the best model with the best parameters
    model.fit(x_train_df, y_train_df)
    return model


def rms(y_pred: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between the predicted values and the actual values.

    Args:
        y_pred (np.ndarray): The predicted values.
        y (np.ndarray): The actual values.

    Returns:
        float: The Root Mean Squared Error (RMSE) value.
    """
    rms = np.sqrt(np.mean((y - y_pred) ** 2))
    return rms


def update_tracker(tracker: dict, metrics: dict):
    for key, item in tracker.items():
        name_lst = key.split("_")
        m_key = "_".join(name_lst[1:])

        if key in ["race_counter", "balance"]:
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
    pprint(tracker)
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
    regression: bool = True,
    save: bool = True,
    x_train_path: str = "utils/x_train_df.csv",
    y_train_path: str = "utils/y_train_df.csv",
    model_path: str = "models/",
) -> Tuple[Any, Any, Any]:
    """
    Train and test the specified model, and save the results if specified.

    Args:
        ticks_df (pd.DataFrame): The ticks DataFrame used for normalization.
        onedrive (Onedrive): The Onedrive object for loading data.
        model_name (str): The name of the model to train and test.
        regression (bool, optional): Whether the model is a regression model. Defaults to True.
        save (bool, optional): Whether to save the results. Defaults to True.
        x_train_path (str, optional): Path to the saved x_train DataFrame. Defaults to "utils/x_train_df.csv".
        y_train_path (str, optional): Path to the saved y_train DataFrame. Defaults to "utils/y_train_df.csv".
        model_path (str, optional): Path to the saved model. Defaults to "models/".

    Returns:
        Tuple[Any, Any, Any]: The trained model, column names, and scaler used for feature scaling.
    """
    x_train_df, y_train_df, mean120_train_df = get_train_data(
        x_train_path, y_train_path, onedrive, ticks_df, regression
    )
    pd.set_option('display.width', 1000)
    clm = x_train_df.columns
    print("columns", clm)
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    x_train_df = pd.DataFrame(scaler.fit_transform(x_train_df), columns=clm)
    y_train_df = y_train_df.values.ravel()

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
        n_trials = 300
        if model_name in [
            "GradientBoostingRegressor",
            "GaussianProcessRegressor",
            "SVR",
            "XGBRegressor",
        ]:
            n_trials = 10
        elif model_name == "KNeighborsRegressor":
            n_trials = 30

        # Create Optuna study and optimize hyperparameters
        study = optuna.create_study(sampler=TPESampler(seed=42), direction="maximize")
        print("scorers", sorted(sklearn.metrics.SCORERS.keys()))
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
        metrics_df = pd.DataFrame.from_dict(metrics)
        metrics_df.to_csv(f"{model_path}metrics/{model_name}_metrics.csv")

    return best_model, clm, scaler


def test_model(
    ticks_df: pd.DataFrame,
    model: Any,
    scaler: Any,
    clm: pd.Index,
    x_train_df: pd.DataFrame,
    y_train_df: pd.DataFrame,
    mean120_train_df: pd.DataFrame,
    test_analysis_df: pd.DataFrame,
    regression: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Test the trained model and compute evaluation metrics.

    Args:
        ticks_df (pd.DataFrame): The ticks DataFrame used for normalization.
        model (Any): The trained model to test.
        scaler (Any): The scaler used for feature scaling.
        clm (pd.Index): The column names of the transformed DataFrame.
        x_train_df (pd.DataFrame): The training feature DataFrame.
        y_train_df (pd.DataFrame): The training target DataFrame.
        mean120_train_df (pd.DataFrame): The mean_120 column from the training DataFrame.
        test_analysis_df (pd.DataFrame): The test analysis DataFrame.
        regression (bool, optional): Whether the model is a regression model. Defaults to True.

    Returns:
        Dict[str, Dict[str, float]]: The evaluation metrics for the model.
    """

    # Filter and clean test_analysis_df
    test_analysis_df = test_analysis_df.dropna()
    test_analysis_df = test_analysis_df[
        (test_analysis_df["mean_120"] <= 50) & (test_analysis_df["mean_120"] > 1.1)
    ]
    test_analysis_df = test_analysis_df[test_analysis_df["mean_14400"] > 0]
    # below is a slight hack ...
    test_analysis_df = test_analysis_df.drop(
        test_analysis_df[test_analysis_df["std_2700"] > 1].index
    )

    # Prepare test_analysis_df_y
    test_analysis_df_y = test_analysis_df[["market_id", "selection_ids", "bsps"]]

    # Normalize the test DataFrame
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
    x_test_df = x_test_df.drop(
        ["bsps_temp", "mean_120_temp"], axis=1
    )  # TEst dropping mean120 temp

    x_test_df = pd.DataFrame(scaler.transform(x_test_df), columns=clm)

    y_pred_train = model.predict(x_train_df)
    y_pred_test = model.predict(x_test_df)

    metrics = {
        "mse": {
            "train": mse(y_train_df, y_pred_train),
            "train_mean_120": mse(y_train_df, mean120_train_df),
            "test": mse(y_test_df, y_pred_test),
            "test_mean_120": mse(y_test_df, mean120_test_df),
        },
        "mae": {
            "train": mae(y_train_df, y_pred_train),
            "train_mean_120": mae(y_train_df, mean120_train_df),
            "test": mae(y_test_df, y_pred_test),
            "test_mean_120": mae(y_test_df, mean120_test_df),
        },
        "rms": {
            "train": rms(y_train_df, y_pred_train),
            "test_mean_120": rms(y_train_df, mean120_train_df),
            "test": rms(y_pred_test, y_test_df),
            "test_mean_120": rms(y_test_df, mean120_test_df),
        },
        "rmse": {
            "train": mse(y_train_df, y_pred_train, squared=False),
            "train_mean_120": mse(y_train_df, mean120_train_df, squared=False),
            "test": mse(y_test_df, y_pred_test, squared=False),
            "test_mean_120": mse(y_test_df, mean120_test_df, squared=False),
        },
        "r2_score": {
            "train": r2_score(y_train_df, y_pred_train),
            "train_mean_120": r2_score(y_train_df, mean120_train_df),
            "test": r2_score(y_test_df, y_pred_test),
            "test_mean_120": r2_score(y_test_df, mean120_test_df),
        },
    }
    pprint(metrics)
    return metrics


def normalized_transform(
    train_df: pd.DataFrame, ticks_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalize and transform train_df to add ratios, WoM, and then turn everything into ticks and normalize.

    Args:
        train_df (pd.DataFrame): The training DataFrame to be transformed.
        ticks_df (pd.DataFrame): The ticks DataFrame used for normalization.

    Returns:
        pd.DataFrame: The transformed and normalized training DataFrame.
    """

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
        print("No BSPS in this df.")

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


def get_train_data(
    x_train_path: str,
    y_train_path: str,
    onedrive: Onedrive,
    ticks_df: pd.DataFrame,
    regression: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get the train data from the given paths or fetch and process it if not available.

    Args:
        x_train_path (str): The path to the X_train data.
        y_train_path (str): The path to the y_train data.
        onedrive (Onedrive): The Onedrive instance for fetching data.
        ticks_df (pd.DataFrame): The DataFrame containing the ticks data.
        regression (bool): If True, perform regression. If False, perform classification.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns the X_train, y_train, and mean120_train DataFrames.
    """
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

        # NOTE might want to drop this for regression too
        if not regression:
            train_df = train_df.drop(["mean_120_temp"], axis=1)

        mean120_train_df = train_df["mean_120"]
        bsp_train_df = train_df["bsps"]
        train_df["bsps"] = ((mean120_train_df - bsp_train_df) > 0).astype(int)

        df_majority = train_df[(train_df["bsps"] == 0)]
        df_minority = train_df[(train_df["bsps"] == 1)]

        # downsample majority
        df_majority = df_majority.head(len(df_minority))

        # Combine majority class with upsampled minority class
        train_df = pd.concat([df_minority, df_majority])
        mean120_train_df = train_df["mean_120"]

        y_train_df = train_df["bsps"]
        if regression:
            y_train_df = train_df["bsps_temp"]

        y_train_df.to_csv(y_train_path, index=False)

        x_train_df = train_df.drop(["bsps"], axis=1)
        x_train_df = x_train_df.drop(["bsps_temp"], axis=1)

    # NOTE test
    x_train_df = x_train_df.drop(["mean_120_temp"], axis=1)
    return x_train_df, y_train_df, mean120_train_df


def visualize_data(onedrive) -> None:
    """
    Visualize the training data using histograms, box plots, and density plots.

    Args:
        X_train (pd.DataFrame): The DataFrame containing the training features.
        y_train (pd.DataFrame): The DataFrame containing the training target variable.
    """
    # Combine X_train and y_train for visualization
    ticks_df = onedrive.get_folder_contents(
        target_folder="ticks", target_file="ticks.csv"
    )
    x_train_df, y_train_df, _ = get_train_data(
        "utils/x_train_df.csv", "utils/y_train_df.csv", onedrive, ticks_df, True
    )
    train_df = pd.concat([x_train_df, y_train_df], axis=1)

    # Calculate correlation matrix
    corr_matrix = train_df.corr()

    # Visualize heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def calculate_stake(stake: float, price_adjusted: float, side: str) -> float:
    stake = (
        round(
            stake / (price_adjusted - 1),
            2,
        )
        if side == "LAY"
        else stake
    )
    return stake


def calculate_kelly_stake(balance: float, odds: float) -> float:
    """
    Calculates the stake using the Kelly staking method with a fixed k% of 3%.

    Args:
        balance (float): The current bank size.
        odds (float): The decimal odds of the selection.

    Returns:
        float: The stake size.
    """
    kelly_stake = (balance * KELLY_PERCENT * (odds - 1)) / (odds - 1)
    return round(kelly_stake, 2)


def calculate_odds(
    predicted_bsp: float,
    confidence_price: float,
    mean120: float,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> float:
    """
    Calculates the odds using a weighted average of the predicted BSP, confidence price, and mean120.

    Args:
        predicted_bsp (float): The predicted BSP for the runner.
        confidence_price (float): The confidence price for the runner.
        mean120 (float): The mean120 for the runner.
        alpha (float, optional): The weight to assign to the predicted BSP. Defaults to 0.5.
        beta (float, optional): The weight to assign to the weighted average of the confidence price and mean120. Defaults to 0.3.

    Returns:
        float: The calculated odds for the runner.
    """
    odds = alpha * predicted_bsp + (1 - alpha) * (
        beta * confidence_price + (1 - beta) * mean120
    )
    return odds


def calculate_gambled(side: str, size_matched: float, price: float) -> float:
    """
    Calculates the amount gambled based on the side, size matched and price.

    Args:
        side (str): The side of the bet ("BACK" or "LAY").
        size_matched (float): The size matched of the order.
        price (float): The price of the order.

    Returns:
        float: The green amount gambled.
    """
    if side == "BACK":
        return size_matched
    else:
        return size_matched * (price - 1)


def calculate_margin(side: str, size: float, price: float, bsp_value: float) -> float:
    """
    Calculates the margin of the bet based on the side, size, price and BSP value.

    Args:
        side (str): The side of the bet ("BACK" or "LAY").
        size (float): The size of the order.
        price (float): The price of the order.
        bsp_value (float): The BSP value.

    Returns:
        float: The margin of the bet.
    """
    if side == "BACK":
        return size * (price - bsp_value) / price
    else:
        return size * (bsp_value - price) / price
