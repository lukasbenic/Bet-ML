import copy
from typing import Any, Dict, Tuple, Union
import joblib
import numpy as np
import os
from pprint import pprint
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
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
    mean_absolute_error as mae,
    mean_squared_error as mse,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from ensemble_regressor import EnsembleRegressor

from onedrive import Onedrive
from utils.constants import KELLY_PERCENT, regression_models
from utils.data_utils import get_test_data, get_train_data


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


def train_test_model(
    onedrive: Onedrive,
    model_name: str,
    model_path: str = "models/",
) -> Tuple[Any, Any, Any]:
    """
    Train and test the specified model, and save the results if specified.

    Args:
        ticks_df (pd.DataFrame): The ticks DataFrame used for normalization.
        onedrive (Onedrive): The Onedrive object for loading data.
        model_name (str): The name of the model to train and test.
        regression (bool, optional): Whether the model is a regression model. Defaults to True.
        x_train_path (str, optional): Path to the saved x_train DataFrame. Defaults to "utils/x_train_df.csv".
        y_train_path (str, optional): Path to the saved y_train DataFrame. Defaults to "utils/y_train_df.csv".
        model_path (str, optional): Path to the saved model. Defaults to "models/".

    Returns:
        Tuple[Any, Any, Any]: The trained model, column names, and scaler used for feature scaling.
    """
    X_train, y_train = get_train_data(onedrive)
    clm = X_train.columns
    scaler = StandardScaler()

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=clm)
    y_train = y_train.values.ravel()

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
    if not model_name in ["Ensemble", "LinearRegression"]:
        n_trials = 400  # temp
        if model_name in [
            "GradientBoostingRegressor",
            "RandomForestRegressor",
            "GaussianProcessRegressor",
            "SVR",
            "XGBRegressor",
            "KNeighborsRegressor",
        ]:
            n_trials = 25
        # elif model_name == "KNeighborsRegressor":
        #     n_trials = 40

        # Create Optuna study and optimize hyperparameters
        study = optuna.create_study(sampler=TPESampler(seed=42), direction="maximize")
        # print("scorers", sorted(sklearn.metrics.SCORERS.keys()))
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, model_name),
            n_trials=n_trials,
        )

        # Get the best model
        best_params = study.best_trial.params
        print(best_params, type(best_params))
        # save the best params in a dict for writing diss
        pd.DataFrame.from_records([best_params], index=[0]).to_csv(
            f"{model_path}best_params/{model_name}_best_params.csv"
        )

    best_model = create_best_model(best_params, model_name, X_train, y_train)

    # m.fit(X_train, y_train_df.values.ravel())
    _ = joblib.dump(best_model, f"{model_path}{model_name}.pkl")
    print("Model successfully trained")

    metrics = test_model(
        onedrive, scaler, clm, best_model, X_train, y_train, model_name
    )

    return best_model, clm, scaler


def test_model(
    onedrive: Onedrive,
    scaler,
    clm,
    model,
    X_train,
    y_train,
    model_name,
    model_path: str = "models/",
    save=True,
):
    X_test, y_test = get_test_data(onedrive)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=clm)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "mse": {
            "train": mse(y_train, y_pred_train),
            "train_mean_120": mse(X_train["mean_120"], y_pred_train),
            "test": mse(y_test, y_pred_test),
            "test_mean_120": mse(X_test["mean_120"], y_pred_test),
        },
        "mae": {
            "train": mae(y_train, y_pred_train),
            "train_mean_120": mae(X_train["mean_120"], y_pred_train),
            "test": mae(y_test, y_pred_test),
            "test_mean_120": mae(X_test["mean_120"], y_pred_test),
        },
        # "rms": {
        #     "train": rms(y_train_df, y_pred_train),
        #     "test_mean_120": rms(y_train_df, X_train["mean_120"]),
        #     "test": rms(y_pred_test, y_test),
        #     "test_mean_120": rms(y_test, X_test["mean_120"]),
        # },
        "rmse": {
            "train": mse(y_train, y_pred_train, squared=False),
            "train_mean_120": mse(X_train["mean_120"], y_pred_train, squared=False),
            "test": mse(y_test, y_pred_test, squared=False),
            "test_mean_120": mse(X_test["mean_120"], y_pred_test, squared=False),
        },
        "r2_score": {
            "train": r2_score(y_train, y_pred_train),
            "train_mean_120": r2_score(X_train["mean_120"], y_pred_train),
            "test": r2_score(y_test, y_pred_test),
            "test_mean_120": r2_score(X_test["mean_120"], y_pred_test),
        },
    }
    if save:
        metrics_df = pd.DataFrame.from_dict(metrics)
        metrics_df.to_csv(f"{model_path}metrics/{model_name}_metrics.csv")

    return metrics


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
        model = LinearRegression()
    elif model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(**best_params)
    elif model_name == "Ensemble":
        model_files = [f for f in os.listdir("models") if ".pkl" in f]
        print(model_files)
        ensemble_models = [joblib.load(f"models/{model}") for model in model_files]
        print("Models used for Ensemble")
        [print(model) for model in model_files]
        model = EnsembleRegressor(ensemble_models)
    else:
        raise ValueError(f"Invalid model name '{model_name}'")

    # Train the best model with the best parameters
    model.fit(x_train_df, y_train_df)
    return model


def objective(
    trial: Trial,
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str,
    metric: str = "neg_mean_squared_error",
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
        n_estimators = trial.suggest_int(
            "n_estimators", 150, 300
        )  # Focused on a narrower range for faster training and better generalization
        max_depth = trial.suggest_int(
            "max_depth", 10, 30, log=True
        )  # Reduced the upper limit to prevent overfitting
        min_samples_split = trial.suggest_int(
            "min_samples_split", 5, 15
        )  # Wider range for better generalization
        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf", 2, 10
        )  # Wider range for better generalization
        max_features = trial.suggest_int(
            "max_features", 30, len(X.columns) // 2
        )  # Reduced upper limit to prevent overfitting and faster training
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
