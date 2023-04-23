import os
from typing import Any, Dict, List, Tuple
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from onedrive import Onedrive
from utils.data_utils import get_test_data

from utils.regressor_model_utils import create_best_model
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse,
    r2_score,
)


def get_timepoints(X: pd.DataFrame) -> List[int]:
    """
    Given a DataFrame X, returns a list of unique timepoints found in the column names in
    descending order.

    Args:
        X: A pandas DataFrame representing the state.

    Returns:
        A list of unique timepoints.

    """
    unique_timepoints = set()
    for col in X.columns:
        if col not in ["predicted_bsp", "lay", "back"]:
            timepoint = int(col.split("_")[-1])
            unique_timepoints.add(timepoint)

    timepoints = sorted(list(unique_timepoints), reverse=True)
    return timepoints


def get_timesteps_and_timepoints(X: pd.DataFrame) -> Tuple[int, List[int]]:
    """
    Given a DataFrame X, returns the total number of timesteps
    and a list of unique timepoints in descending order.

    Args:
        X: A pandas DataFrame representing the state.

    Returns:
        A tuple containing the total number of timesteps and a list of unique timepoints.

    """
    timepoints = get_timepoints(X)
    timesteps = len(timepoints) * len(X)

    return timesteps, timepoints


def get_timesteps(X: pd.DataFrame) -> int:
    """
    Given a DataFrame X, returns the total number of timesteps
    Args:
        X: A pandas DataFrame representing the state.

    Returns:
        The total number of timesteps.

    """
    timepoints = get_timepoints(X)
    timesteps = len(timepoints) * len(X)

    return timesteps


def get_low(X: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame X, returns an array containing the minimum value of each feature used in
    the state representation.

    Args:
        X: A pandas DataFrame representing the state.
        y: A pandas DataFrame representing the target variable.

    Returns:
        An array containing the minimum value of each feature used in the state representation.
    """
    high_values = get_high(X, y)
    print("high", high_values)
    min_bsp = y.min(axis=1).min()
    min_bsp = min_bsp if min_bsp >= 0 else min_bsp * 0.7
    features = ["mean", "std", "volume", "RWoML", "RWoMB"]
    low_values = [
        X.filter(regex=f"^{feature}_").min().min() for feature in features
    ] + [
        min_bsp,
        -high_values[0] * 1.3,
        -high_values[0] * 1.3,
    ]  # set min predicted bsp to be 30% lower than bsps found and back and lay to max negative mean * 1.3
    print("low", low_values)
    return np.array(low_values, dtype=np.float64)


def get_high(X: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame X, returns an array containing the maximum value of each feature used in
    the state representation.

    Args:
        X: A pandas DataFrame representing the state.
        y: A pandas DataFrame representing the target variable.

    Returns:
        An array containing the maximum value of each feature used in the state representation.
    """
    features = ["mean", "std", "volume", "RWoML", "RWoMB"]
    high_bsp = y.max(axis=1).max()
    high_values = [X.filter(regex=f"^{feature}_").max().max() for feature in features]
    high_values = high_values + [
        high_bsp * 1.3,
        high_values[0] * 1.3,
        high_values[0] * 1.3,
    ]  # set back and lay to highest mean seen / might want to set to bsp , add factor of 30% incase predictions from regressor

    return np.array(high_values, dtype=np.float64)


def get_sorted_columns(columns, current_timepoint):
    column_order = ["mean", "std", "volume", "RWoML", "RWoMB"]

    relevant_columns = [
        col
        for col in columns
        if col.split("_")[-1].isdigit()
        and int(col.split("_")[-1]) >= int(current_timepoint)
    ]

    sorted_columns = sorted(
        relevant_columns,
        key=lambda x: (int(x.split("_")[-1]), column_order.index(x.split("_")[0])),
    )

    return sorted_columns


def train_tp_regressors(
    X: pd.DataFrame, y: pd.DataFrame, model_name: str, tpr_dir: str, scaler_dir: str
) -> Dict[str, Any]:
    """
    Trains timepoint-specific regressors using the provided data, and saves each trained regressor to a file.

    Parameters:
        X (pd.DataFrame): The input features for training the regressors.
        y (pd.DataFrame): The target values for training the regressors.
        model_name (str): The name of the model to be used for training.
        tpr_dir (str): The directory to save the trained regressors.
        scaler_dir (str): The directory to save the trained regressor's scaler.


    Returns:
        tp_regressors (Dict[str, Any]): A dictionary of trained regressor models, with keys representing timepoints and values representing the trained models.
    """
    timepoints = get_timepoints(X)
    tp_regressors = {}

    for timepoint in timepoints:
        relevant_cols = get_sorted_columns(X.columns, timepoint)
        X_train = X[relevant_cols]
        y_train = y.values.ravel()
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=relevant_cols
        )
        # use best params obtained from training previous regressors
        best_params = (
            pd.read_csv(f"models/best_params/{model_name}_best_params.csv")
            .drop(["Unnamed: 0"], axis=1)
            .to_dict(orient="records")[0]
        )
        # Train a model on the data
        model = create_best_model(best_params, model_name, X_train_scaled, y_train)

        # Save the model to a file
        joblib.dump(model, f"{tpr_dir}/{model_name}_{timepoint}.pkl")
        joblib.dump(scaler, f"{scaler_dir}/{model_name}_{timepoint}_scaler.pkl")
        tp_regressors[f"{timepoint}"] = {"model": model, "scaler": scaler}

    return tp_regressors


def get_tp_regressors(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str,
    tpr_dir: str = "RL/timepoint_regressors/models",
    scaler_dir: str = "RL/timepoint_regressors/scalers",
) -> Dict[str, Any]:
    """
    Loads timepoint-specific regressors and scalers from directories, or trains them if the directories are empty.

    Parameters:
        X (pd.DataFrame): The input features for training the regressors.
        y (pd.DataFrame): The target values for training the regressors.
        model_name (str): The name of the model to be used for training.
        tpr_dir (str): The directory containing the trained regressors.
        scaler_dir (str): The directory containing the trained scalers.

    Returns:
        tp_regressors (Dict[str, Any]): A dictionary of trained regressor models, with keys representing timepoints and values representing the trained models.
    """
    if len(os.listdir(tpr_dir)) <= 0:
        tp_regressors = train_tp_regressors(
            X=X, y=y, model_name=model_name, tpr_dir=tpr_dir, scaler_dir=scaler_dir
        )
        return tp_regressors

    tp_regressors = {}
    for regressor in os.listdir(tpr_dir):
        timepoint = regressor.split("_")[1].split(".")[0]
        model = joblib.load(f"{tpr_dir}/{regressor}")
        scaler = joblib.load(f"{scaler_dir}/{model_name}_{timepoint}_scaler.pkl")
        tp_regressors[timepoint] = {"model": model, "scaler": scaler}

    return tp_regressors


def test_models(
    onedrive: Onedrive,
    tp_regressors: Dict[str, Any],
    X_train,
    y_train,
    save_path: str = "RL/timepoint_regressors/metrics",
    save=True,
):
    X_test, y_test = get_test_data(onedrive)

    train_metrics_df = pd.DataFrame(columns=["MSE", "MAE", "RMSE", "R2 Score"])
    test_metrics_df = pd.DataFrame(columns=["MSE", "MAE", "RMSE", "R2 Score"])

    for timepoint, tp_regressor in tp_regressors.items():
        scaler = tp_regressor["scaler"]
        model = tp_regressor["model"]

        # Select relevant columns based on the timepoint regressor name
        relevant_cols = get_sorted_columns(X_train.columns, timepoint)

        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train[relevant_cols]), columns=relevant_cols
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test[relevant_cols]), columns=relevant_cols
        )

        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        train_metrics_df = pd.concat(
            [
                train_metrics_df,
                pd.DataFrame(
                    {
                        "MSE": [mse(y_train, y_pred_train)],
                        "MAE": [mae(y_train, y_pred_train)],
                        "RMSE": [mse(y_train, y_pred_train, squared=False)],
                        "R2 Score": [r2_score(y_train, y_pred_train)],
                    },
                    index=[timepoint],
                ),
            ]
        )

        test_metrics_df = pd.concat(
            [
                test_metrics_df,
                pd.DataFrame(
                    {
                        "MSE": [mse(y_test, y_pred_test)],
                        "MAE": [mae(y_test, y_pred_test)],
                        "RMSE": [mse(y_test, y_pred_test, squared=False)],
                        "R2 Score": [r2_score(y_test, y_pred_test)],
                    },
                    index=[timepoint],
                ),
            ]
        )

    if save:
        train_metrics_df.index = train_metrics_df.index.astype(int)
        train_metrics_df.sort_index(inplace=True)
        train_metrics_df.to_csv(
            f"{save_path}/train_metrics.csv", index_label="Regressor"
        )

        test_metrics_df.index = test_metrics_df.index.astype(int)
        test_metrics_df.sort_index(inplace=True)
        test_metrics_df.to_csv(f"{save_path}/test_metrics.csv", index_label="Regressor")
