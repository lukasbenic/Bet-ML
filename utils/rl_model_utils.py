import os
from typing import Any, Dict, List, Tuple
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from utils.regressor_model_utils import create_best_model


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
    Given a DataFrame X, returns the total number of timesteps (i.e., length of the episode)
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
    Given a DataFrame X, returns the total number of timesteps (i.e., length of the episode).

    Args:
        X: A pandas DataFrame representing the state.

    Returns:
        The total number of timesteps.

    """
    timepoints = get_timepoints(X)
    timesteps = len(timepoints) * len(X)

    return timesteps


def get_low(X: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame X, returns an array containing the minimum value of each feature used in
    the state representation.

    Args:
        X: A pandas DataFrame representing the state.

    Returns:
        An array containing the minimum value of each feature used in the state representation.

    """
    features = ["mean", "std", "volume", "RWoML", "RWoMB"]
    low_values = [
        X.filter(regex=f"^{feature}_").min().min() for feature in features
    ] + [0, 0, 0]
    return np.array(low_values, dtype=np.float64)


def get_high(X: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame X, returns an array containing the maximum value of each feature used in
    the state representation.

    Args:
        X: A pandas DataFrame representing the state.

    Returns:
        An array containing the maximum value of each feature used in the state representation.

    """
    features = ["mean", "std", "volume", "RWoML", "RWoMB"]
    high_values = [
        X.filter(regex=f"^{feature}_").max().max() for feature in features
    ] + [0, 0, 0]
    return np.array(high_values, dtype=np.float64)


def train_tp_regressors(
    X: pd.DataFrame, y: pd.DataFrame, model_name: str, tpr_dir: str
) -> Dict[str, Any]:
    """
    Trains timepoint-specific regressors using the provided data, and saves each trained regressor to a file.

    Parameters:
        X (pd.DataFrame): The input features for training the regressors.
        y (pd.DataFrame): The target values for training the regressors.
        model_name (str): The name of the model to be used for training.
        tpr_dir (str): The directory to save the trained regressors.

    Returns:
        tp_regressors (Dict[str, Any]): A dictionary of trained regressor models, with keys representing timepoints and values representing the trained models.
    """
    timepoints = get_timepoints(X)
    tp_regressors = {}

    for timepoint in timepoints:
        cols_to_use = [col for col in X.columns if int(col.split("_")[-1]) >= timepoint]
        X_train = X[cols_to_use]
        y_train = y.values.ravel()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
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
        tp_regressors[f"{timepoint}"] = model

    return tp_regressors


def get_tp_regressors(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str,
    tpr_dir: str = "RL/timepoint_regressors",
) -> Dict[str, Any]:
    """
    Loads timepoint-specific regressors from a directory, or trains them if the directory is empty.

    Parameters:
        X (pd.DataFrame): The input features for training the regressors.
        y (pd.DataFrame): The target values for training the regressors.
        model_name (str): The name of the model to be used for training.
        tpr_dir (str): The directory containing the trained regressors.

    Returns:
        tp_regressors (Dict[str, Any]): A dictionary of trained regressor models, with keys representing timepoints and values representing the trained models.
    """
    if len(os.listdir(tpr_dir)) <= 0:
        tp_regressors = train_tp_regressors(
            X=X, y=y, model_name=model_name, tpr_dir=tpr_dir
        )
        return tp_regressors

    tp_regressors = {
        f"{regressor.split('_')[1].split('.')[0]}": joblib.load(
            f"{tpr_dir}/{regressor}"
        )
        for regressor in os.listdir(tpr_dir)
    }

    return tp_regressors
