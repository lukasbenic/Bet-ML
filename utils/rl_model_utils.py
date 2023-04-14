from typing import List, Tuple
import numpy as np
import pandas as pd


def get_timepoints(X: pd.DataFrame) -> List[int]:
    unique_timepoints = set()
    for col in X.columns:
        if col not in ["lay", "back"]:
            timepoint = int(col.split("_")[-1])
            unique_timepoints.add(timepoint)

    timepoints = sorted(list(unique_timepoints), reverse=True)
    return timepoints


def get_timesteps_and_timepoints(X: pd.DataFrame) -> Tuple[int, List[int]]:
    timepoints = get_timepoints(X)
    timesteps = len(timepoints) * len(X)

    return timesteps, timepoints


def get_timesteps(X: pd.DataFrame) -> int:
    timepoints = get_timepoints(X)
    timesteps = len(timepoints) * len(X)

    return timesteps


def get_low(X: pd.DataFrame) -> np.ndarray:
    features = ["mean", "std", "vol", "RWoML", "RWoMB"]
    low_values = []

    for feature in features:
        filtered_cols = X.filter(regex=f"^{feature}_").columns.tolist()
        low_values.append(X[filtered_cols].min().values)

    mean_low = X.filter(regex="^mean_").min().values
    low_values.extend([mean_low, mean_low])  # Set low for lay and back as the mean low
    low = np.hstack(low_values).astype(np.float64)
    print("low", low.shape)

    return low


def get_high(X: pd.DataFrame) -> np.ndarray:
    features = ["mean", "std", "vol", "RWoML", "RWoMB"]
    high_values = []

    for feature in features:
        filtered_cols = X.filter(regex=f"^{feature}_").columns.tolist()
        high_values.append(X[filtered_cols].max().values)

    mean_high = X.filter(regex="^mean_").max().values
    high_values.extend(
        [mean_high, mean_high]
    )  # Set high for lay and back as the mean high
    high = np.hstack(high_values).astype(np.float64)
    print("high", high.shape)

    return high
