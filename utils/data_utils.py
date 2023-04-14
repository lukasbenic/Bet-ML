import os
from typing import Any, Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from onedrive import Onedrive


def get_train_data_normalized(
    x_train_path: str,
    y_train_path: str,
    onedrive: Onedrive,
    ticks_df: pd.DataFrame,
    regression: bool = True,
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


# 2models and 2results uses this
def preprocess2(df: pd.DataFrame):
    df.dropna(inplace=True)
    df = df[df["mean_14400"] < 30]
    df = df[df["mean_14400"] > 0.5]
    df = df[(df["mean_120"] > 1.1)]
    # df = df[(df["mean_120"] <= 70)]
    df = df[df["mean_14400"] > 0]
    df = df.drop(df[df["std_2700"] > 1].index)  # slight hack

    # df = df.clip(lower=df.quantile(0.05), upper=df.quantile(0.95), axis=1)

    # df = df[df["mean_14400"] < 30]
    # df = df[df["mean_14400"] > 0.5]

    return df


# does not work well with KNeighborsRegressor
# works well with BayesianRidge, Ridge (400 trials tuning)
def preprocess(df: pd.DataFrame):
    def get_large_value_rows_mask(df):
        inf_columns = []
        for column in df.columns:
            if (
                df[column].isin([np.inf, -np.inf]).any()
                or df[column].max() > np.finfo(np.float64).max
            ):
                inf_columns.append(column)

        mask = (df[inf_columns] >= np.finfo(np.float64).max).any(axis=1)
        return mask

    df.fillna(0, inplace=True)
    df = df[(df["mean_120"] > 1.1)]
    # df = df[df["mean_14400"] > 0.5]
    # df = df[(df["mean_120"] > 1.1)]
    # df = df[(df["mean_120"] <= 70)]
    df = df[df["mean_14400"] > 0]
    df = df.drop(df[df["std_2700"] > 1].index)  # slight hack

    # df = df.clip(lower=df.quantile(0.05), upper=df.quantile(0.95), axis=1)

    # df = df[df["mean_14400"] < 30]
    # df = df[df["mean_14400"] > 0.5]
    # Find the rows with large values
    large_value_rows_mask = get_large_value_rows_mask(df)

    # Drop the rows with large values
    df = df[~large_value_rows_mask]

    return df


def get_train_data(
    onedrive: Onedrive,
    x_train_path: str = "utils/data/X_train.csv",
    y_train_path: str = "utils/data/y_train.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the train data from the given paths or fetch and process it if not available.

    Args:
        onedrive (Onedrive): The Onedrive instance for fetching data.
        x_train_path (str, optional): The path to the X_train data. Defaults to "utils/data/X_train.csv".
        y_train_path (str, optional): The path to the y_train data. Defaults to "utils/data/y_train.csv".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Returns the X_train and y_train DataFrames.
    """
    X_train = (
        pd.read_csv(x_train_path, index_col=False)
        if os.path.exists(x_train_path)
        else None
    )
    y_train = (
        pd.read_csv(y_train_path, index_col=False)
        if os.path.exists(y_train_path)
        else None
    )

    # Pre-process data
    if X_train is None or y_train is None:
        print(
            "x_train_df and/or y_train_df not found, commencing fetch and normalization..."
        )
        train_df = onedrive.get_train_df()
        train_df = preprocess(train_df)
        train_df.drop(
            ["Unnamed: 0", "selection_ids", "market_id"], axis=1, inplace=True
        )
        y_train = train_df["bsps"]
        X_train = train_df.drop(["bsps"], axis=1)

        X_train.to_csv(x_train_path, index=False)
        y_train.to_csv(y_train_path, index=False)

    return X_train, y_train


def get_test_data(
    onedrive: Onedrive,
    x_test_path: str = "utils/data/X_test.csv",
    y_test_path: str = "utils/data/y_test.csv",
):
    """
    Get the test data from the given paths or fetch and process it if not available.

    Args:
        onedrive (Onedrive): The Onedrive instance for fetching data.
        x_test_path (str, optional): The path to the X_test data. Defaults to "utils/data/X_test.csv".
        y_test_path (str, optional): The path to the y_test data. Defaults to "utils/data/y_test.csv".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Returns the X_test and y_test DataFrames.
    """
    X_test = (
        pd.read_csv(x_test_path, index_col=False)
        if os.path.exists(x_test_path)
        else None
    )
    y_test = (
        pd.read_csv(y_test_path, index_col=False)
        if os.path.exists(y_test_path)
        else None
    )
    if X_test is None or y_test is None:
        test_df = onedrive.get_test_df()
        test_df = preprocess(test_df)
        test_df.drop(["Unnamed: 0", "selection_ids", "market_id"], axis=1, inplace=True)
        y_test = test_df["bsps"]
        X_test = test_df.drop(["bsps"], axis=1)
        X_test.to_csv(x_test_path, index=False)
        y_test.to_csv(y_test_path, index=False)

    return X_test, y_test


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


def preprocess_test_analysis(test_analysis_df):
    test_analysis_df = preprocess(test_analysis_df)

    # test_analysis_df = test_analysis_df[test_analysis_df["mean_14400"] < 30]
    # test_analysis_df = test_analysis_df[test_analysis_df["mean_14400"] > 0.5]

    test_analysis_df_y = pd.DataFrame().assign(
        market_id=test_analysis_df["market_id"],
        selection_ids=test_analysis_df["selection_ids"],
        bsps=test_analysis_df["bsps"],
    )

    return test_analysis_df, test_analysis_df_y


def preprocess_test_analysis_normalized(test_analysis_df):
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


def visualize_data(onedrive) -> None:
    """
    Visualize the training data using histograms, box plots, and density plots.

    Args:
        X_train (pd.DataFrame): The DataFrame containing the training features.
        y_train (pd.DataFrame): The DataFrame containing the training target variable.
    """
    x_train_df, y_train_df = get_train_data(onedrive)
    train_df = pd.concat([x_train_df, y_train_df], axis=1)

    # Calculate correlation matrix
    corr_matrix = train_df.corr()

    # Visualize heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()
