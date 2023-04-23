import numpy as np
import os
from collections import deque
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pprint import pprint
from onedrive import Onedrive
from scipy import stats


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
    # Profit after commission, 5% after each market on profit
    if metrics["m_c_margin"] + metrics["m_i_margin"] > 0:
        if len(tracker["actual_profit_plotter"]) > 0:
            tracker["actual_profit_plotter"].append(
                (0.95 * (metrics["m_c_margin"] + metrics["m_i_margin"]))
                + tracker["actual_profit_plotter"][-1]
            )
        # case of no entry
        else:
            tracker["actual_profit_plotter"].append(
                (0.95 * (metrics["m_c_margin"] + metrics["m_i_margin"]))
            )

    # loss, no commission
    else:
        if len(tracker["actual_profit_plotter"]) > 0:
            tracker["actual_profit_plotter"].append(
                ((metrics["m_c_margin"] + metrics["m_i_margin"]))
                + tracker["actual_profit_plotter"][-1]
            )
        # case of no enntry
        else:
            tracker["actual_profit_plotter"].append(
                ((metrics["m_c_margin"] + metrics["m_i_margin"]))
            )

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
            # edit this to show profit after comission
            if "expected" in key:
                item.append(tracker["total_m_c_margin"] + tracker["total_m_i_margin"])
            if "green" in key:
                item.append(tracker["total_green_margin"])

    tracker["race_counter"] += 1
    pprint(tracker)
    return tracker


def calculate_performance_metrics(tracker):
    total_bets = tracker["total_matched_correct"] + tracker["total_matched_incorrect"]
    winning_bets = tracker["total_matched_correct"]
    losing_bets = tracker["total_matched_incorrect"]
    net_profit = tracker["actual_profit_plotter"][-1]
    total_amount_gambled = tracker["total_amount_gambled"]

    metrics = {
        "total_bets": total_bets,
        "winning_bets": winning_bets,
        "losing_bets": losing_bets,
    }

    # ROI
    roi = net_profit / total_amount_gambled
    metrics["roi"] = roi

    # Average win/loss
    profits = np.diff([0] + tracker["actual_profit_plotter"])
    avg_win = sum([x for x in profits if x > 0]) / winning_bets
    avg_loss = sum([x for x in profits if x < 0]) / losing_bets
    metrics["avg_win"] = avg_win
    metrics["avg_loss"] = avg_loss

    # Max Drawdown (MDD) and Max Run-up (MRU)
    cum_profit = np.array(tracker["actual_profit_plotter"])
    running_max = np.maximum.accumulate(cum_profit)
    running_min = np.minimum.accumulate(cum_profit)
    mdd = np.abs(running_max - cum_profit).max()
    mru = np.abs(running_min - cum_profit).max()
    metrics["mdd"] = mdd
    metrics["mru"] = mru

    # Risk Adjusted Rate of Return (RAR)
    rar = (net_profit - mdd) / total_amount_gambled
    metrics["rar"] = rar

    # Pessimistic return on margin (PROM)
    prom = (net_profit - 2 * mdd) / total_amount_gambled
    metrics["prom"] = prom

    # Perfect Profit (PP)
    pp = sum(profits)
    metrics["pp"] = pp

    # Strategy Efficiency (SE)
    se = net_profit / pp
    metrics["se"] = se

    # T-test
    t_stat, p_value = stats.ttest_1samp(profits, 0)
    metrics["t_stat"] = t_stat
    metrics["p_value"] = p_value

    return metrics


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


def calculate_stake(stake: float, price_adjusted: float, side: str) -> float:
    """Calculate the stake amount for a given price adjustment and side.

    Args:
        stake (float): The desired stake amount.
        price_adjusted (float): The adjusted price for the bet.
        side (str): The side of the bet, either "LAY" or "BACK".

    Returns:
        float: The stake amount adjusted for the price.
    """
    stake = (
        round(
            stake / (price_adjusted - 1),
            2,
        )
        if side == "LAY"
        else stake
    )
    return stake



def kelly_stake(p, odds, bankroll):
    """
    Calculate the optimal bet size using the Kelly criterion.

    :param p: Probability of winning (0 <= p <= 1)
    :param odds: Decimal odds (odds >= 1)
    :param bankroll: Available capital (bankroll > 0)
    :return: Optimal bet size
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if odds < 1:
        raise ValueError("Odds must be greater than or equal to 1.")
    if bankroll <= 0:
        raise ValueError("Bankroll must be greater than 0.")

    # Calculate the optimal fraction to bet
    b = odds - 1
    q = 1 - p
    optimal_fraction = (b * p - q) / b

    # Calculate the optimal bet size
    optimal_bet = bankroll * optimal_fraction

    return optimal_bet


def calculate_odds2(
    predicted_bsp: float,
    confidence_price: float,
    mean120: float,
    std_120: float,
    volume_120: float,
    RWoML_120: float,
    RWoMB_120: float,
    alpha: float = 0.5,
    beta: float = 0.3,
    mean120_weight: float = 0.2,  # weight to assign to mean120
    std120_weight: float = 0.1,  # weight to assign to std_120
    volume120_weight: float = 0.2,  # weight to assign to volume_120
    RWoML120_weight: float = 0.25,  # weight to assign to RWoML_120
    RWoMB120_weight: float = 0.25,  # weight to assign to RWoMB_120
) -> float:
    """
    Calculates the odds using a weighted average of the predicted BSP, confidence price, and mean120, std_120, volume_120, RWoML_120, and RWoMB_120.

    Args:
        predicted_bsp (float): The predicted BSP for the runner.
        confidence_price (float): The confidence price for the runner.
        mean120 (float): The mean120 for the runner.
        alpha (float, optional): The weight to assign to the predicted BSP. Defaults to 0.5.
        beta (float, optional): The weight to assign to the weighted average of the confidence price and mean120. Defaults to 0.3.
        mean120_weight (float, optional): The weight to assign to the mean_120 feature. Defaults to 0.2.
        std120_weight (float, optional): The weight to assign to the std_120 feature. Defaults to 0.1.
        volume120_weight (float, optional): The weight to assign to the volume_120 feature. Defaults to 0.2.
        RWoML120_weight (float, optional): The weight to assign to the RWoML_120 feature. Defaults to 0.25.
        RWoMB120_weight (float, optional): The weight to assign to the RWoMB_120 feature. Defaults to 0.25.

    Returns:
        float: The calculated odds for the runner.
    """
    # Calculate the weighted average of the five features
    features_avg = (
        mean120_weight * mean120
        + std120_weight * std_120
        + volume120_weight * volume_120
        + RWoML120_weight * RWoML_120
        + RWoMB120_weight * RWoMB_120
    )

    odds = alpha * predicted_bsp + (1 - alpha) * (
        beta * confidence_price + (1 - beta) * features_avg
    )
    return odds


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
