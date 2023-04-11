import os
from typing import List, Tuple
import joblib
import numpy as np
import pyro
from stable_baselines3 import PPO
from deep_learning.bayesian_regression import (
    BayesianRegressionModel,
    prepare_data,
    train_bayesian_regression,
)
from onedrive import Onedrive
from strategies.bayesian_regression_strategy import BayesianRegressionStrategy
from strategies.mean_120_regression import Mean120Regression
from strategies.rl_strategy import RLStrategy
from utils.utils import train_test_model
from strategies.bayesian_regression_strategy import (
    BayesianRegressionStrategy,
)
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoDiagonalNormal
from betfairlightweight.resources import MarketBook, RunnerBook


def get_strategy(
    strategy: str,
    market_file: List[str],
    onedrive: Onedrive,
    model_name: str,
    balance: float,
):
    ticks_df = onedrive.get_folder_contents(
        target_folder="ticks", target_file="ticks.csv"
    )

    test_analysis_df = onedrive.get_test_df(target_folder="Analysis_files")

    if strategy == "Mean120RegressionGreen":
        model, clm, scaler = train_test_model(
            ticks_df,
            onedrive,
            model_name=model_name,
        )
        strategy_pick = Mean120Regression(
            model=model,
            ticks_df=ticks_df,
            clm=clm,
            scaler=scaler,
            balance=balance,
            test_analysis_df=test_analysis_df,
            market_filter={"markets": market_file},
            max_trade_count=100000,
            max_live_trade_count=100000,
            max_order_exposure=10000,
            max_selection_exposure=100000,
            green_enabled=True,
        )
    if strategy == "Mean120Regression":
        model, clm, scaler = train_test_model(
            ticks_df,
            onedrive,
            model_name=model_name,
        )
        strategy_pick = Mean120Regression(
            model=model,
            ticks_df=ticks_df,
            clm=clm,
            scaler=scaler,
            balance=balance,
            test_analysis_df=test_analysis_df,
            market_filter={"markets": market_file},
            max_trade_count=100000,
            max_live_trade_count=100000,
            max_order_exposure=10000,
            max_selection_exposure=100000,
        )
    if strategy == "RLStrategy":
        rl_agent = PPO.load(f"RL/{model_name}/{model_name}_model")
        strategy_pick = RLStrategy(
            rl_agent=rl_agent,
            ticks_df=ticks_df,
            balance=balance,
            test_analysis_df=test_analysis_df,
            market_filter={"markets": market_file},
            max_trade_count=100000,
            max_live_trade_count=100000,
            max_order_exposure=10000,
            max_selection_exposure=100000,
        )

    if strategy == "RLStrategyGreen":
        rl_agent = PPO.load(f"RL/{model_name}/{model_name}_model")
        strategy_pick = RLStrategy(
            rl_agent=rl_agent,
            ticks_df=ticks_df,
            balance=balance,
            test_analysis_df=test_analysis_df,
            market_filter={"markets": market_file},
            max_trade_count=100000,
            max_live_trade_count=100000,
            max_order_exposure=10000,
            max_selection_exposure=100000,
            green_enabled=True,
        )

    # TODO Implement Balance
    if strategy == "BayesianRegressionStrategy":
        x_train_tensor, y_train_tensor = prepare_data(
            x_train_path="utils/x_train_df.csv", y_train_path="utils/y_train_df.csv"
        )

        print(y_train_tensor.shape)
        print("data prepared")

        optimizer = ClippedAdam(
            {"lr": 1.0e-3, "lrd": 0.1, "clip_norm": 10.0},
        )

        num_features = x_train_tensor.shape[1]
        br = BayesianRegressionModel(num_features)
        guide = AutoDiagonalNormal(br)

        pyro.clear_param_store()
        svi = SVI(br, guide, optimizer, loss=Trace_ELBO())

        if os.path.exists(f"models/{model_name}.pkl"):
            print("Loaded pre-existing VAE model.")
        else:
            print("Commencing SVI training...")
            train_bayesian_regression(svi, x_train_tensor, y_train_tensor, 1, 500)

        strategy_pick = BayesianRegressionStrategy(
            model=svi,
            ticks_df=ticks_df,
            test_analysis_df=test_analysis_df,
            market_filter={"markets": market_file},
            max_trade_count=100000,
            max_live_trade_count=100000,
            max_order_exposure=10000,
            max_selection_exposure=100000,
        )
    return strategy_pick


def calculate_fixed_stake(
    side: str,
    stake: float,
    price_adjusted: float,
) -> float:
    """
    Calculates the size of the bet based on the side, stake, price_adjusted and bsp_value.

    Args:
        side (str): The side of the bet ("BACK" or "LAY").
        stake (float): The stake to be used in the bet.
        price_adjusted (float): The adjusted price for the order.
        bsp_value (float): The Betfair Starting Price value for the runner.

    Returns:
        float: The size of the bet.
    """
    if side == "BACK":
        return stake
    else:
        lay_stake = stake / (price_adjusted - 1)
        return round(lay_stake, 2)
