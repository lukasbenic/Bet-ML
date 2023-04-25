import os
from typing import List
import joblib
import pyro
from stable_baselines3 import PPO
from deep_learning.bayesian_regression import BayesianRegressionModel
from strategies.mean_120_bayesian_regression import Mean120BayesianRegression
from utils.bayesian_regression_utils import prepare_data, train_bayesian_regression
from onedrive import Onedrive
from strategies.mean_120_regression import Mean120Regression
from strategies.rl_strategy import RLStrategy
from utils.regressor_model_utils import train_test_model

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal

import torch


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
        models = model_name.split("_")
        rl_agent = PPO.load(f"RL/{models[0]}/{model_name}/{model_name}_model_4_3")
        tp_regressor = joblib.load(
            f"RL/timepoint_regressors/models/{models[1]}_120.pkl"
        )
        # tp_regressor = joblib.load(f"models/BayesianRidge.pkl")
        strategy_pick = RLStrategy(
            model=rl_agent,
            tp_regressor=tp_regressor,
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
        models = model_name.split("_")
        rl_agent = PPO.load(f"RL/{models[0]}/{model_name}/{model_name}_model")
        tp_regressor = joblib.load(f"RL/timepoint_regressors/{models[1]}_120.pkl")
        strategy_pick = RLStrategy(
            model=rl_agent,
            tp_regressor=tp_regressor,
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

    if strategy == "Mean120BayesianRegressionStrategy":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_train_tensor, y_train_tensor, scaler, clm = prepare_data(
            x_train_path="utils/data/X_train.csv",
            y_train_path="utils/data/y_train.csv",
        )

        num_features = x_train_tensor.shape[1]
        br = BayesianRegressionModel(num_features).to(device)
        guide = AutoMultivariateNormal(br).to(device)

        if os.path.exists(f"models/{model_name}.pkl"):
            print("Loaded pre-existing Bayesian Regression model.")
            guide = AutoMultivariateNormal(br).to(device)
            pyro.get_param_store().load(f"models/{model_name}.pkl")
        else:
            print("Commencing Bayesian Regression training...")
            pyro.clear_param_store()
            optimizer = ClippedAdam(
                {"lr": 1.0e-3, "lrd": 0.1, "clip_norm": 10.0},
            )
            svi = SVI(br, guide, optimizer, loss=Trace_ELBO())
            guide = train_bayesian_regression(
                svi, x_train_tensor, y_train_tensor, 24, 500, device
            )
            pyro.get_param_store().save(f"models/{model_name}.pkl")

        strategy_pick = Mean120BayesianRegression(
            model=br,
            guide=guide,
            device=device,
            clm=clm,
            scaler=scaler,
            ticks_df=ticks_df,
            balance=balance,
            test_analysis_df=test_analysis_df,
            market_filter={"markets": market_file},
            max_trade_count=100000,
            max_live_trade_count=100000,
            max_order_exposure=10000,
            max_selection_exposure=100000,
        )
    return strategy_pick
