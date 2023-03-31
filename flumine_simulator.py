import logging
import os
import time
from pprint import pprint
import pyro
import torch

import yaml
from deep_learning.bayesian_regression import (
    BayesianRegressionModel,
    model_gamma,
    prepare_data,
    train_bayesian_regression,
)
from onedrive import Onedrive
from strategies.mean_120_regression_strategy import Mean120RegressionStrategy
from flumine import FlumineSimulation
from pythonjsonlogger import jsonlogger
from flumine.clients import SimulatedClient
from strategies.bayesian_regression_strategy import (
    BayesianRegressionStrategy,
)
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam, SGD
from pyro.infer.autoguide import AutoDiagonalNormal

# from pyro.params.param_store import get_param_store


from utils.utils import (
    train_test_model,
    update_tracker,
)
from deep_learning.vae_regressor import VAE

import atexit
import multiprocessing as mp


def cleanup():
    mp.set_start_method("spawn", force=True)


atexit.register(cleanup)


def run(strategy, client: SimulatedClient, races):
    framework = FlumineSimulation(client=client)
    framework.add_strategy(strategy)
    market_files = strategy.market_filter["markets"]

    tracker = {
        "total_profit": 0,
        "total_matched_correct": 0,
        "total_matched_incorrect": 0,
        "total_back_matched_correct": 0,
        "total_back_matched_incorrect": 0,
        "total_lay_matched_correct": 0,
        "total_lay_matched_incorrect": 0,
        "total_m_c_margin": 0,
        "total_m_i_margin": 0,
        "total_green_margin": 0,
        "total_amount_gambled": 0,
        "actual_profit_plotter": [],
        "expected_profit_plotter": [],
        "green_plotter": [],
        "race_counter": 0,
        "total_q_correct": 0,
        "total_q_incorrect": 0,
        "total_q_margin": 0,
    }

    for index, market_file in enumerate(market_files):
        print(len(market_files))
        # for smaller test run
        if races and index == races:
            break

        market_filter = {"markets": [market_file]}

        print(f"Race {index + 1}/{races}")
        print("lukas markets", framework.markets)

        strategy.set_market_filter(market_filter=market_filter)
        strategy.reset_metrics()

        framework.run()
        print(f"Race {index + 1} finished...")
        for market in framework.markets:
            # print(
            #     "Profit: {0:.2f}".format(
            #         sum([o.simulated.profit for o in market.blotter])
            #     )
            # )
            strategy.metrics["profit"] += sum(
                [o.simulated.profit for o in market.blotter]
            )
        update_tracker(tracker, strategy.metrics)
        pprint(tracker)

    return tracker


def get_strategy(
    strategy: str,
    market_file,  #: List[str] | str,
    onedrive: Onedrive,
    model_name: str,
):
    mp.set_start_method("spawn", force=True)
    market_file = market_file if isinstance(market_file, list) else [market_file]

    ticks_df = onedrive.get_folder_contents(
        target_folder="ticks", target_file="ticks.csv"
    )

    test_analysis_df = onedrive.get_test_df(target_folder="Analysis_files")

    if strategy == "Mean120RegressionStrategy":
        model, clm, scaler = train_test_model(
            ticks_df,
            onedrive,
            model_name=model_name,
        )
        strategy_pick = Mean120RegressionStrategy(
            model=model,
            ticks_df=ticks_df,
            clm=clm,
            scaler=scaler,
            test_analysis_df=test_analysis_df,
            market_filter={"markets": market_file},
            max_trade_count=100000,
            max_live_trade_count=100000,
            max_order_exposure=10000,
            max_selection_exposure=100000,
        )

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
        my_guide = AutoDiagonalNormal(model_gamma)

        pyro.clear_param_store()
        svi = SVI(model_gamma, my_guide, optimizer, loss=Trace_ELBO())

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


def piped_run(
    strategy: str,
    onedrive: Onedrive,
    client: SimulatedClient,
    test_folder_path: str,
    bsps_path: str,
    model_name: str,
    races=None,
    save=False,
    log_lvl=logging.CRITICAL,
):
    logger = logging.getLogger(__name__)
    custom_format = "%(asctime) %(levelname) %(message)"
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(custom_format)
    formatter.converter = time.gmtime
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(log_lvl)  # Set to logging.CRITICAL to speed up backtest

    bsp_df = onedrive.get_bsps(target_folder=bsps_path)

    test_folder_files = os.listdir(test_folder_path)
    number_files = len(test_folder_files)

    if number_files == 0:
        print("Starting test folder download...")
        onedrive.download_test_folder(target_folder="horses_jul_wins")
        print("Test folder download finished.")

    file_paths = [
        os.path.join(test_folder_path, f_name)
        for f_name in test_folder_files
        if float(f_name) in bsp_df["EVENT_ID"].values
    ]
    strategy_pick = get_strategy(strategy, file_paths, onedrive, model_name)
    print("herehereherhre")
    tracker = run(strategy_pick, client, races)

    if save:
        with open(f"results/{strategy}_results.yaml", "w") as f:
            yaml.dump(tracker, f)

    return tracker
