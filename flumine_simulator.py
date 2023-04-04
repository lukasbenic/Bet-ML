import logging
import os
import time
from pprint import pprint
import yaml
from onedrive import Onedrive
from flumine import FlumineSimulation
from pythonjsonlogger import jsonlogger
from flumine.clients import SimulatedClient
from strategies.utils import get_strategy
from utils.utils import update_tracker
from deep_learning.vae_regressor import VAE


def run(
    strategy,
    framework: SimulatedClient,
):
    framework.add_strategy(strategy)
    framework.run()

    for market in framework.markets:
        strategy.metrics["profit"] += sum([o.simulated.profit for o in market.blotter])

    return strategy.metrics


def piped_run(
    strategy_name: str,
    onedrive: Onedrive,
    test_folder_path: str,
    bsps_path: str,
    model_name: str,
    races,
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

    if len(test_folder_files) == 0:
        print("Starting test folder download...")
        onedrive.download_test_folder(target_folder="horses_jul_wins")
        print("Test folder download finished.")

    market_files = [
        os.path.join(test_folder_path, f_name)
        for f_name in test_folder_files
        if float(f_name) in bsp_df["EVENT_ID"].values
    ]
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
        "balance": 10000.00,
    }
    for market_file in market_files[0:races]:
        framework = FlumineSimulation(client=SimulatedClient())
        strategy = get_strategy(
            strategy_name,
            [market_file],
            onedrive,
            model_name,
            balance=tracker["balance"],
        )
        metrics = run(strategy, framework)
        update_tracker(tracker, metrics)
        tracker["balance"] = strategy.balance

    if save:
        with open(f"results/{strategy_name}_results.yaml", "w") as f:
            yaml.dump(tracker, f)

    return tracker
