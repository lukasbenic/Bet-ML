from pprint import pprint

import yaml

from onedrive import Onedrive
from flumine import clients
from flumine_simulator import piped_run
from matplotlib import pyplot as plt
from utils.config import app_principal, SITE_URL
import argparse


plt.rcParams["figure.figsize"] = (20, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--races",
        type=int,
        default=1,
        help="Number of races(markets) to run, the max is 929",
    )
    parser.add_argument(
        "--test_folder_path",
        type=str,
        default="horses_jul_wins",
        help="Path to test folder",
    )
    parser.add_argument(
        "--bsps_path", type=str, default="july_22_bsps", help="Path to BSPs folder"
    )
    parser.add_argument(
        "--strategy_name",
        type=str,
        default="BayesianRegressionStrategy",
        help="Name of the strategy to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BayesianRegression",
        help="Name of the model to use",
    )
    args = parser.parse_args()

    tracker = piped_run(
        strategy=args.strategy_name,
        onedrive=Onedrive(
            client_id=app_principal["client_id"],
            client_secret=app_principal["client_secret"],
            site_url=SITE_URL,
        ),
        client=clients.SimulatedClient(),
        test_folder_path=args.test_folder_path,
        bsps_path=args.bsps_path,
        model_name=args.model_name,
        races=args.races,
        save=True,
    )

    # tracker = dict()
    # with open("models/BayesianRidge_results.yaml", "r") as f:
    #     tracker = yaml.load(f, Loader=yaml.UnsafeLoader)
    # fig = get_simulation_plot(tracker, "Strategy1")
    # plt.show()
    # metrics = dict()
    # with open("dummy_data/metrics.yaml", "r") as f:
    #     metrics = yaml.safe_load(f)

    pprint(tracker)
