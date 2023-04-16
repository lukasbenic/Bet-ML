import argparse
from enum import Enum
import math
import os
from typing import Dict, List, Tuple
import gym
from gym.spaces import Discrete, Box
import joblib
from matplotlib import pyplot as plt
import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
from RL.TensorBoardCallback import TensorBoardRewardLogger
from onedrive import Onedrive
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DDPG, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from utils.rl_model_utils import (
    get_high,
    get_low,
    get_timepoints,
    get_timesteps,
    get_tp_regressors,
    train_tp_regressors,
)

from utils.config import app_principal, SITE_URL
from pandas import DataFrame
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from utils.data_utils import get_train_data
from sklearn.preprocessing import StandardScaler

from utils.strategy_utils import calculate_margin, calculate_stake


class Actions(Enum):
    NOTHING = 0
    BACK = 1
    LAY = 2


class PreLiveHorseRaceEnv(gym.Env):
    def __init__(
        self, X: DataFrame, y: DataFrame, tp_regressors: Dict, ticks: pd.DataFrame
    ):
        super(PreLiveHorseRaceEnv, self).__init__()
        self.X = X
        self.y = y
        self.tp_regressors = tp_regressors
        self.predict_row = []
        self.ticks_df = ticks
        self.scaler = StandardScaler()
        self.timesteps = get_timesteps(X)
        self.balance = 100000.00
        self.max_stake = self.balance * 0.08
        self.current_step = 0
        self.episode = 0
        self.timepoints = get_timepoints(X)
        self.current_timepoint = self.timepoints[0]
        self.action_space = Discrete(len(Actions))
        self.observation_space = Box(
            low=get_low(X),
            high=get_high(X),
            shape=(8,),  # mean, std, vol, RWoMB, RWoML, predicted_bsp, back, lay
            dtype=np.float64,
        )

    def step(self, action):
        observation = self._get_observation()
        bsp = self.y.iloc[self.current_step].to_frame().T["bsps"].values[0]
        predicted_bsp = self._predict(observation)

        side = (
            "back" if action == Actions.BACK else "lay" if action == Actions.LAY else ""
        )

        self._make_bet(side, bsp, predicted_bsp, observation)

        done = self._is_done(observation)
        reward = self._get_reward(done, observation)

        if not done:
            self.current_step += 1
            self.current_timepoint = self.timepoints[self.current_step]

        next_observation = None if done else self._get_observation()

        info = {
            "current_step": self.current_step,
            "current_timepoint": self.current_timepoint,
            "reward": reward,
            "bsp": bsp,
            "side": side,
        }

        return next_observation, reward, done, info

    def _predict(self, observation):
        # Add current observation[:4] to predict_row
        self.predict_row[:0] = observation[:5]
        predicted_bsp = self.tp_regressors[f"{self.current_timepoint}"].predict(
            self.scaler.fit_transform(np.array(self.predict_row).reshape(1, -1))
        )
        observation[-3] = predicted_bsp
        return predicted_bsp

    def _make_bet(self, side, bsp, predicted_bsp, observation):
        mean = observation[0]
        if side:
            (
                price_adjusted,
                confidence_price,
            ) = self._get_adjusted_prices(mean=mean, side=side)

            stake = calculate_stake(self.max_stake, price_adjusted, side.upper())
            margin = calculate_margin(side.upper(), stake, price_adjusted, bsp)

            # need to check if bet has already been made as self._is_done only
            # terminates when both back and lay has been order placed
            if (
                side == "back"
                and predicted_bsp < confidence_price
                and observation[-1] == 0
            ):
                observation[-1] = margin

            if (
                side == "lay"
                and predicted_bsp > confidence_price
                and observation[-2] == 0
            ):
                observation[-2] = margin

    def reset(self):
        self.current_step = 0  # start at first row of observation space
        self.episode += 1
        self.current_timepoint = self.timepoints[0]  # start at 14400 before race starts
        self.balance = 100000.00
        self.predict_row = []
        observation = self._get_observation()

        return observation

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _get_reward(self, done, observation):
        mean = observation[0]
        reward = 0
        if done:
            # Back set to mean_120
            if observation[-1] == 0:
                observation[-1] = mean

            # Lay set to mean_120
            if observation[-2] == 0:
                observation[-2] = mean

            reward = observation[-1] + observation[-2]

        return reward

    def _is_done(self, observation):
        done = (
            True
            if self.balance <= 0
            or self.current_timepoint == self.timepoints[-1]
            or (
                observation[-1] and observation[-2]
            )  # check if we have backed and layed
            else False
        )  # end of row

        return done

    def _get_observation(self) -> np.ndarray:
        row = self.X.iloc[
            self.episode - 1
        ]  # episode will start at 1 due to +=1 in reset
        filtered_row = row.filter(
            regex=f"^(?!predicted_bsp|lay|back).*_{self.current_timepoint}$"
        ).values
        lay_back_bsp_values = row[["predicted_bsp", "lay", "back"]].values
        observation = np.hstack([filtered_row, lay_back_bsp_values])

        return observation

    def _get_adjusted_prices(
        self,
        mean: np.float64,
        side: str,
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Calculate adjusted prices for back and lay bets along with the BSP value.

        :param test_analysis_df_y: A DataFrame containing the test analysis data
        :param mean_120: The mean_120 value for the current runner
        :param runner: The current RunnerBook object
        :param market_id: The market ID for the current market
        :return: A tuple containing the adjusted price, confidence price, and BSP value
        """
        number = self.ticks_df.iloc[self.ticks_df["tick"].sub(mean).abs().idxmin()][
            "number"
        ]
        number_adjust = number
        confidence_number = number + 12 if side == "lay" else number - 4
        confidence_price = self.ticks_df.iloc[
            self.ticks_df["number"].sub(confidence_number).abs().idxmin()
        ]["tick"]
        price_adjusted = self.ticks_df.iloc[
            self.ticks_df["number"].sub(number_adjust).abs().idxmin()
        ]["tick"]

        return price_adjusted, confidence_price


def create_model(model_name: str, env, device, net_arch, learning_rate):
    if model_name.lower() == "dqn":
        policy_kwargs = dict(
            net_arch=net_arch["pi"],
        )
    else:
        policy_kwargs = dict(
            net_arch=net_arch,
        )

    model_class = {
        "ppo": PPO,
        "ddpg": DDPG,
        "dqn": DQN,
    }.get(model_name.lower())

    if model_class is None:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        seed=42,
        device=device,
    )

    return model


def train_model(
    model,
    env: PreLiveHorseRaceEnv,
    model_name: str,
    n_timesteps,
    save=False,
    callbacks=False,
):
    eval_env = Monitor(env)
    eval_callback = EvalCallback(
        eval_env,
        log_path=f"RL/{model_name}/logs/",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )
    if callbacks:
        # checkpoint_callback = CheckpointCallback(
        #     save_freq=5000, save_path=f"RL/{model_name}/logs/", name_prefix="model"
        # )
        # tensorboard_callback = TensorBoardRewardLogger(f"RL/{model_name}/tensorboard/")
        print("model trainining commenced...")
        model.learn(
            total_timesteps=n_timesteps,  # timepoints each row x rows
            callback=eval_callback,
        )
    else:
        model.learn(total_timesteps=n_timesteps)

    if save:
        model.save(f"RL/{model_name}/{model_name}_model")

    return eval_callback.best_mean_reward


def train_optimize_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_trials=100,
    n_timesteps=25000,
    timeout=600,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the objective function for Optuna
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        pi_layers = trial.suggest_categorical("pi_layers", [32, 64, 128])
        vf_layers = trial.suggest_categorical("vf_layers", [32, 64, 128])

        net_arch = dict(pi=[pi_layers, pi_layers], vf=[vf_layers, vf_layers])

        # ensure no interaction between different models trained, so we create a new one for each model
        env = PreLiveHorseRaceEnv(X, y)
        model = create_model(model_name, env, device, net_arch, learning_rate)

        # Train the model and evaluate it
        # Env for evaluation
        eval_env = PreLiveHorseRaceEnv(X, y)
        eval_metric = train_model(
            model=model,
            env=eval_env,
            model_name=model_name,
            callbacks=True,
            n_timesteps=n_timesteps,
            save=False,
        )
        print(f"Trial {trial.number}, Evaluation metric: {eval_metric}")
        return eval_metric

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)

    print(f"Best hyperparameters: {study.best_params}")
    new_env = PreLiveHorseRaceEnv(X, y)

    best_params = study.best_params
    best_net_arch = dict(
        pi=[best_params["pi_layers"], best_params["pi_layers"]],
        vf=[best_params["vf_layers"], best_params["vf_layers"]],
    )
    best_model = create_model(
        model_name, new_env, device, best_net_arch, best_params["learning_rate"]
    )
    _ = train_model(
        model=best_model,
        env=new_env,
        callbacks=False,
        n_timesteps=n_timesteps,
        save=True,
    )


def train_model2(
    rl_model_name: str, tpr_name: str, env: PreLiveHorseRaceEnv, save: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = {
        "ppo": PPO,
    }.get(rl_model_name.lower())

    if model_class is None:
        raise ValueError(f"Unsupported model: {rl_model_name}")

    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        device=device,
    )
    model.learn(total_timesteps=env.timesteps)
    if save:
        model.save(f"RL/{rl_model_name}/{rl_model_name}_{tpr_name}_model")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl_model",
        type=str,
        default="PPO",
        help="RL algorithm to use.",
    )
    parser.add_argument(
        "--tp_regressors",
        type=str,
        default="BayesianRidge",
        help="RL algorithm to use.",
    )
    args = parser.parse_args()

    onedrive = Onedrive(
        client_id=app_principal["client_id"],
        client_secret=app_principal["client_secret"],
        site_url=SITE_URL,
    )

    X, y = get_train_data(
        onedrive,
    )
    # Split data into 40% for regressors and 60% for RL algorithm
    X_regressors, X_rl, y_regressors, y_rl = train_test_split(
        X, y, test_size=0.6, random_state=42
    )
    ticks_df = onedrive.get_folder_contents(
        target_folder="ticks", target_file="ticks.csv"
    )
    X_rl[["predicted_bsp", "lay", "back"]] = 0

    tp_regressors = get_tp_regressors(X_regressors, y_regressors, args.tp_regressors)

    # print(X_rl)
    # sort regressors by timepoint in ascending order
    env = PreLiveHorseRaceEnv(X_rl, y_rl, tp_regressors, ticks_df)
    model = train_model2(args.rl_model, args.tp_regressors, env)

# # lstm_net_arch = dict(pi=[64, 64, ("lstm", 64)], vf=[64, 64, ("lstm", 64)])
# # small_net_arch = dict(pi=[32, 32], vf=[32,  32])
# train_optimize_model(
#     model_name=args.rl_model,
#     X=X,
#     y=y,
# )
# model = PPO.load("RL/PPO/PPO_model")
# print("policy", model.policy)
