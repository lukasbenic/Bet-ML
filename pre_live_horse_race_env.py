import argparse
from enum import Enum
import math
import gym
from gym.spaces import Discrete, Box
import joblib
from matplotlib import pyplot as plt
import numpy as np
import optuna
from sklearn.base import BaseEstimator
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
    get_timesteps_and_timepoints,
)
from utils.strategy_utils import (
    calculate_kelly_stake,
    calculate_margin,
    calculate_odds,
    calculate_stake,
)
from utils.config import app_principal, SITE_URL
from pandas import DataFrame
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from utils.data_utils import get_train_data


class Actions(Enum):
    BACK = 0
    LAY = 1


class PreLiveHorseRaceEnv(gym.Env):
    def __init__(self, X: DataFrame, y: DataFrame):
        super(PreLiveHorseRaceEnv, self).__init__()
        self.X = X
        self.balance = 100000.00
        self.y = y
        self.current_step = 0
        self.timepoints = get_timepoints(X)
        self.current_timepoint = self.timepoints[0]
        self.action_space = Discrete(len(Actions))
        self.observation_space = Box(
            low=get_low(
                X, self.timepoints[self.current_step]
            ),  # update this as we need to get the low and high based onn all the observations from different time points e.g. look at all means and find low and high 
                # do the same for rest of features
            high=get_high(X, self.timepoints[self.current_step]),
            shape=(7,),
            dtype=np.float64,
        )

    def step(self, action):
        reward = 1

        observation = self._get_observation()
        print(observation)
        bsp = self.y.iloc[self.current_step].to_frame().T["bsp"].values[0]

        self.current_step += 1
        self.current_timepoint = self.timepoints[self.current_step]

        done = self._is_done()

        next_observation = self._get_observation()

        info = {
            "current_step": self.current_step,
            "current_timepoint": self.current_timepoint,
            "reward": reward,
            "bsp": bsp,
        }

        return next_observation, reward, done, info

    def _calculate_reward(self, action, state, target):
        pass

    def reset(self):
        self.current_step = 0  # start at first row of observation space
        self.current_timepoint = self.timepoints[0]  # start at 14400 before race starts
        self.balance = 100000.00
        observation = self._get_observation()

        return observation

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _get_reward(self, action, observation, bsp):
        pass

    def _is_done(self):
        return (
            True
            if self.balance <= 0 or self.current_timepoint == self.timepoints[-1]
            else False  # end of row
        )

    def _get_observation(self) -> np.ndarray:
        row = self.X.iloc[self.current_step]
        filtered_row = row.filter(
            regex=f"^(?!lay|back).*_{self.current_timepoint}$"
        ).values
        lay_back_values = row[["lay", "back"]].values
        observation = np.hstack([filtered_row, lay_back_values])

        if observation.shape[0] != 7:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")

        return observation


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


def train_model2(model_name: str, env: PreLiveHorseRaceEnv, timesteps: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = {
        "ppo": PPO,
    }.get(model_name.lower())

    if model_class is None:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        device=device,
    )
    model.learn(total_timesteps=timesteps)

    return model


if __name__ == "__main__":
    onedrive = Onedrive(
        client_id=app_principal["client_id"],
        client_secret=app_principal["client_secret"],
        site_url=SITE_URL,
    )

    X, y = get_train_data(
        onedrive,
    )
    X[["lay", "back"]] = 0
    timesteps = get_timesteps(X)

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--rl_model",
    #     type=str,
    #     default="PPO",
    #     help="RL algorithm to use.",
    # )
    # args = parser.parse_args()
    env = PreLiveHorseRaceEnv(X, y)
    model = train_model2("PPO", env, timesteps)
    # # lstm_net_arch = dict(pi=[64, 64, ("lstm", 64)], vf=[64, 64, ("lstm", 64)])
    # # small_net_arch = dict(pi=[32, 32], vf=[32,  32])
    # train_optimize_model(
    #     model_name=args.rl_model,
    #     X=X,
    #     y=y,
    # )
    # model = PPO.load("RL/PPO/PPO_model")
    # print("policy", model.policy)
