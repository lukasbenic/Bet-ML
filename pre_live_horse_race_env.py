import argparse
from enum import Enum
import math
import gym
from gym.spaces import Discrete, Box
import joblib
import numpy as np
import optuna
from sklearn.base import BaseEstimator
import torch
from RL.TensorBoardCallback import TensorBoardRewardLogger
from onedrive import Onedrive
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DDPG, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from utils.utils import (
    calculate_kelly_stake,
    calculate_margin,
    calculate_odds,
    calculate_stake,
)
from utils.config import app_principal, SITE_URL
from pandas import DataFrame
from stable_baselines3.common.monitor import Monitor
import pandas as pd


from utils.utils import get_train_data

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Actions(Enum):
    BACK = 0
    LAY = 1


class PreLiveHorseRaceEnv(gym.Env):
    def __init__(self, obs_df: DataFrame, target_df: DataFrame, regressor: str):
        super(PreLiveHorseRaceEnv, self).__init__()
        self.obs_df = obs_df
        self.regressor = joblib.load(f"models/{regressor}.pkl")
        self.regressor_name = regressor
        self.balance = 100000.00
        self.target_df = target_df
        self.current_step = 0
        self.action_space = Discrete(len(Actions))
        self.observation_space = Box(
            low=obs_df.min().values,
            high=obs_df.max().values,
            shape=(obs_df.shape[1],),
            dtype=np.float64,
        )

    def step(self, action):
        # NOTE - this might not be necessary
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()
        elif isinstance(action, list) and len(action) == 1:
            action = action[0]

        # reward = 0
        state = self.obs_df.iloc[self.current_step]
        predict_row = state.to_frame().T

        target = self.target_df.iloc[self.current_step]

        # Inform our decision
        predicted_bsp = self.regressor.predict(predict_row)[0]

        target_value = target["bsps_temp"]
        mean_120 = state["mean_120"]
        vol_120 = state["volume_120"]
        std_120 = state["std_120"]

        # predicted_odds = calculate_odds(predicted_bsp, target_value, mean_120)
        reward, stake, margin = self._calculate_reward(
            action, predicted_bsp, target_value
        )
        self.current_step += 1
        self.balance += reward  # update for accurate kelly_stake if using

        done = (
            True
            if self.current_step >= self.obs_df.shape[0] or self.balance <= 0
            else False
        )

        next_state = None if done else self.obs_df.iloc[self.current_step].values
        info = {
            # "predicted_odds": predicted_odds,
            "predicted_bsp": predicted_bsp,
            "stake": stake,
            "margin": margin,
            "balance": self.balance,
            "step": self.current_step,
        }

        return next_state, reward / 1000, done, info

    def _calculate_reward(self, action, predicted_bsp, target_value):
        side = "BACK" if action == Actions.BACK.value else "LAY"
        stake = calculate_stake(500, predicted_bsp, side)
        margin = calculate_margin(side, stake, predicted_bsp, target_value)
        reward = (
            margin * 0.95 if margin > 0 else margin
        )  # remove 5% for commission onn profit

        return reward, stake, margin

    def reset(self):
        self.current_step = 0
        self.balance = 100000.00
        return self.obs_df.iloc[self.current_step].values

    def render(self, mode="human"):
        pass

    def close(self):
        pass


# NOTE support for ddpg and dqn not available
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
            total_timesteps=n_timesteps,
            callback=eval_callback,
        )
    else:
        model.learn(total_timesteps=n_timesteps)

    if save:
        model.save(f"RL/{model_name}/{model_name}_{env.regressor_name}_model")

    return eval_callback.best_mean_reward


def train_optimize_model(
    model_name: str,
    obs_df: pd.DataFrame,
    target_df: pd.DataFrame,
    regressor,
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
        env = PreLiveHorseRaceEnv(obs_df, target_df, regressor=regressor)
        model = create_model(model_name, env, device, net_arch, learning_rate)

        # Train the model and evaluate it
        # Env for evaluation
        eval_env = PreLiveHorseRaceEnv(obs_df, target_df, regressor=regressor)
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
    new_env = PreLiveHorseRaceEnv(obs_df, target_df, regressor=regressor)

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


if __name__ == "__main__":
    onedrive = Onedrive(
        client_id=app_principal["client_id"],
        client_secret=app_principal["client_secret"],
        site_url=SITE_URL,
    )
    ticks_df = onedrive.get_folder_contents(
        target_folder="ticks", target_file="ticks.csv"
    )
    bsp_df = onedrive.get_bsps(target_folder="july_22_bsps")

    obs_df, target_df, _ = get_train_data(
        "utils/x_train_df.csv",
        "utils/y_train_df.csv",
        onedrive,
        ticks_df,
        regression=True,
    )
    # env = DummyVecEnv([lambda: PreLiveHorseRaceEnv(obs_df, target_df)])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl_model",
        type=str,
        default="PPO",
        help="RL algorithm to use.",
    )

    parser.add_argument(
        "--regression_model",
        type=str,
        default="Ensemble",
        help="Regression model to use.",
    )

    args = parser.parse_args()
    env = PreLiveHorseRaceEnv(obs_df, target_df, regressor=args.regression_model)
    # lstm_net_arch = dict(pi=[64, 64, ("lstm", 64)], vf=[64, 64, ("lstm", 64)])
    # small_net_arch = dict(pi=[32, 32], vf=[32,  32])
    train_optimize_model(
        model_name=args.rl_model,
        obs_df=obs_df,
        target_df=target_df,
        regressor=args.regression_model,
    )
    # model = PPO.load("RL/PPO/PPO_model")
    # print("policy", model.policy)
