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
from sklearn.model_selection import train_test_split
import torch
from RL.TensorBoardCallback import TensorBoardRewardLogger
from onedrive import Onedrive
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnMaxEpisodes,
    CallbackList,
)
from utils.rl_model_utils import (
    get_high,
    get_low,
    get_sorted_columns,
    get_timepoints,
    get_timesteps,
    get_tp_regressors,
    save_rolling_rewards,
    save_timesteps,
)

from utils.config import app_principal, SITE_URL
from pandas import DataFrame
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from utils.data_utils import get_train_data
import matplotlib.pyplot as plt
from utils.strategy_utils import calculate_margin, calculate_stake
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy


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
        self.columns = X.columns.difference(["predicted_bsp", "lay", "back"])
        self.y = y
        self.tp_regressors = tp_regressors
        self.predict_row = []
        self.ticks_df = ticks
        self.timesteps = get_timesteps(X)
        self.balance = 100000.00
        self.max_stake = self.balance * 0.08
        self.current_step = 0
        self.episode = 0
        self.timepoints = get_timepoints(X)
        self.current_timepoint = self.timepoints[0]
        self.action_space = Discrete(len(Actions))
        self.observation_space = Box(
            low=get_low(X, y),
            high=get_high(X, y),
            shape=(8,),  # mean, std, vol, RWoMB, RWoML, predicted_bsp, back, lay
            dtype=np.float64,
        )

    def step(self, action):
        # print(self.episode)
        observation = self._get_observation()
        bsp = self.y.iloc[self.episode - 1].to_frame().T["bsps"].values[0]
        predicted_bsp = self._predict(observation)

        side = (
            "back"
            if action == Actions.BACK.value
            else "lay"
            if action == Actions.LAY.value
            else ""
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
        # print(
        #     self.X.iloc[1][["back", "lay", "predicted_bsp"]],
        # )

        return next_observation, reward, done, info

    def _predict(self, observation):
        # Add current observation[:4] to predict_row
        self.predict_row[:0] = observation[:5]
        regressor = self.tp_regressors[f"{self.current_timepoint}"]["model"]
        scaler = self.tp_regressors[f"{self.current_timepoint}"]["scaler"]

        relevant_cols = get_sorted_columns(self.columns, self.current_timepoint)
        # Create a DataFrame from the observation array with the relevant columns
        observation_df = pd.DataFrame([self.predict_row], columns=relevant_cols)

        # Transform the observation DataFrame using the scaler
        observation_scaled = pd.DataFrame(
            scaler.transform(observation_df), columns=relevant_cols
        )

        predicted_bsp = regressor.predict(observation_scaled)[0]

        observation[-3] = predicted_bsp
        self._set_predicted_bsp_observation(predicted_bsp)
        return predicted_bsp

    def _set_margin_observation(self, margin: float, side: str) -> None:
        """
        Set the margin observation in the original dataframe.

        Args:
        margin (float): Margin value to set.
        side (str): The side to set the margin value for.

        Returns:
        None
        """
        iloc = -1 if side == "back" else -2
        self.X.iat[self.episode - 1, iloc] = margin

    def _set_predicted_bsp_observation(self, predicted_bsp: float) -> None:
        """
        Set the predicted_bsp observation in the original dataframe.

        Args:
        predicted_bsp (float): Predicted BSP value to set.

        Returns:
        None
        """
        self.X.iat[self.episode - 1, -3] = predicted_bsp

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

            if (
                side == "back"
                and predicted_bsp < confidence_price
                and observation[-1] == 0
            ):
                observation[-1] = margin

                self._set_margin_observation(margin, side)

            if (
                side == "lay"
                and predicted_bsp > confidence_price
                and observation[-2] == 0
            ):
                observation[-2] = margin

                self._set_margin_observation(margin, side)

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
            # maybe see what happens when we don't set a final bet?

            # Back set to mean_120
            # if observation[-1] == 0:
            #     observation[-1] = mean

            # # Lay set to mean_120
            # if observation[-2] == 0:
            #     observation[-2] = mean

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
        ].copy()  # episode will start at 1 due to +=1 in reset
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

        :param mean: The mean value for the current runner
        :param side: The side of the order
        :return: A tuple containing the adjusted price, confidence price
        """
        number = self.ticks_df.iloc[self.ticks_df["tick"].sub(mean).abs().idxmin()][
            "number"
        ]
        number_adjust = number
        # -2 +2 hex
        confidence_number = number - 2 if side == "lay" else number + 2
        confidence_price = self.ticks_df.iloc[
            self.ticks_df["number"].sub(confidence_number).abs().idxmin()
        ]["tick"]
        price_adjusted = self.ticks_df.iloc[
            self.ticks_df["number"].sub(number_adjust).abs().idxmin()
        ]["tick"]

        return price_adjusted, confidence_price


def create_model(model_name: str, env, device, policy_kwargs):
    model_class = {
        "ppo": PPO,
        "rppo": RecurrentPPO,
    }.get(model_name.lower())

    if model_class is None:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name.lower() == "ppo":
        model = model_class(
            "MlpPolicy",
            env,
            verbose=1,
            seed=42,
            device=device,
            policy_kwargs=policy_kwargs,
        )
        return model

    if model_name.lower() == "rppo":
        model = model_class("MlpLstmPolicy", env, verbose=1, seed=42, device=device)
        return model


def train_model(
    model,
    env: PreLiveHorseRaceEnv,
    model_name: str = "",
    eval_model=False,
    trial=None,
    study_path="",
):
    max_eps = StopTrainingOnMaxEpisodes(max_episodes=(len(env.X) - 1), verbose=1)
    if eval_model:
        rl_name = model_name.split("_")[0]
        eval_callback = EvalCallback(
            env,
            log_path=f"RL/{study_path}/{rl_name}/{model_name}/trial_{trial}",
            eval_freq=200,
            deterministic=True,
            render=False,
            callback_after_eval=max_eps,
        )
        # tensorboard_callback = TensorBoardRewardLogger(f"RL/{model_name}/tensorboard/")
        print("model trainining commenced...")
        model.learn(
            total_timesteps=env.timesteps,  # timepoints each row x rows
            callback=CallbackList([max_eps, eval_callback]),
        )
        return eval_callback.best_mean_reward
    else:
        trained_model = model.learn(total_timesteps=env.timesteps, callback=max_eps)
        return trained_model


def train_optimize_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    tpr_name: str,
    tp_regressors: Dict,
    ticks: pd.DataFrame,
    n_trials=2,
    timeout=600,
    study_save_path="optuna_study",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the objective function for Optuna
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        pi_layers = trial.suggest_categorical("pi_layers", [32, 64, 128])
        vf_layers = trial.suggest_categorical("vf_layers", [32, 64, 128])

        net_arch = dict(pi=[pi_layers, pi_layers], vf=[vf_layers, vf_layers])

        env = PreLiveHorseRaceEnv(X, y, tp_regressors, ticks)
        env = Monitor(env)
        model = create_model(model_name, env, device, net_arch, learning_rate)

        # Train the model and evaluate it
        eval_metric = train_model(
            env=env,
            model_name=f"{model_name}_{tpr_name}",
            model=model,
            eval_model=True,
            trial=trial.number,
            study_path=study_save_path,
        )
        print(f"Trial {trial.number}, Evaluation metric: {eval_metric}")
        return eval_metric

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)

    print(f"Best hyperparameters: {study.best_params}")

    # Save the study for later inspection and analysis
    os.makedirs(study_save_path, exist_ok=True)
    study_file = f"RL/{study_save_path}/{model_name}/{model_name}_{tpr_name}_study.pkl"

    with open(study_file, "wb") as f:
        joblib.dump(study, f)

    print(f"Best hyperparameters: {study.best_params}")
    new_env = PreLiveHorseRaceEnv(X, y, tp_regressors, ticks)

    best_params = study.best_params
    best_net_arch = dict(
        pi=[best_params["pi_layers"], best_params["pi_layers"]],
        vf=[best_params["vf_layers"], best_params["vf_layers"]],
    )
    best_model = create_model(
        model_name, new_env, device, best_net_arch, best_params["learning_rate"]
    )
    train_model(
        model=best_model,
        env=new_env,
    )

    # save best model and best hyperparams
    best_model.save(f"RL/{model_name}/{model_name}_{tpr_name}_model")
    pd.DataFrame.from_records([best_params], index=[0]).to_csv(
        f"RL/{model_name}/{model_name}_{tpr_name}_best_params.csv"
    )


def train_model2(
    rl_model_name: str,
    tpr_name: str,
    env: PreLiveHorseRaceEnv,
    eval_env: PreLiveHorseRaceEnv = None,
    policy_kwargs=dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    ),
    save: bool = True,
    saved_model_path="",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = {"ppo": PPO, "rppo": RecurrentPPO}.get(rl_model_name.lower())

    if model_class is None:
        raise ValueError(f"Unsupported model: {rl_model_name}")

    # Wrap the environment with Monitor
    log_dir_train = f"RL/{rl_model_name}/{rl_model_name}_{tpr_name}"
    log_dir_eval = f"RL/{rl_model_name}/{rl_model_name}_{tpr_name}/eval"

    env = Monitor(env, f"{log_dir_train}/train_monitor.csv")

    callback_max_ep = StopTrainingOnMaxEpisodes(
        max_episodes=(len(env.X) - 1), verbose=1
    )
    if eval_env:
        eval_env = Monitor(eval_env, f"{log_dir_eval}/eval_monitor.csv")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{log_dir_eval}/best_model/",
            log_path=f"{log_dir_eval}/eval_log/",
            eval_freq=500,
            callback_after_eval=callback_max_ep,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

    # Combine the callbacks using CallbackList
    callback_list = (
        CallbackList([callback_max_ep, eval_callback]) if eval_env else callback_max_ep
    )

    model = (
        PPO.load(
            f"RL/{rl_model_name}/{rl_model_name}_{tpr_name}/{rl_model_name}_{tpr_name}_model_{saved_model_path}"
        )
        if saved_model_path
        else create_model(rl_model_name, env, device, policy_kwargs=policy_kwargs)
    )
    if saved_model_path:
        model.set_env(env)

    trained_model = model.learn(total_timesteps=env.timesteps, callback=callback_list)

    if save:
        trained_model.save(
            f"RL/{rl_model_name}/{rl_model_name}_{tpr_name}/{rl_model_name}_{tpr_name}_model"
        )

    return trained_model


def visualize_environment_trained_model(env, model, n_samples=1000):
    # Generate samples from the environment using the trained model
    observations = np.zeros((n_samples, env.observation_space.shape[0]))
    actions = np.zeros(n_samples, dtype=int)

    obs = env.reset()
    for i in range(n_samples):
        action, _ = model.predict(obs)
        next_obs, _, done, _ = env.step(action)
        observations[i, :] = obs
        actions[i] = action

        if done:
            obs = env.reset()
        else:
            obs = next_obs

    # Calculate the differences between consecutive observations
    obs_diff = np.diff(observations, axis=0)

    # State labels
    state_labels = [
        "mean",
        "std",
        "vol",
        "RWoMB",
        "RWoML",
        "predicted_bsp",
        "back",
        "lay",
    ]

    # Plot the changes in state variables for each action
    fig, axes = plt.subplots(len(Actions), len(state_labels), figsize=(15, 9))
    for i, action in enumerate(Actions):
        for j, label in enumerate(state_labels):
            mask = actions[:-1] == action.value
            axes[i, j].scatter(
                np.arange(n_samples - 1)[mask], obs_diff[mask, j], s=5, alpha=0.7
            )
            axes[i, j].set_ylim(np.min(obs_diff[:, j]), np.max(obs_diff[:, j]))
            if i == 0:
                axes[i, j].set_title(label)
            if j == 0:
                axes[i, j].set_ylabel(action.name)

    plt.tight_layout()
    plt.show()


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
    env = PreLiveHorseRaceEnv(X_rl, y_rl, tp_regressors, ticks_df)
    # eval_env = PreLiveHorseRaceEnv(X_rl, y_rl, tp_regressors, ticks_df)
    model = train_model2(
        args.rl_model, args.tp_regressors, env, saved_model_path="2_-2_+2"
    )

    # train_optimize_model("PPO", X_rl, y_rl, args.tp_regressors, tp_regressors, ticks_df)
    # save_rolling_rewards(
    #     file_path="RL/PPO/PPO_BayesianRidge/train_monitor_128_+10_-3.csv",
    #     save_path="RL/PPO/PPO_BayesianRidge/train_monitor_rolling_128_+10_-3.csv",
    # )
