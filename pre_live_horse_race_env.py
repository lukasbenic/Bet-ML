from collections import defaultdict
from enum import Enum
import os
import gym
from gym.spaces import Discrete, Box
import joblib
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import stable_baselines3
import torch
from onedrive import Onedrive
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from utils.config import app_principal, SITE_URL


from utils.utils import get_train_data


class Actions(Enum):
    BACK = 0
    LAY = 1


class PreLiveHorseRaceEnv(gym.Env):
    def __init__(self, x_train, y_train, mean_120_train):
        super(PreLiveHorseRaceEnv, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.mean_120_train = mean_120_train
        self.current_step = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=-15, high=89, shape=(x_train.shape[1] + 1,), dtype=np.float32
        )

    def step(self, action):
        reward = 0
        # Calculate the next state based on the current state and action
        next_state = np.hstack(
            [self.x_train[self.current_step], self.mean_120_train[self.current_step]]
        )

        # Determine the reward based on the next state or the action taken
        target_value = self.y_train[self.current_step]
        mean_120 = self.mean_120_train[self.current_step]
        if action == Actions.BACK.value:
            if target_value > mean_120:
                reward = target_value - mean_120
            else:
                reward = -10
        elif action == Actions.LAY.value:
            if target_value <= mean_120:
                reward = mean_120 - target_value
            else:
                reward = -10

        self.current_step += 1

        # Check if the environment has reached a terminal state (done)
        if self.current_step >= len(self.x_train):
            done = True
        else:
            done = False

        info = {}

        return next_state, reward, done, info

    def reset(self):
        self.current_step = 0
        return np.hstack(
            [self.x_train[self.current_step], self.mean_120_train[self.current_step]]
        )

    def render(self, mode="human"):
        pass

    def close(self):
        pass


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

    x_train_df, y_train_df, _ = get_train_data(
        "utils/x_train_df.csv",
        "utils/y_train_df.csv",
        onedrive,
        ticks_df,
        regression=True,
    )
    mean_120_train_np = x_train_df["mean_120"].to_numpy()
    x_train_np = x_train_df.drop(columns=["mean_120"]).to_numpy()
    y_train_np = y_train_df.to_numpy()

    env = PreLiveHorseRaceEnv(x_train_np, y_train_np, mean_120_train_np)

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=25000)
    model.save("ppo2_pre_horse_race")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo2_pre_horse_race")

    # Enjoy trained agent
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

    env.close()
