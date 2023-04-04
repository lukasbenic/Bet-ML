from collections import defaultdict
from enum import Enum
import os
import gym
from gym.spaces import Discrete, Box
import joblib
import numpy as np
import stable_baselines3
import torch
from onedrive import Onedrive
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from utils.config import app_principal, SITE_URL
from pandas import DataFrame
import pandas as pd

from utils.utils import get_train_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Actions(Enum):
    BACK = 0
    LAY = 1


class PreLiveHorseRaceEnv(gym.Env):
    def __init__(self, obs_df: DataFrame, target_df: DataFrame, threshold: float):
        super(PreLiveHorseRaceEnv, self).__init__()
        self.obs_df = obs_df
        self.target_df = target_df
        self.threshold = threshold
        self.current_step = 0
        self.action_space = Discrete(len(Actions))
        self.observation_space = gym.spaces.Dict(
            {
                column_name: gym.spaces.Box(
                    low=min(obs_df[column_name]),
                    high=max(obs_df[column_name]),
                    shape=obs_df[column_name].shape,
                    dtype=np.float64,
                )
                for column_name in obs_df.columns
            }
        )

    def step(self, action):
        reward = 0
        state = self.obs_df.iloc[self.current_step]
        target = self.target_df.iloc[self.current_step]
        target_value = target["bsps_temp"]
        mean_120 = state["mean_120"]

        print("shape", self.obs_df["mean_120"].shape)

        print("target", target_value)
        print("mean120", mean_120)
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
        next_state = (
            self.obs_df.iloc[self.current_step].to_dict()
            if self.current_step < len(self.obs_df)
            else None
        )
        done = True if not next_state else False
        print("Done", done)
        info = {}

        return next_state, reward, done, info

    def reset(self):
        self.current_step = 0
        return self.obs_df.iloc[self.current_step].to_dict()

    def render(self, mode="human"):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    policy_kwargs = dict(
        net_arch=[64, 64],  # Smaller network architecture
    )

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

    env = DummyVecEnv([lambda: PreLiveHorseRaceEnv(obs_df, target_df, 1.2)])

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        seed=42,
        device=device,
    )
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
