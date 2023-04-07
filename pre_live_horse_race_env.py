from enum import Enum
import gym
from gym.spaces import Discrete, Box
import joblib
import numpy as np
import torch
from RL.TensorBoardCallback import TensorBoardRewardLogger
from onedrive import Onedrive
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from utils.utils import (
    calculate_kelly_stake,
    calculate_margin,
    calculate_odds,
)
from utils.config import app_principal, SITE_URL
from pandas import DataFrame
from stable_baselines3.common.monitor import Monitor


from utils.utils import get_train_data

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Actions(Enum):
    BACK = 0
    LAY = 1


class PreLiveHorseRaceEnv(gym.Env):
    def __init__(self, obs_df: DataFrame, target_df: DataFrame):
        super(PreLiveHorseRaceEnv, self).__init__()
        self.obs_df = obs_df
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

        reward = 0
        state = self.obs_df.iloc[self.current_step]
        target = self.target_df.iloc[self.current_step]

        target_value = target["bsps_temp"]
        mean_120 = state["mean_120"]

        predicted_odds = calculate_odds(target_value, target_value, mean_120)
        stake = calculate_kelly_stake(self.balance, predicted_odds)

        if action == Actions.BACK.value:
            margin = calculate_margin("BACK", stake, predicted_odds, target_value)
            reward = (
                margin * 0.95 if margin > 0 else margin
            )  # remove 5% for commission onn profit
        elif action == Actions.LAY.value:
            margin = calculate_margin("LAY", stake, predicted_odds, target_value)
            reward = (
                margin * 0.95 if margin > 0 else margin
            )  # remove 5% for commission on profit

        self.current_step += 1
        self.balance += reward  # update for accurate kelly_stake
        done = self.current_step >= self.obs_df.shape[0]
        next_state = None if done else self.obs_df.iloc[self.current_step].values
        info = {
            "predicted_odds": predicted_odds,
            "stake": stake,
            "margin": margin,
            "balance": self.balance,
            "step": self.current_step,
        }

        return next_state, reward, done, info

    def reset(self):
        self.current_step = 0
        return self.obs_df.iloc[self.current_step].values

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def create_model(model_name: str, env, device, net_arch):
    # policy_kwargs = dict(net_arch=[64, 64])  # Smaller network architecture
    policy_kwargs = dict(
        net_arch=net_arch,
    )

    model_class = {
        "ppo": PPO,
        "ddpg": DDPG,
    }.get(model_name.lower())

    if model_class is None:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        seed=42,
        device=device,
    )

    return model


def train_model(
    env: PreLiveHorseRaceEnv,
    model_name: str,
    save=False,
    callbacks=True,
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, env, device, net_arch)

    if callbacks:
        eval_callback = EvalCallback(
            Monitor(env),
            log_path=f"RL/{model_name}/logs/",
            eval_freq=1000,
            deterministic=True,
            render=False,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=5000, save_path=f"RL/{model_name}/logs/", name_prefix="model"
        )
        tensorboard_callback = TensorBoardRewardLogger(f"RL/{model_name}/tensorboard/")

        model.learn(
            total_timesteps=25000,
            callback=[eval_callback, checkpoint_callback, tensorboard_callback],
        )
    else:
        model.learn(total_timesteps=25000)

    if save:
        model.save(f"RL/{model_name}/{model_name}_model")


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

    env = PreLiveHorseRaceEnv(obs_df, target_df)
    # lstm_net_arch = dict(pi=[64, 64, ("lstm", 64)], vf=[64, 64, ("lstm", 64)])
    small_net_arch = dict(pi=[32, 32], vf=[32, 32])
    train_model(
        env, save=True, model_name="PPO", callbacks=False, net_arch=small_net_arch
    )
    # model = PPO.load("RL/PPO/PPO_model")
    # print("policy", model.policy)
