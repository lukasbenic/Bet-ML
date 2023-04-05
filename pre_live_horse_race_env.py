from enum import Enum
import gym
from gym.spaces import Discrete, Box
import joblib
import numpy as np
import torch
from RL.TensorBoardCallback import TensorBoardRewardLogger
from onedrive import Onedrive
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
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
        reward = 0
        state = self.obs_df.iloc[self.current_step]
        target = self.target_df.iloc[self.current_step]

        target_value = target["bsps_temp"]
        mean_120 = state["mean_120"]

        if action == Actions.BACK.value:
            if target_value > mean_120:
                reward = (target_value - mean_120) * 0.95  # remove 5% for commission
            else:
                reward = -20
        elif action == Actions.LAY.value:
            if target_value <= mean_120:
                reward = (mean_120 - target_value) * 0.95  # remove 5% for commission
            else:
                reward = -10
        self.current_step += 1
        done = self.current_step >= self.obs_df.shape[0]
        next_state = None if done else self.obs_df.iloc[self.current_step].values
        info = {}

        return next_state, reward, done, info

    def reset(self):
        self.current_step = 0
        return self.obs_df.iloc[self.current_step].values

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def train_model(env: PreLiveHorseRaceEnv, save=False):
    policy_kwargs = dict(
        net_arch=[64, 64],  # Smaller network architecture
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        seed=42,
        device=device,
    )
    # Create the EvalCallback with TensorBoard logging enabled
    log_path = "RL/logs/"
    eval_env = Monitor(
        env
    )  # You can use the same environment for evaluation, or create a new one if needed
    eval_callback = EvalCallback(
        eval_env, log_path=log_path, eval_freq=1000, deterministic=True, render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, save_path="./logs/", name_prefix="rl_model"
    )
    tensorboard_callback = TensorBoardRewardLogger("RL/tensorboard/")

    model.learn(
        total_timesteps=25000,
        callback=[eval_callback, checkpoint_callback, tensorboard_callback],
    )

    if save:
        model.save("RL/ppo_pre_horse_race")

    return model


def test_model(env, model=None, model_path="RL/ppo_pre_horse_race"):
    if not model:
        model = PPO.load(model_path)

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print("action", action, "_states", _states, "reward", rewards, "done", done)

        env.render()

    env.close()


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
    # print("target_df shape", target_df.shape)
    # print("obs_df shape", obs_df.shape)
    env = DummyVecEnv([lambda: PreLiveHorseRaceEnv(obs_df, target_df)])
    model = train_model(env, save=False)
    # test_model(env, model=model)
