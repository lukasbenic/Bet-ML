from collections import defaultdict
from enum import Enum
import os
import gymnasium as gym
from gymnasium import Discrete, Box, Env, Space
import joblib
import numpy as np
from RL.rl_simulation import FlumineRLSimulation
from onedrive import Onedrive
from strategies.rl_strategy import RLStrategy
from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from flumine import clients

from utils.utils import get_strategy_files, train_test_model


class Actions(Enum):
    BACK = 1
    LAY = 2


class FlumineEnv(gym.Env):
    def __init__(self, strategy, markets):
        super(FlumineEnv, self).__init__()
        self.__init(FlumineRLSimulation(clients.SimulatedClient()), strategy)
        self.action_space = Discrete(3)
        self.state_space = Box(low=-np.inf, high=np.inf, dtype=np.float64)
        self.market_files = markets

    def __init(self, flumine_simulator, strategy):
        self.flumine_sim = flumine_simulator
        self.flumine_sim.add_strategy(strategy)

    def step(self, action):
        # Get market data and calculate observation, reward, done, and info
        self.current_time += 1  # Advance the current time
        if self.current_time >= self.end_time:
            self.done = True  # Set done to True if the end time is reached
        else:
            self.flumine_simulation.step()  # Advance the simulation by one step
        market_book = self.flumine_simulation.markets[0].market_book
        observation = [
            market_book.runners[0].ex.available_to_back[0].price,
            market_book.runners[0].ex.available_to_lay[0].price,
            market_book.runners[0].ex.traded_volume[0].price,
        ]
        reward = 0
        done = self.done
        info = {}
        return np.array(observation), reward, done, info

    def reset(self):
        self.current_time = self.start_time
        self.done = False
        self.flumine_simulation.run_to_time(
            self.current_time
        )  # Advance the simulation to the desired start time
        market_book = self.flumine_simulation.markets[0].market_book
        observation = [
            market_book.runners[0].ex.available_to_back[0].price,
            market_book.runners[0].ex.available_to_lay[0].price,
            market_book.runners[0].ex.traded_volume[0].price,
        ]
        return np.array(observation)


if __name__ == "__main__":
    onedrive = Onedrive()
    bsp_df, test_analysis_df, market_files, ticks_df = get_strategy_files(onedrive)

    regressor = model, clm, scaler = train_test_model(
        ticks_df,
        onedrive,
        model="BayesianRidge",
    )

    strategy = RLStrategy(
        informer=regressor,
        ticks_df=ticks_df,
        clm=clm,
        scaler=scaler,
        test_analysis_df=test_analysis_df,
        market_filter={"markets": market_files},
        max_trade_count=100000,
        max_live_trade_count=100000,
        max_order_exposure=10000,
        max_selection_exposure=100000,
    )

    env = DummyVecEnv([lambda: FlumineEnv(strategy, market_files)])

    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo2_pre_horse_race")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo2_pre_horse_race")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
