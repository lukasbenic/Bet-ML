from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class TensorBoardRewardLogger(BaseCallback):
    def __init__(self, log_dir):
        super(TensorBoardRewardLogger, self).__init__()
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", None)
        if reward is not None:
            step = self.num_timesteps
            self.writer.add_scalar("reward", reward.mean(), step)
        return True
