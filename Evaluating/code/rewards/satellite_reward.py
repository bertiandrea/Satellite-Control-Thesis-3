# satellite_reward.py

from code.utils.satellite_util import quat_diff_rad, quat_diff

import isaacgym #BugFix
import torch

from abc import ABC, abstractmethod
import math
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

class RewardFunction(ABC):
    """Abstract base class for all reward functions."""

    def __init__(self, log_reward: bool, log_reward_interval: int):
        self.global_step: int = 0
        self.log_reward: bool = log_reward
        self.log_reward_interval: int = log_reward_interval

        self.writer: Optional[SummaryWriter] = None
        if self.log_reward:
            self.writer = SummaryWriter(comment="_satellite_reward")

    @abstractmethod
    def compute(
        self,
        quats: torch.Tensor,
        ang_vels: torch.Tensor,
        ang_accs: torch.Tensor,
        goal_quat: torch.Tensor,
        goal_ang_vel: torch.Tensor,
        goal_ang_acc: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the reward value."""
        pass

    def _log_scalar(self, tag: str, value: float):
        """Helper to log scalar values with TensorBoard."""
        if self.log_reward and self.writer:
            if self.global_step % self.log_reward_interval == 0:
                self.writer.add_scalar(tag, value, global_step=self.global_step)

    @staticmethod
    def _assert_valid_tensor(tensor: torch.Tensor, name: str):
        """Ensure tensors have no NaN or Inf values."""
        assert not torch.isnan(tensor).any(), f"{name} has NaN values"
        assert not torch.isinf(tensor).any(), f"{name} has Inf values"

class SimpleReward(RewardFunction):
    def __init__(self, log_reward: bool, log_reward_interval: int):
        super().__init__(log_reward, log_reward_interval)
        self.alpha_q = 1.0
        self.alpha_omega = 0.0
        self.alpha_acc = 0.0

    def compute(
        self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions, step, total_steps
    ):
        phi = quat_diff_rad(quats, goal_quat)
        omega_err = torch.norm(ang_vels - goal_ang_vel, dim=1)
        acc_err = torch.norm(ang_accs - goal_ang_acc, dim=1)

        r_q = self.alpha_q * (1.0 / (1.0 + phi))
        r_omega = r_q * self.alpha_omega * (1.0 / (1.0 + omega_err))
        r_acc = r_q * self.alpha_acc * (1.0 / (1.0 + acc_err))

        reward = r_q + r_omega + r_acc
        self._assert_valid_tensor(reward, "reward")

        self.global_step += 1
        return reward

class ExponentialReward(RewardFunction):
    def __init__(self, log_reward: bool, log_reward_interval: int,
                 lambda_u, lambda_du, max_torque):
        super().__init__(log_reward, log_reward_interval)
        self.prev_quat_err: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None
        self.lambda_u = lambda_u
        self.lambda_du = lambda_du
        self.max_torque = max_torque

    def compute(
        self, quats, ang_vels, ang_accs, goal_quat, goal_ang_vel, goal_ang_acc, actions, step, total_steps
    ):
        phi = quat_diff_rad(quats, goal_quat)
        quat_err = quat_diff(quats, goal_quat)
        ang_vel_err = torch.norm(ang_vels - goal_ang_vel, dim=1)

        r_q = torch.exp(-phi / (0.14 * 2 * math.pi))

        if self.prev_quat_err is None:
            reward = torch.zeros_like(phi)
        else:
            reward = torch.where(quat_err[:, 3] > self.prev_quat_err[:, 3], r_q - 1.0, r_q)
        
        u_sq = torch.sum(actions ** 2, dim=-1)
        u_sq_norm = u_sq / (actions.shape[-1] * self.max_torque ** 2)

        if self.prev_actions is None:
            du_sq = torch.zeros_like(phi)
            du_sq_norm = torch.zeros_like(phi)
        else:
            du_sq = torch.sum((actions - self.prev_actions) ** 2, dim=-1)
            du_sq_norm = du_sq / (actions.shape[-1] * (2.0 * self.max_torque) ** 2)

        r_effort = self.lambda_u * u_sq_norm
        r_smooth = self.lambda_du * du_sq_norm

        final_reward = reward - r_effort - r_smooth
        self._assert_valid_tensor(final_reward, "reward")

        self.prev_quat_err = quat_err.clone()
        self.prev_actions = actions.clone()

        self._log_scalar("Reward_policy/actions[0, 0]", actions[0, 0])
        self._log_scalar("Reward_policy/action[0, 1]", actions[0, 1])
        self._log_scalar("Reward_policy/action[0, 2]", actions[0, 2])
        self._log_scalar("Reward_policy/phi[0]", phi[0].item() * (180 / torch.pi))

        self._log_scalar("Reward_policy/max_torque", actions.abs().max().item())

        ################# MODE LOG #################
        self._log_scalar("Reward_policy/phi_mode", phi.mode().values.item() * (180 / torch.pi))
        self._log_scalar("Reward_policy/phi_mode_count", torch.isclose(phi * (180 / torch.pi), phi.mode().values * (180 / torch.pi), atol=1e-2).sum().item())

        ################# MEAN LOG #################
        self._log_scalar("Reward_policy/phi_mean", phi.mean().item() * (180 / torch.pi))
        self._log_scalar("Reward_policy/ang_vel_err_mean", ang_vel_err.mean().item() * (180 / torch.pi))

        self._log_scalar("Reward_policy/energy_mean", u_sq.mean().item())
        self._log_scalar("Reward_policy/r_effort", r_effort.mean().item())
        self._log_scalar("Reward_policy/du_energy_mean", du_sq.mean().item())
        self._log_scalar("Reward_policy/r_smooth", r_smooth.mean().item())
        self._log_scalar("Reward_policy/max_torque_mean", actions.abs().max(dim=1).values.mean().item())

        self._log_scalar("Reward_policy/reward_mean", final_reward.mean().item())

        self.global_step += 1
        return final_reward

REWARD_MAP = {
    "SimpleReward": SimpleReward,
    "ExponentialReward": ExponentialReward,
}