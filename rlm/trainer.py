"""Trainer module for reinforcement learning from human feedback on language models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from rlm.core import RLMConfig, RolloutBuffer


@dataclass
class TrainerConfig:
    """Configuration for the RLM Trainer."""

    learning_rate: float = 1e-5
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 64
    target_kl: Optional[float] = 0.01
    normalize_advantages: bool = True


class RLMTrainer:
    """Trainer class implementing PPO for language model fine-tuning.

    This trainer handles the core PPO training loop, including advantage
    estimation, policy updates, and value function training.

    Args:
        policy_model: The language model policy to be trained.
        ref_model: The reference (frozen) language model for KL penalty.
        reward_fn: A callable that computes rewards given model outputs.
        config: RLMConfig instance with model/environment settings.
        trainer_config: TrainerConfig instance with training hyperparameters.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable,
        config: RLMConfig,
        trainer_config: Optional[TrainerConfig] = None,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.trainer_config = trainer_config or TrainerConfig()

        # Freeze the reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = Adam(
            self.policy_model.parameters(),
            lr=self.trainer_config.learning_rate,
        )
        self.rollout_buffer = RolloutBuffer()
        self._step = 0

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        Args:
            rewards: Tensor of shape (T,) containing per-step rewards.
            values: Tensor of shape (T,) containing value estimates.
            dones: Tensor of shape (T,) indicating episode terminations.

        Returns:
            Tuple of (advantages, returns) tensors.
        """
        cfg = self.trainer_config
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + cfg.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def ppo_update(
        self,
        observations: Any,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform a single PPO update step.

        Args:
            observations: Batch of input observations/prompts.
            actions: Token actions taken by the policy.
            old_log_probs: Log probabilities from the behavior policy.
            advantages: Computed advantage estimates.
            returns: Computed return targets for the value function.

        Returns:
            Dictionary of training metrics.
        """
        cfg = self.trainer_config
        metrics: Dict[str, float] = {}

        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get current policy outputs
        log_probs, values, entropy = self.policy_model(observations, actions)

        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        value_loss = cfg.value_coef * nn.functional.mse_loss(values.squeeze(-1), returns)
        entropy_loss = -cfg.entropy_coef * entropy.mean()

        total_loss = policy_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        metrics["policy_loss"] = policy_loss.item()
        metrics["value_loss"] = value_loss.item()
        metrics["entropy"] = entropy.mean().item()
        metrics["approx_kl"] = ((ratio - 1) - torch.log(ratio)).mean().item()
        self._step += 1

        return metrics

    @property
    def global_step(self) -> int:
        """Return the current global training step."""
        return self._step
