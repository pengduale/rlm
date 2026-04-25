"""Core module for rlm — reinforcement learning with language models.

This module provides the foundational classes and utilities for training
and evaluating language models using reinforcement learning techniques.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RLMConfig:
    """Configuration class for RLM training.

    Attributes:
        model_name: Pretrained model identifier (HuggingFace hub or local path).
        learning_rate: Learning rate for the optimizer.
        gamma: Discount factor for future rewards.
        clip_epsilon: PPO clipping parameter.
        value_coef: Coefficient for value loss term.
        entropy_coef: Coefficient for entropy bonus term.
        max_grad_norm: Maximum gradient norm for clipping.
        batch_size: Number of samples per training batch.
        num_epochs: Number of optimization epochs per rollout.
        device: Target device ('cpu', 'cuda', or 'auto').
    """

    model_name: str = "gpt2"
    learning_rate: float = 1e-5
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    batch_size: int = 32
    num_epochs: int = 4
    device: str = "auto"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug("RLMConfig initialized: device=%s", self.device)


@dataclass
class RolloutBuffer:
    """Stores transitions collected during environment rollouts."""

    observations: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(
        self,
        obs: Any,
        action: Any,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        """Reset the buffer, discarding all stored transitions."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_returns(
        self, last_value: float = 0.0, gamma: float = 0.99
    ) -> List[float]:
        """Compute discounted returns from stored rewards.

        Args:
            last_value: Bootstrap value for the final state.
            gamma: Discount factor.

        Returns:
            List of discounted return values aligned with stored transitions.
        """
        returns: List[float] = []
        running_return = last_value
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            running_return = reward + gamma * running_return * (1.0 - float(done))
            returns.insert(0, running_return)
        return returns
