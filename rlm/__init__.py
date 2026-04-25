"""rlm: Reinforcement Learning with Language Models.

A library for training and evaluating language models using
reinforcement learning techniques.
"""

__version__ = "0.1.0"
__author__ = "rlm contributors"
__license__ = "MIT"

from rlm.trainer import RLMTrainer
from rlm.environment import TextEnvironment
from rlm.reward import RewardModel

__all__ = [
    "RLMTrainer",
    "TextEnvironment",
    "RewardModel",
    "__version__",
]
