"""Starter agent for the Snake reinforcement learning challenge.

The provided RandomAgent is intentionally simple. Candidates should replace it
with their own RL agent, such as Q-learning, DQN, or PPO.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class RandomAgent:
    """A runnable baseline that chooses uniformly random actions."""

    def __init__(self, action_size: int = 3, seed: int | None = None) -> None:
        self.action_size = action_size
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, Any], training: bool = True) -> int:
        del observation, training
        return int(self.rng.integers(self.action_size))

    def observe(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        next_observation: dict[str, Any],
        done: bool,
    ) -> None:
        del observation, action, reward, next_observation, done

    def train_step(self) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"agent_type": "random", "action_size": self.action_size}
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "RandomAgent":
        path = Path(path)
        if not path.exists():
            return cls(seed=seed)

        with path.open("rb") as file:
            payload = pickle.load(file)
        return cls(action_size=int(payload.get("action_size", 3)), seed=seed)


def encode_observation(observation: dict[str, Any]) -> np.ndarray:
    """Example state encoder.

    Candidates are expected to redesign this representation. The starter
    version only includes relative food position and current direction.
    """

    head_x, head_y = observation["head"]
    food_x, food_y = observation["food"]
    grid_size = observation["grid"].shape[0]

    return np.array(
        [
            (food_x - head_x) / grid_size,
            (food_y - head_y) / grid_size,
            observation["direction"] / 3.0,
        ],
        dtype=np.float32,
    )


def build_agent(seed: int | None = None) -> RandomAgent:
    """Factory used by train.py.

    Replace this function when implementing a learning agent.
    """

    return RandomAgent(seed=seed)


def load_agent(path: str | Path, seed: int | None = None) -> RandomAgent:
    """Load a trained agent for evaluation."""

    return RandomAgent.load(path, seed=seed)
