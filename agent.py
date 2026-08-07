"""Agent for the Snake RL interview challenge.

Stage 2: added the Q-network and replay buffer building blocks for DQN.
Still using the random baseline agent as the active policy while these are
wired together in the next step.
"""

from __future__ import annotations

import pickle
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

STATE_SIZE = 11
ACTION_SIZE = 3

# Matches snake_env.py's DIRECTIONS array exactly: index 0/1/2/3 = up/right/
# down/left, in (x, y) coordinates where "up" decreases y.
DIRECTION_VECTORS = np.array(
    [
        [0, -1],  # up
        [1, 0],  # right
        [0, 1],  # down
        [-1, 0],  # left
    ]
)

SNAKE_BODY = 1
SNAKE_HEAD = 2
# FOOD is intentionally excluded, walking onto it is the goal, not a collision.


def _is_blocked(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    """True if pos is out of bounds or occupied by the snake's own body/head."""
    grid_size = grid.shape[0]
    x, y = pos
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return True
    cell = grid[x, y]
    return cell in (SNAKE_BODY, SNAKE_HEAD)


def encode_observation(observation: dict[str, Any]) -> np.ndarray:
    """Compact 11-feature state instead of feeding the raw grid to the network:

    [danger_straight, danger_right, danger_left,
     moving_up, moving_right, moving_down, moving_left,
     food_up, food_down, food_left, food_right]
    """
    grid = observation["grid"]
    head_x, head_y = observation["head"]
    food_x, food_y = observation["food"]
    direction = observation["direction"]

    straight_dir = direction
    right_dir = (direction + 1) % 4
    left_dir = (direction - 1) % 4

    def next_pos(d: int) -> tuple[int, int]:
        dx, dy = DIRECTION_VECTORS[d]
        return int(head_x + dx), int(head_y + dy)

    danger_straight = _is_blocked(grid, next_pos(straight_dir))
    danger_right = _is_blocked(grid, next_pos(right_dir))
    danger_left = _is_blocked(grid, next_pos(left_dir))

    moving_up = direction == 0
    moving_right = direction == 1
    moving_down = direction == 2
    moving_left = direction == 3

    food_up = food_y < head_y
    food_down = food_y > head_y
    food_left = food_x < head_x
    food_right = food_x > head_x

    return np.array(
        [
            danger_straight, danger_right, danger_left,
            moving_up, moving_right, moving_down, moving_left,
            food_up, food_down, food_left, food_right,
        ],
        dtype=np.float32,
    )


class QNetwork(nn.Module):
    """Small MLP mapping an 11-feature state to a Q-value per action."""

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Uniform random experience replay, breaks correlation between
    consecutive steps so the network doesn't overfit to recent experience."""

    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class RandomAgent:
    """Baseline agent, still the active policy until DQNAgent is wired up."""

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


def build_agent(seed: int | None = None) -> RandomAgent:
    """Factory used by train.py."""
    return RandomAgent(seed=seed)


def load_agent(path: str | Path, seed: int | None = None) -> RandomAgent:
    """Load a trained agent for evaluation."""
    return RandomAgent.load(path, seed=seed)