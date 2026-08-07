"""DQN agent for the Snake RL interview challenge.

Implements a small feedforward Deep Q-Network with experience replay and a
target network, wired into the act / observe / train_step / save / load
interface that train.py and evaluate.py already call. train.py and
evaluate.py are left untouched, only this file changes.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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


class DQNAgent:
    """DQN agent matching the act/observe/train_step/save/load interface
    that train.py and evaluate.py already call."""

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        seed: int | None = None,
        lr: float = 1e-3,
        gamma: float = 0.9,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.train_step_count = 0

    def act(self, observation: dict[str, Any], training: bool = True) -> int:
        state = encode_observation(observation)
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def observe(
        self,
        observation: dict[str, Any],
        action: int,
        reward: float,
        next_observation: dict[str, Any],
        done: bool,
    ) -> None:
        state = encode_observation(observation)
        next_state = encode_observation(next_observation)
        self.buffer.push(state, action, reward, next_state, done)
        if done:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self) -> dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent_type": "dqn",
                "state_size": self.state_size,
                "action_size": self.action_size,
                "model_state_dict": self.online_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "DQNAgent":
        path = Path(path)
        agent = cls(seed=seed)
        if not path.exists():
            return agent
        checkpoint = torch.load(path, map_location=agent.device)
        agent.online_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net.load_state_dict(checkpoint["model_state_dict"])
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon_end)
        return agent


def build_agent(seed: int | None = None) -> DQNAgent:
    """Factory used by train.py."""
    return DQNAgent(seed=seed)


def load_agent(path: str | Path, seed: int | None = None) -> DQNAgent:
    """Load a trained agent for evaluation."""
    return DQNAgent.load(path, seed=seed)