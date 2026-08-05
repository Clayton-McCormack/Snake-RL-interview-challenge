"""Simplified Snake environment for reinforcement learning interviews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EMPTY = 0
SNAKE_BODY = 1
SNAKE_HEAD = 2
FOOD = 3

DIRECTIONS = np.array(
    [
        [0, -1],  # up
        [1, 0],  # right
        [0, 1],  # down
        [-1, 0],  # left
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class SnakeConfig:
    grid_size: int = 10
    max_steps_without_food: int = 100
    food_reward: float = 10.0
    collision_penalty: float = -10.0
    survival_reward: float = 0.01
    closer_reward: float = 0.1
    farther_penalty: float = -0.1
    timeout_penalty: float = -2.0


class SnakeEnv:
    """Small dependency-light environment with a Gymnasium-like API."""

    action_meanings = {
        0: "straight",
        1: "turn_right",
        2: "turn_left",
    }

    def __init__(self, config: SnakeConfig | None = None, seed: int | None = None):
        self.config = config or SnakeConfig()
        if self.config.grid_size < 5:
            raise ValueError("grid_size must be at least 5")

        self.rng = np.random.default_rng(seed)
        self.snake: list[tuple[int, int]] = []
        self.direction = 1
        self.food = (0, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

    @property
    def action_space_n(self) -> int:
        return 3

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        center = self.config.grid_size // 2
        self.direction = 1
        self.snake = [(center, center), (center - 1, center)]
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.food = self._sample_empty_cell()

        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if action not in self.action_meanings:
            raise ValueError(f"Invalid action {action}; expected 0, 1, or 2")

        old_distance = self._food_distance()
        self._turn(action)

        head = np.array(self.snake[0], dtype=np.int16)
        next_head_array = head + DIRECTIONS[self.direction]
        next_head = (int(next_head_array[0]), int(next_head_array[1]))

        self.steps += 1
        self.steps_since_food += 1

        ate_food = next_head == self.food
        occupied = set(self.snake if ate_food else self.snake[:-1])

        if not self._inside_grid(next_head) or next_head in occupied:
            reward = self.config.collision_penalty
            return self._observation(), reward, True, False, self._info()

        self.snake.insert(0, next_head)

        reward = self.config.survival_reward
        if ate_food:
            self.score += 1
            self.steps_since_food = 0
            reward += self.config.food_reward
            if len(self.snake) < self.config.grid_size * self.config.grid_size:
                self.food = self._sample_empty_cell()
        else:
            self.snake.pop()
            new_distance = self._food_distance()
            if new_distance < old_distance:
                reward += self.config.closer_reward
            elif new_distance > old_distance:
                reward += self.config.farther_penalty

        truncated = self.steps_since_food >= self.config.max_steps_without_food
        if truncated:
            reward += self.config.timeout_penalty

        return self._observation(), float(reward), False, truncated, self._info()

    def render(self) -> str:
        grid = self._grid()
        symbols = {
            EMPTY: ".",
            SNAKE_BODY: "o",
            SNAKE_HEAD: "H",
            FOOD: "*",
        }
        rows = [" ".join(symbols[int(cell)] for cell in row) for row in grid.T]
        return "\n".join(rows)

    def _turn(self, action: int) -> None:
        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

    def _inside_grid(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size

    def _sample_empty_cell(self) -> tuple[int, int]:
        occupied = set(self.snake)
        empty_cells = [
            (x, y)
            for x in range(self.config.grid_size)
            for y in range(self.config.grid_size)
            if (x, y) not in occupied
        ]
        if not empty_cells:
            return self.snake[0]

        index = int(self.rng.integers(len(empty_cells)))
        return empty_cells[index]

    def _food_distance(self) -> int:
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return abs(food_x - head_x) + abs(food_y - head_y)

    def _grid(self) -> np.ndarray:
        grid = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.int8)
        for x, y in self.snake[1:]:
            grid[x, y] = SNAKE_BODY
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        grid[head_x, head_y] = SNAKE_HEAD
        grid[food_x, food_y] = FOOD
        return grid

    def _observation(self) -> dict[str, Any]:
        return {
            "grid": self._grid(),
            "head": np.array(self.snake[0], dtype=np.int16),
            "food": np.array(self.food, dtype=np.int16),
            "direction": int(self.direction),
            "score": int(self.score),
            "steps": int(self.steps),
            "steps_since_food": int(self.steps_since_food),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "score": int(self.score),
            "length": len(self.snake),
            "steps": int(self.steps),
            "steps_since_food": int(self.steps_since_food),
            "food_distance": int(self._food_distance()),
        }
