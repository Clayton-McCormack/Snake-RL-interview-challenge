import numpy as np
import pytest

from snake_env import FOOD, SNAKE_HEAD, SnakeConfig, SnakeEnv


def test_reset_returns_observation_with_expected_fields():
    env = SnakeEnv(seed=7)
    observation, info = env.reset()

    assert set(observation) == {
        "grid",
        "head",
        "food",
        "direction",
        "score",
        "steps",
        "steps_since_food",
    }
    assert observation["grid"].shape == (10, 10)
    assert info["score"] == 0
    assert np.count_nonzero(observation["grid"] == SNAKE_HEAD) == 1
    assert np.count_nonzero(observation["grid"] == FOOD) == 1


def test_step_rejects_invalid_action():
    env = SnakeEnv(seed=7)
    env.reset()

    with pytest.raises(ValueError):
        env.step(99)


def test_environment_eventually_ends_with_repeated_straight_actions():
    env = SnakeEnv(config=SnakeConfig(grid_size=5), seed=7)
    observation, _ = env.reset()
    done = False

    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        if done:
            break

    assert done
    assert info["steps"] > 0
    assert reward <= 0


def test_same_seed_produces_same_start_state():
    first = SnakeEnv(seed=123)
    second = SnakeEnv(seed=123)

    first_observation, _ = first.reset()
    second_observation, _ = second.reset()

    np.testing.assert_array_equal(first_observation["grid"], second_observation["grid"])
