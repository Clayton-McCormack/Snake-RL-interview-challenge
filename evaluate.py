"""Evaluate a trained Snake agent."""

from __future__ import annotations

import argparse
from statistics import mean

from agent import load_agent
from snake_env import SnakeConfig, SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps-without-food", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model-path", type=str, default="model.pt")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    config = SnakeConfig(
        grid_size=args.grid_size,
        max_steps_without_food=args.max_steps_without_food,
    )
    env = SnakeEnv(config=config, seed=args.seed)
    agent = load_agent(args.model_path, seed=args.seed)

    scores: list[int] = []
    steps: list[int] = []
    successes = 0

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        done = False

        while not done:
            action = agent.act(observation, training=False)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if args.render:
                print(env.render())
                print()

        scores.append(info["score"])
        steps.append(info["steps"])
        successes += int(info["score"] > 0)

    return {
        "average_score": float(mean(scores)),
        "average_steps": float(mean(steps)),
        "success_rate": successes / args.episodes,
        "best_score": float(max(scores)),
    }


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    print(f"model_path={args.model_path}")
    print(f"episodes={args.episodes}")
    print(f"average_score={metrics['average_score']:.2f}")
    print(f"average_steps={metrics['average_steps']:.1f}")
    print(f"success_rate={metrics['success_rate']:.2%}")
    print(f"best_score={metrics['best_score']:.0f}")


if __name__ == "__main__":
    main()
