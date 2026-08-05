"""Training entrypoint for the Snake RL challenge.

The default implementation runs a random baseline. Candidates should replace
the agent and training logic with a learning algorithm.
"""

from __future__ import annotations

import argparse
from statistics import mean

from agent import build_agent
from snake_env import SnakeConfig, SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps-without-food", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--model-path", type=str, default="artifacts/model.pt")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, float]:
    config = SnakeConfig(
        grid_size=args.grid_size,
        max_steps_without_food=args.max_steps_without_food,
    )
    env = SnakeEnv(config=config, seed=args.seed)
    agent = build_agent(seed=args.seed)

    recent_scores: list[int] = []
    recent_steps: list[int] = []

    for episode in range(1, args.episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        done = False

        while not done:
            action = agent.act(observation, training=True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.observe(observation, action, reward, next_observation, done)
            agent.train_step()

            observation = next_observation

        recent_scores.append(info["score"])
        recent_steps.append(info["steps"])

        if episode % args.log_every == 0:
            window_scores = recent_scores[-args.log_every :]
            window_steps = recent_steps[-args.log_every :]
            print(
                "episode="
                f"{episode:04d} avg_score={mean(window_scores):.2f} "
                f"avg_steps={mean(window_steps):.1f}"
            )

    agent.save(args.model_path)

    return {
        "average_score": float(mean(recent_scores)),
        "average_steps": float(mean(recent_steps)),
        "best_score": float(max(recent_scores)),
    }


def main() -> None:
    args = parse_args()
    metrics = train(args)
    print(
        "saved_model="
        f"{args.model_path} average_score={metrics['average_score']:.2f} "
        f"average_steps={metrics['average_steps']:.1f} "
        f"best_score={metrics['best_score']:.0f}"
    )


if __name__ == "__main__":
    main()
