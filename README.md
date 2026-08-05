# Train a Reinforcement Learning Agent to Play Snake

This repository is a starter kit for an interview challenge. A simplified Snake
environment is provided so candidates can focus on reinforcement learning
decisions: state design, reward design, training method, evaluation, and
debugging.

The starter agent is intentionally weak. It runs end to end, but it is only a
random baseline.

## Candidate Task

Given the simplified Snake environment in `snake_env.py`, train an AI agent that
survives as long as possible and eats food consistently.

You may use Python with your preferred RL stack, including PyTorch, TensorFlow,
Gymnasium, Stable-Baselines3, or a hand-written implementation. You do not need
to achieve a perfect score. You do need to explain your training logic clearly.

## Deliverables

Submit:

- `train.py`: training entrypoint
- `agent.py`: model, policy, replay buffer, feature extraction, or agent logic
- `README.md`: setup instructions and a short report
- `model.pt` or equivalent trained model checkpoint
- Any extra files needed to reproduce your results

Your report should explain:

- Algorithm used, such as Q-learning, DQN, PPO, or another method
- State or observation design
- Reward function design
- Training duration and compute used
- Final average score, success rate, and average survival steps
- Problems encountered and future improvements

## Environment Summary

The environment uses a square grid. The snake starts near the center and must
eat food while avoiding walls and its own body.

Action space:

- `0`: go straight
- `1`: turn right
- `2`: turn left

Observation returned by `SnakeEnv.step()`:

- `grid`: 2D array with empty cells, snake body, snake head, and food
- `head`: current head position
- `food`: current food position
- `direction`: current direction index
- `score`: number of food items eaten in the episode
- `steps`: total episode steps
- `steps_since_food`: steps since the last food was eaten

You may transform this observation into any state representation you think is
appropriate.

## Reward Design

The starter environment includes a simple shaped reward. You may keep it, tune
it, or replace it if you can justify the change.

Example reward ideas:

- Eat food: `+10`
- Hit a wall or self: `-10`
- Survive one step: small positive or negative value
- Move closer to food: small positive value
- Move farther from food: small negative value
- Go too long without eating: light penalty or episode truncation

Be prepared to discuss whether reward shaping could teach the wrong behavior,
such as looping without eating.

## Getting Started

Create a virtual environment and install starter dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the provided tests:

```bash
python -m pytest -q
```

Run the random baseline:

```bash
python train.py --episodes 50 --model-path artifacts/random-baseline.pt
python evaluate.py --episodes 20 --model-path artifacts/random-baseline.pt
```

## Suggested Timebox

- Take-home: 3 to 4 hours
- Live discussion: 60 to 90 minutes
- Follow-up walkthrough: 15 to 20 minutes

## Rules

- Do not copy a complete existing Snake AI project.
- You may use documentation, libraries, examples, and small snippets, but cite
  external code sources in your report.
- Keep the environment simple unless you are explicitly asked to extend it.
- Prioritize clear reasoning, reproducibility, and debuggability over maximum
  score.

## Evaluation Guidance

A typical evaluation command:

```bash
python evaluate.py --episodes 100 --model-path model.pt --seed 123
```

Suggested metrics:

- Average score
- Average survival steps
- Success rate, where success means eating at least one food item
- Variance across multiple random seeds

## Follow-Up Modifications

During the technical discussion, you may be asked to adapt your solution to one
small change:

- Two food types: one positive and one negative
- Random map size
- Obstacles
- Partial observability
- Sparse rewards only when food is eaten
