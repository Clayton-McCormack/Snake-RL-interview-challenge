# Snake RL Challenge - Clayton McCormack

## Setup

Create a virtual environment and install dependencies:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Run training:

    python train.py --episodes 800 --model-path artifacts/model.pt

Run evaluation:

    python evaluate.py --episodes 100 --model-path artifacts/model.pt --seed 123

## Report

### Algorithm

I used a Deep Q-Network (DQN) with experience replay and a target network. DQN fit this problem's discrete 3-action space and small state space well. I considered PPO but ruled it out given the time constraint, since PPO typically needs more tuning to converge reliably and is better suited to continuous action spaces or more complex environments than this one.

### State design

Rather than feeding the raw grid to the network, I encoded an 11-feature state vector:
- Danger straight, danger right, danger left (whether the next cell in each of those three directions is a wall or the snake's own body)
- Current direction, one-hot encoded (up, right, down, left)
- Food direction relative to the head (up, down, left, right)

This compact representation trains much faster than a raw grid input and is directly interpretable, each feature has an obvious meaning, which made debugging straightforward.

### Reward design

I used the reward shaping already built into `snake_env.py`: +10 for eating food, -10 for collision, a small survival reward per step, +0.1 for moving closer to food, -0.1 for moving farther, and a timeout penalty if the snake goes too long without eating. I kept this default rather than override it, since it already addresses the failure mode the assessment calls out.

The closer/farther shaping reward is a double-edged sword. It gives the agent a dense learning signal early on, letting it improve before it experiences many food events, but it can also teach a locally greedy policy that doesn't account for the snake's growing body. Late in training, the snake sometimes needs to move away from food to avoid trapping itself with its own tail, and a pure distance-based reward doesn't naturally encourage that. The timeout penalty for going too long without eating is what prevents the specific "looping without eating" failure mode.

### Training duration

800 episodes, approximately 3 minutes 36 seconds on CPU (no GPU used).

### Final metrics

Evaluated over 100 episodes per seed, greedy policy:

| Seed | Avg score | Avg steps | Success rate | Best score |
|------|-----------|-----------|---------------|------------|
| 123  | 20.14     | 165.6     | 100%          | 41         |
| 456  | 19.25     | 154.8     | 100%          | 42         |
| 789  | 18.55     | 147.4     | 100%          | 35         |
| 101  | 19.95     | 165.5     | 100%          | 41         |
| 202  | 19.57     | 160.3     | 100%          | 40         |
| **Mean** | **19.49** | **158.7** | **100%** | — |

Results are highly consistent across five random seeds: 100% success rate in every run, and average score ranging only from 18.55 to 20.14 (a spread of under 9%). This indicates the learned policy generalizes across different food placement patterns rather than overfitting to one seed.

### Baseline comparison

I compared against the starter repo's RandomAgent, which selects a uniformly random action at every step with no learning. Evaluated under the same conditions as the trained model, 100 episodes per seed across the same five seeds:

| Seed | Avg score | Avg steps | Success rate | Best score |
|------|-----------|-----------|---------------|------------|
| 123  | 0.15      | 21.8      | 15%           | 1          |
| 456  | 0.08      | 20.7      | 8%            | 1          |
| 789  | 0.10      | 20.0      | 9%            | 2          |
| 101  | 0.07      | 19.8      | 7%            | 1          |
| 202  | 0.17      | 22.1      | 14%           | 3          |
| **Mean** | **0.11** | **20.9** | **10.6%** | — |

The trained DQN agent improved average score by roughly 177x over the random baseline (0.11 to 19.49 mean), raised success rate from 10.6% to 100%, and increased average survival steps more than sevenfold (20.9 to 158.7). This shows the agent learned to actively navigate toward food and avoid collisions, rather than simply surviving passively, since the random agent's occasional low success rate reflects stumbling into food by chance rather than deliberate pursuit.

Evaluation command used for the baseline: `python evaluate_baseline.py --episodes 100 --seed <seed>`.

### Problems encountered and future improvements

- Early training was noisy, average score didn't show a clear upward trend until roughly episode 150-200, consistent with the replay buffer needing to fill with varied experience before learning stabilizes.
- With more time, I'd experiment with a slightly larger hidden layer or a small convolutional encoder over the raw grid to see whether it outperforms the hand-crafted state, particularly for follow-up variations like obstacles or partial observability where the hand-crafted danger features would need redesigning.
- I'd add reward logging over training (not just score/steps) to more precisely diagnose whether the closer/farther shaping is helping or hurting in later episodes, once the snake is long enough that self-collision risk dominates.