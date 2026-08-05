# Interviewer Guide

## 推荐题目

**Train a Reinforcement Learning Agent to Play Snake**

这个题目适合筛选 AI Engineer、ML Engineer、Robotics RL Intern、Computer Vision + AI junior candidate。仓库已经提供简化版 Snake 环境，候选人应该主要投入在 agent 训练、state 设计、reward 设计、评估和解释。

## 建议流程

第一轮：take-home，3 到 4 小时，不超过半天。

第二轮：15 到 20 分钟代码和思路讲解，再给一个小改动观察候选人的迁移和 debug 能力。

可选 live discussion：60 到 90 分钟，适合更强调现场思考的候选人。

## 候选人需要交付

- 可运行代码：`train.py`、`agent.py`、`README.md`
- 训练好的模型文件：例如 `model.pt`
- 简短报告，说明算法、state/observation、reward、训练时长、最终平均分、成功率、存活步数、遇到的问题和改进方向

## 约束

- 可以用 Python、PyTorch、TensorFlow、Gymnasium
- 不要求最高分，但要求解释清楚训练逻辑
- 不允许直接复制现成完整 Snake AI 项目
- 可以查文档，但要说明外部代码来源
- 重点看候选人是否真的理解训练过程，而不是只套代码

## 评分标准

| 维度 | 权重 | 看什么 |
| --- | ---: | --- |
| RL 基础理解 | 25% | 是否懂 state、action、reward、policy、episode |
| Reward 设计 | 20% | 是否能避免稀疏奖励、惩罚撞墙、鼓励靠近食物 |
| 工程实现 | 20% | 代码能否运行，结构是否清楚 |
| 训练与评估 | 20% | 是否有训练曲线、平均分、baseline、重复测试 |
| Debug/表达 | 15% | 是否能解释失败原因和改进方案 |

## 面试追问

- 你为什么这样设计 state？
- 你的 action space 是什么？左转/右转/直行，还是上下左右？
- 为什么选择 DQN、PPO、Q-learning 或其他算法？
- reward shaping 会不会让模型学到错误行为？
- 怎么判断模型是真的学会了，而不是偶然跑得好？
- 如果换一个更大的地图，模型还能用吗？
- 如果训练不收敛，你会怎么 debug？
- 你有没有设置 random seed？结果稳定吗？
- 你怎么防止 agent 只学会绕圈不吃食物？
- 如果要部署到真实机器人或自动驾驶场景，这个任务和现实有什么差距？

## 现场小改动

从下面选一个即可，不需要全做：

- 食物变成两个，一个加分一个扣分
- 地图尺寸随机变化
- 加入障碍物
- 只能看到局部视野
- reward 变稀疏，只在吃到食物时给奖励

真正懂 training AI 的人，通常能解释为什么模型会学、为什么失败、怎么改。只会套代码的人，一改环境就很难继续推进。

## 机器人方向替代题

如果候选人更偏机器人，可以把 Snake 换成 **2D robot navigation**：小机器人从起点走到目标点，不能撞墙。它更贴近导航、路径规划和安全约束，但评价维度可以保持一致。
