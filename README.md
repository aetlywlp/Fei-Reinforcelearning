# Fei_Zibeng_Reinforcelearning
This GitHub repository is created for practice to get started with reinforcement learning, including code implementations of various algorithms and insights from reading related papers.
## Study notes and implement the code for various classic reinforcement learning algorithms.
### Before everything，I studied [MathFoundationRL taught by Zhao Shiyu](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) for about thirty hours, gaining a contextual understanding of reinforcement learning algorithms.
#### It includes：
- Basic Concepts
- Bellman Equation
- Bellman Optimality Equation
- Value Iteration and Policy Iteration
- Monte Carlo Learning
- Stochastic Approximation and SGD
- Temporal-Difference Learning
- Value Function Approximation
- Policy Gradient Methods
### [Gymnasium_tutorial](https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/)
- A2C（with better advantage GAE，and offpolicy more like PPO）
By using it, I can train a quadruped robot dog (although it looks quite clumsy).
- REINFORCE(An early policy gradient algorithm, the principle of which is to maximize the Monte Carlo mean)
By using it ,I can trian a Inverted Pendulum.
### [reinforcement_learning_by_dennybritz](https://github.com/dennybritz/reinforcement-learning)
Next, I will study, transplant, and organize Denny Britz's code according to the following format.
```
project/
├── agents/                # 智能体实现
│   ├── base_agent.py      # 基类定义接口
│   ├── dqn_agent.py       # DQN实现
│   └── ppo_agent.py       # PPO实现
├── networks/              # 神经网络模型
│   ├── mlp.py             # 多层感知器
│   ├── cnn.py             # 卷积网络
│   └── policy_nets.py     # 策略网络
├── memory/                # 经验回放
│   ├── replay_buffer.py   # 基本回放缓冲区
│   └── prioritized_replay.py # 优先级回放
├── environments/          # 环境封装
│   ├── env_wrappers.py    # 环境预处理
│   └── env_utils.py       # 辅助函数
├── utils/                 # 工具函数
│   ├── logger.py          # 日志工具
│   ├── plot_utils.py      # 绘图函数
│   └── misc.py            # 其他工具
├── config/                # 配置文件
│   ├── dqn_config.yaml    # DQN超参数
│   └── ppo_config.yaml    # PPO超参数
└── train.py               # 训练入口脚本
│
├── requirements.txt              # 项目依赖
├── setup.py                      # 安装脚本
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
```
- MDPs and bella equations
- 

