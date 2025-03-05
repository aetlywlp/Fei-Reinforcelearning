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
reinforcement-learning-project/
│
├── config/                       # 配置文件目录
│   ├── default_config.yaml       # 默认配置
│   └── custom_configs/           # 自定义配置
│       ├── cartpole_config.yaml
│       └── lunarlander_config.yaml
│
├── src/                          # 源代码目录
│   ├── agents/                   # 智能体实现
│   │   ├── __init__.py
│   │   ├── base_agent.py         # 基础智能体类
│   │   ├── dqn_agent.py          # DQN智能体
│   │   └── ppo_agent.py          # PPO智能体
│   │
│   ├── models/                   # 神经网络模型
│   │   ├── __init__.py
│   │   ├── dqn_network.py        # DQN网络结构
│   │   └── policy_network.py     # 策略网络结构
│   │
│   ├── utils/                    # 工具函数
│   │   ├── __init__.py
│   │   ├── replay_buffer.py      # 经验回放缓冲区
│   │   ├── logger.py             # 日志工具
│   │   └── visualization.py      # 可视化工具
│   │
│   ├── train.py                  # 训练脚本
│   └── evaluate.py               # 评估脚本
│
├── notebooks/                    # Jupyter笔记本
│   ├── exploration.ipynb         # 环境探索
│   └── results_analysis.ipynb    # 结果分析
│
├── results/                      # 结果目录
│   └── <env_id>_<timestamp>/     # 特定训练运行的结果
│       ├── checkpoints/          # 模型检查点
│       ├── logs/                 # 日志文件
│       └── plots/                # 图表和可视化
│
├── tests/                        # 测试目录
│   ├── test_agents.py
│   ├── test_models.py
│   └── test_utils.py
│
├── requirements.txt              # 项目依赖
├── setup.py                      # 安装脚本
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
```
- MDPs and bella equations
- 

