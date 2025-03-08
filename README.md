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
PY_PROJECT/
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
└──jupyter               # 迁移进入的jupyter类文件，比较独立
└── paper               # 阅读前沿论文
│
├── requirements.txt              # 项目依赖
├── setup.py                      # 安装脚本
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
+--------------------+    配置加载     +--------------------+
|     配置文件       | --------------> |     训练脚本       |
| (config/*.yaml)    |                 | (train.py)         |
+--------------------+                 +----------|---------+
                                                  |
                                                  | 实例化
                                                  v
+--------------------+    创建环境     +--------------------+    创建智能体    +--------------------+
|    环境工具模块    | <-------------> |    训练循环流程    | --------------> |     智能体模块     |
| (environments/*)   |                 |  (train.py主循环)  |                 | (agents/*)         |
+--------------------+                 +----------|---------+                 +----------|----------+
                                                  |                                     |
                                                  | 记录与可视化                        | 创建与使用
                                                  v                                     v
+--------------------+               +--------------------+               +--------------------+
|    工具函数模块    | <-----------> |    日志与监控      | <-----------> |    神经网络模块    |
| (utils/*)          |               | (utils/logger.py)  |               | (networks/*)       |
+--------------------+               +--------------------+               +----------|----------+
                                                                                     |
                                                                                     | 使用
                                                                                     v
                                                                         +--------------------+
                                                                         |    记忆缓冲区     |
                                                                         | (memory/*)         |
                                                                         +--------------------+
```
### code思路
- 创建智能体类：继承BaseAgent并实现算法逻辑
- 创建配置文件：定义算法超参数
- 创建运行脚本：设置训练和评估流程
### Very important learning materials
- [openai spiningup](https://spinningup.qiwihui.com/zh-cn/latest/user/introduction.html):It includes some relatively cutting-edge papers, but the foundation is insufficient.
- [openai_Gymnasium](https://gymnasium.farama.org/):A user-friendly framework,Especially the Gymnasium tutorial.
- [reinforcement-learning](https://github.com/dennybritz/reinforcement-learning)：The classic introductory algorithms are implemented, which are basic and solid, but it requires rewriting Gym and TensorFlow.
- [cleanrl](https://github.com/vwxyzjn/cleanrl):Minimal implementation, emphasizing code transparency and single-file reproducibility. All algorithms are implemented as independent scripts without abstraction layers.
- [stable-baseline3](https://github.com/DLR-RM/stable-baselines3):Stable, modular industrial-grade implementation with high encapsulation, providing a unified API and modular components.

