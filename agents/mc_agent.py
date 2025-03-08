"""
蒙特卡罗方法实现，包括策略评估和策略控制。
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from agents.base_agent import BaseAgent


class MCAgent(BaseAgent):
    """实现蒙特卡罗方法的智能体"""

    def __init__(self, state_dim, action_dim, config, device=None):
        """
        初始化蒙特卡罗智能体

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            config: 配置字典
            device: 设备 (蒙特卡罗方法不需要PyTorch)
        """
        super().__init__(state_dim, action_dim, device="cpu")

        # 从配置中提取参数
        self.gamma = config.get('gamma', 0.9)  # 折扣因子
        self.method = config.get('mc_method', 'every_visit')  # first_visit 或 every_visit
        self.epsilon = config.get('epsilon', 0.1)  # epsilon-greedy策略的epsilon值

        # 初始化状态-动作价值函数
        self.Q = np.zeros((state_dim, action_dim))

        # 初始化策略
        if config.get('policy_type', 'epsilon_greedy') == 'epsilon_greedy':
            # epsilon-greedy策略：对最优动作以1-epsilon的概率选择，其余动作平分epsilon
            self.policy = np.ones((state_dim, action_dim)) * self.epsilon / action_dim
            # 初始时每个状态随机选择一个动作作为贪婪动作
            for s in range(state_dim):
                self.policy[s, np.random.randint(action_dim)] = 1.0 - self.epsilon + self.epsilon / action_dim
        else:
            # 完全随机策略
            self.policy = np.ones((state_dim, action_dim)) / action_dim

        # 用于记录状态-动作对的访问次数
        self.N = np.zeros((state_dim, action_dim))

        # 用于存储单个情节的数据
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def select_action(self, state, training=True):
        """
        根据当前策略选择动作

        Args:
            state: 当前状态
            training: 是否处于训练模式

        Returns:
            action: 选择的动作
        """
        # 对于表格型环境，state通常是一个整数索引
        state_idx = state

        if training:
            # 按照策略选择动作
            return np.random.choice(self.action_dim, p=self.policy[state_idx])
        else:
            # 评估时使用贪婪策略
            return np.argmax(self.Q[state_idx])

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储转换

        Args:
            state: 当前状态
            action: 选择的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def update(self):
        """
        更新智能体 (当一个情节结束时调用)

        Returns:
            info: 包含训练信息的字典
        """
        # 检查是否有完整的情节
        if not self.episode_states:
            return {}

        # 计算每一步的回报
        G = 0
        returns = []
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # 更新状态-动作价值函数
        if self.method == 'first_visit':
            # First-visit MC: 对每个状态-动作对，只考虑第一次访问
            visited = set()
            for t in range(len(self.episode_states)):
                s = self.episode_states[t]
                a = self.episode_actions[t]
                sa_pair = (s, a)

                if sa_pair not in visited:
                    visited.add(sa_pair)
                    self.N[s, a] += 1
                    self.Q[s, a] += (returns[t] - self.Q[s, a]) / self.N[s, a]
        else:
            # Every-visit MC: 考虑每次访问
            for t in range(len(self.episode_states)):
                s = self.episode_states[t]
                a = self.episode_actions[t]

                self.N[s, a] += 1
                self.Q[s, a] += (returns[t] - self.Q[s, a]) / self.N[s, a]

        # 更新策略 (对于epsilon-greedy策略)
        for s in range(self.state_dim):
            best_a = np.argmax(self.Q[s])

            # 更新为epsilon-greedy策略
            self.policy[s] = np.ones(self.action_dim) * self.epsilon / self.action_dim
            self.policy[s, best_a] = 1.0 - self.epsilon + self.epsilon / self.action_dim

        # 清空情节数据
        episode_length = len(self.episode_states)
        episode_return = sum(self.episode_rewards)

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        return {
            "episode_length": episode_length,
            "episode_return": episode_return,
            "mean_value": np.mean(self.Q),
            "max_value": np.max(self.Q)
        }

    def save(self, path):
        """
        保存智能体

        Args:
            path: 保存路径
        """
        np.savez(path, Q=self.Q, policy=self.policy, N=self.N)

    def load(self, path):
        """
        加载智能体

        Args:
            path: 加载路径
        """
        data = np.load(path, allow_pickle=True)
        self.Q = data['Q']
        self.policy = data['policy']
        self.N = data['N']