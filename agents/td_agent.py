"""
时序差分学习方法实现，包括SARSA和Q-learning。
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from agents.base_agent import BaseAgent


class TDAgent(BaseAgent):
    """实现时序差分学习的智能体"""

    def __init__(self, state_dim, action_dim, config, device=None):
        """
        初始化时序差分学习智能体

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            config: 配置字典
            device: 设备 (时序差分算法不需要PyTorch)
        """
        super().__init__(state_dim, action_dim, device="cpu")

        # 从配置中提取参数
        self.gamma = config.get('gamma', 0.9)  # 折扣因子
        self.alpha = config.get('alpha', 0.1)  # 学习率
        self.epsilon = config.get('epsilon', 0.1)  # epsilon-greedy策略的epsilon值
        self.algorithm = config.get('td_algorithm', 'sarsa')  # 'sarsa' 或 'q_learning'

        # 初始化状态-动作价值函数
        self.Q = np.zeros((state_dim, action_dim))

        # 为epsilon-greedy策略预计算概率分布
        self.policy = np.ones((state_dim, action_dim)) * self.epsilon / action_dim
        for s in range(state_dim):
            self.policy[s, np.random.randint(action_dim)] = 1.0 - self.epsilon + self.epsilon / action_dim

        # 用于存储当前转换
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = None

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
            # 对于训练，使用epsilon-greedy策略
            if np.random.random() < self.epsilon:
                # 探索：随机选择动作
                return np.random.randint(0, self.action_dim)
            else:
                # 利用：选择最优动作
                return np.argmax(self.Q[state_idx])
        else:
            # 对于评估，使用贪婪策略
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
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

        # 对于SARSA，我们需要知道下一个动作
        if self.algorithm == 'sarsa' and not done:
            self.next_action = self.select_action(next_state)

    def update(self):
        """
        更新智能体

        Returns:
            info: 包含训练信息的字典
        """
        # 检查是否有转换
        if self.state is None:
            return {}

        # 计算当前的Q值
        current_q = self.Q[self.state, self.action]

        # 根据算法计算目标Q值
        if self.algorithm == 'sarsa':
            # SARSA (on-policy TD)
            if self.done:
                target_q = self.reward
            else:
                target_q = self.reward + self.gamma * self.Q[self.next_state, self.next_action]
        else:
            # Q-Learning (off-policy TD)
            if self.done:
                target_q = self.reward
            else:
                target_q = self.reward + self.gamma * np.max(self.Q[self.next_state])

        # 更新Q值
        self.Q[self.state, self.action] += self.alpha * (target_q - current_q)

        # 更新策略 (对于epsilon-greedy策略)
        best_a = np.argmax(self.Q[self.state])
        self.policy[self.state] = np.ones(self.action_dim) * self.epsilon / self.action_dim
        self.policy[self.state, best_a] = 1.0 - self.epsilon + self.epsilon / self.action_dim

        # 清空存储的转换
        td_error = target_q - current_q
        state = self.state

        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None

        if self.algorithm == 'sarsa':
            self.next_action = None

        return {
            "td_error": td_error,
            "q_value": current_q,
            "state": state
        }

    def save(self, path):
        """
        保存智能体

        Args:
            path: 保存路径
        """
        np.savez(path, Q=self.Q, policy=self.policy)

    def load(self, path):
        """
        加载智能体

        Args:
            path: 加载路径
        """
        data = np.load(path, allow_pickle=True)
        self.Q = data['Q']
        self.policy = data['policy']