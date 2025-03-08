"""
动态规划算法实现，包括策略评估、策略改进和价值迭代。
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from agents.base_agent import BaseAgent


class DPAgent(BaseAgent):
    """实现动态规划算法的智能体"""

    def __init__(self, state_dim, action_dim, config, device=None):
        """
        初始化动态规划智能体

        Args:
            state_dim: 状态空间维度 (对于表格型环境，应该是离散状态数量)
            action_dim: 动作空间维度 (对于表格型环境，应该是离散动作数量)
            config: 配置字典，包含算法参数
            device: 设备 (动态规划不需要PyTorch，所以忽略此参数)
        """
        super().__init__(state_dim, action_dim, device="cpu")  # DP不需要GPU

        # 从配置中提取参数
        self.gamma = config.get('gamma', 0.9)  # 折扣因子
        self.theta = config.get('theta', 1e-6)  # 收敛阈值
        self.max_iterations = config.get('max_iterations', 1000)  # 最大迭代次数

        # 初始化价值函数
        self.V = np.zeros(state_dim)

        # 初始化策略 (可以是确定性或随机策略)
        if config.get('policy_type', 'random') == 'random':
            # 随机策略：对每个状态，所有动作概率相等
            self.policy = np.ones((state_dim, action_dim)) / action_dim
        else:
            # 确定性策略：每个状态对应一个动作
            self.policy = np.zeros((state_dim, action_dim))
            # 初始动作可以随机选择或根据配置指定
            for s in range(state_dim):
                a = np.random.randint(0, action_dim)
                self.policy[s, a] = 1.0

        # 保存环境模型 (需要在policy_evaluation方法中设置)
        self.P = None  # 状态转移概率和奖励的字典

    def policy_evaluation(self, env) -> np.ndarray:
        """
        策略评估算法：计算给定策略的价值函数

        Args:
            env: 具有P属性的环境，提供状态转移概率和奖励

        Returns:
            V: 更新后的价值函数
        """
        # 获取环境模型 (对于表格型环境，应有P属性)
        if not hasattr(env, 'P'):
            raise ValueError("环境必须有P属性，提供状态转移概率和奖励")
        self.P = env.P

        # 初始化价值函数
        V = np.zeros(self.state_dim)

        # 迭代直到收敛
        for i in range(self.max_iterations):
            delta = 0
            # 对每个状态
            for s in range(self.state_dim):
                v = V[s]

                # 计算状态价值 V(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
                new_v = 0
                for a in range(self.action_dim):
                    for prob, next_state, reward, done in self.P[s][a]:
                        new_v += self.policy[s, a] * prob * (reward + self.gamma * V[next_state] * (1 - done))

                V[s] = new_v
                delta = max(delta, abs(v - V[s]))

            # 检查收敛
            if delta < self.theta:
                break

        # 更新智能体的价值函数
        self.V = V
        return V

    def policy_improvement(self) -> bool:
        """
        策略改进算法：根据当前价值函数改进策略

        Returns:
            policy_stable: 策略是否稳定（没有变化）
        """
        policy_stable = True

        # 对每个状态
        for s in range(self.state_dim):
            old_action = np.argmax(self.policy[s])

            # 找到使状态-动作价值最大化的动作
            action_values = np.zeros(self.action_dim)
            for a in range(self.action_dim):
                for prob, next_state, reward, done in self.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * self.V[next_state] * (1 - done))

            best_action = np.argmax(action_values)

            # 更新策略为确定性策略 (对最佳动作概率为1，其他为0)
            self.policy[s] = np.zeros(self.action_dim)
            self.policy[s, best_action] = 1.0

            # 检查策略是否变化
            if old_action != best_action:
                policy_stable = False

        return policy_stable

    def policy_iteration(self, env) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略迭代算法：交替进行策略评估和策略改进

        Args:
            env: 具有P属性的环境

        Returns:
            V: 最优价值函数
            policy: 最优策略
        """
        # 获取环境模型
        if not hasattr(env, 'P'):
            raise ValueError("环境必须有P属性，提供状态转移概率和奖励")
        self.P = env.P

        # 迭代直到策略稳定
        for i in range(self.max_iterations):
            # 1. 策略评估
            self.policy_evaluation(env)

            # 2. 策略改进
            policy_stable = self.policy_improvement()

            # 如果策略稳定，结束迭代
            if policy_stable:
                break

        return self.V, self.policy

    def value_iteration(self, env) -> Tuple[np.ndarray, np.ndarray]:
        """
        价值迭代算法：直接计算最优价值函数，然后提取最优策略

        Args:
            env: 具有P属性的环境

        Returns:
            V: 最优价值函数
            policy: 最优策略
        """
        # 获取环境模型
        if not hasattr(env, 'P'):
            raise ValueError("环境必须有P属性，提供状态转移概率和奖励")
        self.P = env.P

        # 初始化价值函数
        V = np.zeros(self.state_dim)

        # 迭代直到收敛
        for i in range(self.max_iterations):
            delta = 0
            # 对每个状态
            for s in range(self.state_dim):
                v = V[s]

                # 计算每个动作的价值，并取最大值
                action_values = np.zeros(self.action_dim)
                for a in range(self.action_dim):
                    for prob, next_state, reward, done in self.P[s][a]:
                        action_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))

                V[s] = np.max(action_values)
                delta = max(delta, abs(v - V[s]))

            # 检查收敛
            if delta < self.theta:
                break

        # 提取最优策略
        policy = np.zeros((self.state_dim, self.action_dim))
        for s in range(self.state_dim):
            action_values = np.zeros(self.action_dim)
            for a in range(self.action_dim):
                for prob, next_state, reward, done in self.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))

            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0

        # 更新智能体的价值函数和策略
        self.V = V
        self.policy = policy

        return V, policy

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

        # 根据策略选择动作
        if np.sum(self.policy[state_idx]) > 0:  # 如果有有效策略
            # 确定性策略：选择概率最高的动作
            if np.max(self.policy[state_idx]) == 1.0:
                return np.argmax(self.policy[state_idx])
            # 随机策略：根据概率分布采样
            else:
                return np.random.choice(self.action_dim, p=self.policy[state_idx])
        else:
            # 如果没有有效策略，随机选择
            return np.random.randint(0, self.action_dim)

    def update(self):
        """
        更新智能体 (对于动态规划，主要步骤在policy_evaluation中完成)

        Returns:
            info: 包含训练信息的字典
        """
        # 对于动态规划，这个方法可以留空或返回一些统计信息
        return {"value_mean": np.mean(self.V),
                "value_max": np.max(self.V),
                "value_min": np.min(self.V)}

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储转换 (动态规划不使用经验回放，所以此方法为空)
        """
        pass

    def save(self, path):
        """
        保存智能体

        Args:
            path: 保存路径
        """
        np.savez(path, V=self.V, policy=self.policy)

    def load(self, path):
        """
        加载智能体

        Args:
            path: 加载路径
        """
        data = np.load(path, allow_pickle=True)
        self.V = data['V']
        self.policy = data['policy']