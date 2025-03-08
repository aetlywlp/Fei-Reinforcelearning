import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import random

# 简单的多智能体环境
class SimpleMultiAgentEnv:
    """
    两个智能体的简单协作环境
    智能体必须选择相同的动作才能获得正向奖励
    """
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)  # 上下左右
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size-1, shape=(4,), dtype=np.float32
        )  # 两个智能体的位置 [x1, y1, x2, y2]
        self.max_steps = 100
        self.reset()
    
    def reset(self):
        # 随机初始化两个智能体的位置
        self.agent1_pos = np.array([
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ])
        self.agent2_pos = np.array([
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ])
        self.steps = 0
        obs = np.concatenate([self.agent1_pos, self.agent2_pos])
        return obs, {}
    
    def step(self, actions):
        """
        执行两个智能体的动作
        
        参数:
            actions: 包含两个智能体动作的列表 [agent1_action, agent2_action]
        
        返回:
            obs: 新的观察
            rewards: 两个智能体的奖励 [r1, r2]
            done: 是否结束
            info: 额外信息
        """
        agent1_action, agent2_action = actions
        
        # 根据动作更新位置
        # 0: 上, 1: 右, 2: 下, 3: 左
        if agent1_action == 0:
            self.agent1_pos[1] = min(self.agent1_pos[1] + 1, self.grid_size - 1)
        elif agent1_action == 1:
            self.agent1_pos[0] = min(self.agent1_pos[0] + 1, self.grid_size - 1)
        elif agent1_action == 2:
            self.agent1_pos[1] = max(self.agent1_pos[1] - 1, 0)
        elif agent1_action == 3:
            self.agent1_pos[0] = max(self.agent1_pos[0] - 1, 0)
        
        if agent2_action == 0:
            self.agent2_pos[1] = min(self.agent2_pos[1] + 1, self.grid_size - 1)
        elif agent2_action == 1:
            self.agent2_pos[0] = min(self.agent2_pos[0] + 1, self.grid_size - 1)
        elif agent2_action == 2:
            self.agent2_pos[1] = max(self.agent2_pos[1] - 1, 0)
        elif agent2_action == 3:
            self.agent2_pos[0] = max(self.agent2_pos[0] - 1, 0)
        
        # 计算奖励
        # 如果两个智能体在同一位置，获得正奖励
        # 如果两个智能体选择相同动作，获得小的正奖励
        # 其他情况获得小的负奖励
        same_pos = np.array_equal(self.agent1_pos, self.agent2_pos)
        same_action = agent1_action == agent2_action
        
        if same_pos:
            rewards = [10.0, 10.0]  # 合作成功
        elif same_action:
            rewards = [0.5, 0.5]    # 行为协调
        else:
            rewards = [-0.1, -0.1]  # 不协调
        
        # 更新步数
        self.steps += 1
        done = self.steps >= self.max_steps
        terminated = done
        truncated = False
        
        # 构建观察
        obs = np.concatenate([self.agent1_pos, self.agent2_pos])
        
        return obs, rewards, terminated, truncated, {}

class QNetwork(nn.Module):
    """Q网络"""
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, actions, rewards, next_state, done):
        self.buffer.append((state, actions, rewards, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class IndependentQLearningAgent:
    """独立Q学习智能体"""
    def __init__(self, state_dim, action_dim, idx, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.idx = idx  # 智能体索引
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
    
    def select_action(self, state, training=True):
        """选择动作，使用epsilon-greedy策略"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """更新Q网络"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        # 获取当前Q值
        q_values = self.q_network(state_tensor)
        q_value = q_values.gather(0, action_tensor)
        
        # 获取目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor)
            max_next_q = next_q_values.max(0)[0]
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q
        
        # 计算损失并更新
        loss = F.mse_loss(q_value, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class MultiAgentQLearning:
    """多智能体Q学习系统"""
    def __init__(self, env, buffer_size=10000, batch_size=64, update_every=4, target_update=100):
        self.env = env
        self.n_agents = 2  # 假设总是两个智能体
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update = target_update
        
        # 创建智能体
        self.agents = [
            IndependentQLearningAgent(
                state_dim=env.observation_space.shape[0], 
                action_dim=env.action_space.n, 
                idx=i
            )
            for i in range(self.n_agents)
        ]
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)
        
        # 统计数据
        self.training_step = 0
    
    def train(self, n_episodes=1000, max_steps=100):
        """训练多智能体系统"""
        rewards_history = []
        avg_rewards_history = []
        
        for episode in tqdm(range(n_episodes), desc="Training"):
            state, _ = self.env.reset()
            episode_rewards = [0, 0]
            
            for step in range(max_steps):
                # 每个智能体选择动作
                actions = [agent.select_action(state) for agent in self.agents]
                
                # 执行动作
                next_state, rewards, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                
                # 存储经验
                self.memory.add(state, actions, rewards, next_state, done)
                
                # 更新状态和累积奖励
                state = next_state
                episode_rewards = [r1 + r2 for r1, r2 in zip(episode_rewards, rewards)]
                
                # 学习
                if len(self.memory) > self.batch_size and self.training_step % self.update_every == 0:
                    self.learn()
                
                self.training_step += 1
                
                # 更新目标网络
                if self.training_step % self.target_update == 0:
                    for agent in self.agents:
                        agent.update_target_network()
                
                if done:
                    break
            
            # 记录本episode的奖励
            rewards_history.append(episode_rewards)
            avg_reward = np.mean(rewards_history[-100:], axis=0) if len(rewards_history) >= 100 else np.mean(rewards_history, axis=0)
            avg_rewards_history.append(avg_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Rewards: {episode_rewards}, Avg Rewards: {avg_reward}")
        
        return rewards_history, avg_rewards_history
    
    def learn(self):
        """从经验回放缓冲区中学习"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样一批经验
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 更新每个智能体
        for i, agent in enumerate(self.agents):
            for batch_idx in range(self.batch_size):
                state = states[batch_idx]
                action = actions[batch_idx][i]
                reward = rewards[batch_idx][i]
                next_state = next_states[batch_idx]
                done = dones[batch_idx]
                
                agent.update(state, action, reward, next_state, done)
    
    def test(self, n_episodes=10):
        """测试训练好的智能体"""
        test_rewards = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_rewards = [0, 0]
            done = False
            
            while not done:
                # 选择动作
                actions = [agent.select_action(state, training=False) for agent in self.agents]
                
                # 执行动作
                next_state, rewards, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                
                # 更新状态和累积奖励
                state = next_state
                episode_rewards = [r1 + r2 for r1, r2 in zip(episode_rewards, rewards)]
            
            test_rewards.append(episode_rewards)
            print(f"Test Episode {episode}, Rewards: {episode_rewards}")
        
        return test_rewards

def train_multi_agent_system():
    """训练多智能体系统"""
    # 创建环境和多智能体系统
    env = SimpleMultiAgentEnv(grid_size=5)
    marl_system = MultiAgentQLearning(env)
    
    # 训练
    rewards_history, avg_rewards_history = marl_system.train(n_episodes=1000)
    
    # 可视化训练结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    rewards_array = np.array(rewards_history)
    plt.plot(rewards_array[:, 0], alpha=0.3, label='Agent 1')
    plt.plot(rewards_array[:, 1], alpha=0.3, label='Agent 2')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    avg_rewards_array = np.array(avg_rewards_history)
    plt.plot(avg_rewards_array[:, 0], label='Agent 1')
    plt.plot(avg_rewards_array[:, 1], label='Agent 2')
    plt.title('Average Rewards (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('marl_learning_curve.png')
    plt.show()
    
    # 测试
    test_rewards = marl_system.test(n_episodes=10)
    
    return marl_system, rewards_history

if __name__ == "__main__":
    marl_system, rewards_history = train_multi_agent_system()
