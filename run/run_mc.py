"""
运行蒙特卡罗方法的脚本。
"""

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mc_agent import MCAgent
from utils.misc import set_random_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行蒙特卡罗方法')
    parser.add_argument('--config', type=str, default='config/mc_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='results/mc',
                        help='输出目录')
    parser.add_argument('--method', type=str, default=None,
                        help='蒙特卡罗方法: first_visit 或 every_visit')
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_value_function(Q, env, title, save_path=None):
    """可视化状态价值函数 (取每个状态下最大的动作价值)"""
    V = np.max(Q, axis=1)  # 取每个状态的最大动作价值

    # 对于FrozenLake等网格世界环境
    if hasattr(env, 'nrow') and hasattr(env, 'ncol'):
        V_grid = V.reshape(env.nrow, env.ncol)

        plt.figure(figsize=(8, 6))
        sns.heatmap(V_grid, annot=True, fmt='.3f', cmap='coolwarm', cbar=True)
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()
    else:
        # 对于其他环境，简单绘制价值函数
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(V)), V)
        plt.title(title)
        plt.xlabel('State')
        plt.ylabel('Value')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()


def visualize_policy(policy, env, title, save_path=None):
    """可视化策略"""
    # 对于FrozenLake等网格世界环境
    if hasattr(env, 'nrow') and hasattr(env, 'ncol'):
        # 将策略转换为动作索引
        policy_indices = np.argmax(policy, axis=1)

        # 转换为箭头符号: ↑ → ↓ ←
        arrows = [['↑', '→', '↓', '←'][a] for a in policy_indices]
        arrows_grid = np.array(arrows).reshape(env.nrow, env.ncol)

        # 创建文本网格
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)
        ax.set_xticks(np.arange(env.ncol))
        ax.set_yticks(np.arange(env.nrow))
        ax.set_yticklabels(range(env.nrow))
        ax.set_xticklabels(range(env.ncol))

        # 添加网格线
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(-0.5, env.ncol, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.nrow, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

        # 添加箭头文本
        for i in range(env.nrow):
            for j in range(env.ncol):
                ax.text(j, i, arrows_grid[i, j], ha='center', va='center', fontsize=20)

        # 反转y轴使得(0,0)在左上角
        ax.invert_yaxis()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    else:
        # 对于其他环境，简单绘制策略
        plt.figure(figsize=(12, 6))
        for s in range(policy.shape[0]):
            plt.subplot(1, policy.shape[0], s + 1)
            plt.bar(range(policy.shape[1]), policy[s])
            plt.title(f'State {s}')
            plt.xlabel('Action')
            plt.ylabel('Probability')

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()


def run_mc_experiment(config, output_dir, method=None):
    """运行蒙特卡罗实验"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置随机种子
    set_random_seed(config.get('seed', 0))

    # 创建环境
    env_kwargs = {}
    if config['env_id'] == 'FrozenLake-v1':
        env_kwargs['is_slippery'] = config.get('env_is_slippery', False)

    env = gym.make(config['env_id'], **env_kwargs)

    # 如果指定了方法，覆盖配置中的方法
    if method:
        config['mc_method'] = method

    # 创建智能体
    agent = MCAgent(
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        config=config
    )

    # 训练参数
    num_episodes = config.get('num_episodes', 10000)
    eval_freq = config.get('eval_freq', 500)
    eval_episodes = config.get('eval_episodes', 10)

    # 记录训练过程
    episode_returns = []
    eval_returns = []
    eval_points = []

    # 训练循环
    for episode in tqdm(range(num_episodes), desc="Training"):
        # 重置环境
        state, _ = env.reset()
        done = False

        # 收集一个完整的情节
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            state = next_state

        # 更新智能体
        update_info = agent.update()
        episode_returns.append(update_info.get('episode_return', 0))

        # 定期评估
        if (episode + 1) % eval_freq == 0:
            eval_return = evaluate_agent(agent, env, eval_episodes)
            eval_returns.append(eval_return)
            eval_points.append(episode + 1)

            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Eval Return: {eval_return:.4f}, "
                  f"Mean Q: {np.mean(agent.Q):.4f}")

    # 可视化学习曲线
    if config.get('plot_returns', True):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(episode_returns)
        plt.title('Training Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return')

        plt.subplot(1, 2, 2)
        plt.plot(eval_points, eval_returns)
        plt.title('Evaluation Returns')
        plt.xlabel('Episode')
        plt.ylabel('Average Return')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
        plt.show()

    # 可视化最终价值函数
    if config.get('plot_value_function', True):
        visualize_value_function(
            agent.Q, env,
            f"状态-动作价值函数 ({config['mc_method']} MC, γ={config['gamma']})",
            os.path.join(output_dir, 'value_function.png')
        )

    # 可视化最终策略
    if config.get('plot_policy', True):
        visualize_policy(
            agent.policy, env,
            f"策略 ({config['mc_method']} MC)",
            os.path.join(output_dir, 'policy.png')
        )

    # 保存智能体
    agent.save(os.path.join(output_dir, 'mc_agent.npz'))

    # 关闭环境
    env.close()

    return agent


def evaluate_agent(agent, env, num_episodes):
    """评估智能体的性能"""
    total_returns = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += reward
            state = next_state

        total_returns += episode_return

    return total_returns / num_episodes


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_mc_experiment(config, args.output_dir, args.method)