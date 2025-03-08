"""
运行动态规划算法的脚本。
"""

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dp_agent import DPAgent
from utils.misc import set_random_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行动态规划算法')
    parser.add_argument('--config', type=str, default='config/dp_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='results/dp',
                        help='输出目录')
    parser.add_argument('--method', type=str, default=None,
                        help='动态规划方法: policy_evaluation, policy_iteration, value_iteration')
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_value_function(V, env, title, save_path=None):
    """可视化价值函数"""
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


def run_dp_experiment(config, output_dir, method=None):
    """运行动态规划实验"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置随机种子
    set_random_seed(config.get('seed', 0))

    # 创建环境
    env_kwargs = {}
    if config['env_id'] == 'FrozenLake-v1':
        env_kwargs['is_slippery'] = config.get('env_is_slippery', False)

    env = gym.make(config['env_id'], **env_kwargs)

    # 创建智能体
    agent = DPAgent(
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        config=config
    )

    # 如果指定了方法，覆盖配置中的方法
    if method:
        config['dp_method'] = method

    # 运行指定的动态规划方法
    dp_method = config.get('dp_method', 'policy_evaluation')

    if dp_method == 'policy_evaluation':
        # 初始化随机策略
        agent.policy = np.ones((agent.state_dim, agent.action_dim)) / agent.action_dim

        # 策略评估
        V = agent.policy_evaluation(env)
        print(f"策略评估完成，价值函数: {V}")

        # 可视化价值函数
        if config.get('plot_value_function', True):
            visualize_value_function(
                V, env,
                f"价值函数 (γ={config['gamma']})",
                os.path.join(output_dir, 'value_function.png')
            )

        # 可视化策略
        if config.get('plot_policy', True):
            visualize_policy(
                agent.policy, env,
                "随机策略",
                os.path.join(output_dir, 'random_policy.png')
            )

    elif dp_method == 'policy_iteration':
        # 策略迭代
        V, policy = agent.policy_iteration(env)
        print(f"策略迭代完成，价值函数: {V}")

        # 可视化价值函数
        if config.get('plot_value_function', True):
            visualize_value_function(
                V, env,
                f"最优价值函数 (策略迭代, γ={config['gamma']})",
                os.path.join(output_dir, 'optimal_value_function_pi.png')
            )

        # 可视化策略
        if config.get('plot_policy', True):
            visualize_policy(
                policy, env,
                "最优策略 (策略迭代)",
                os.path.join(output_dir, 'optimal_policy_pi.png')
            )

    elif dp_method == 'value_iteration':
        # 价值迭代
        V, policy = agent.value_iteration(env)
        print(f"价值迭代完成，价值函数: {V}")

        # 可视化价值函数
        if config.get('plot_value_function', True):
            visualize_value_function(
                V, env,
                f"最优价值函数 (价值迭代, γ={config['gamma']})",
                os.path.join(output_dir, 'optimal_value_function_vi.png')
            )

        # 可视化策略
        if config.get('plot_policy', True):
            visualize_policy(
                policy, env,
                "最优策略 (价值迭代)",
                os.path.join(output_dir, 'optimal_policy_vi.png')
            )

    else:
        raise ValueError(f"未知的动态规划方法: {dp_method}")

    # 保存智能体
    agent.save(os.path.join(output_dir, 'dp_agent.npz'))

    # 关闭环境
    env.close()

    return agent


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_dp_experiment(config, args.output_dir, args.method)