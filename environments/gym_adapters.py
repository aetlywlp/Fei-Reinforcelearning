"""
适配各种Gymnasium环境到我们的框架
"""
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple


def create_gym_env(
        env_id: str,
        render_mode: Optional[str] = None,
        seed: int = 0,
        **kwargs
) -> gym.Env:
    """
    创建任何Gymnasium环境

    Args:
        env_id: 环境ID
        render_mode: 渲染模式 ('human', 'rgb_array', None等)
        seed: 随机种子
        **kwargs: 传递给环境的额外参数

    Returns:
        env: 创建的环境
    """
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    env.reset(seed=seed)
    return env


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """
    获取环境的关键信息

    Args:
        env: Gymnasium环境

    Returns:
        info: 环境信息字典
    """
    info = {
        'env_id': env.spec.id,
        'observation_space': env.observation_space,
        'action_space': env.action_space,
    }

    # 添加观察空间信息
    if hasattr(env.observation_space, 'shape'):
        info['obs_shape'] = env.observation_space.shape
        info['obs_dtype'] = env.observation_space.dtype

    # 添加动作空间信息
    if hasattr(env.action_space, 'n'):
        info['action_dim'] = env.action_space.n
        info['discrete_actions'] = True
    elif hasattr(env.action_space, 'shape'):
        info['action_dim'] = env.action_space.shape[0]
        info['action_low'] = env.action_space.low
        info['action_high'] = env.action_space.high
        info['discrete_actions'] = False

    return info