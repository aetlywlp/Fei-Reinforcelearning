"""
Utility functions for working with environments.

This module provides utility functions for creating, modifying, and
interacting with Gymnasium environments.
"""

from typing import Dict, Any, Tuple, List, Optional, Union, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import matplotlib.pyplot as plt
from environments.env_wrappers import (
    FrameStack, FrameSkip, GrayScaleObservation, ResizeObservation,
    NormalizeObservation, ClipReward, TimeLimit, MonitorEpisodeStats
)
from environments.gym_adapters import create_gym_env, get_env_info











def make_atari_env(env_id, seed=0, frame_stack=4, clip_rewards=True):
    """创建Atari环境并应用常用包装器"""
    from gymnasium.wrappers import AtariPreprocessing, FrameStack as GymFrameStack, ClipReward

    # 创建环境
    env = gym.make(env_id, render_mode=None)

    # 设置随机种子
    env.reset(seed=seed)

    # 应用Atari预处理
    env = AtariPreprocessing(
        env,
        frame_skip=4,  # 跳过4帧
        grayscale_obs=True,  # 转为灰度图
        scale_obs=True,  # 将像素值缩放到[0,1]
        terminal_on_life_loss=False
    )

    # 帧堆叠
    if frame_stack > 1:
        env = GymFrameStack(env, frame_stack)

    # 奖励裁剪
    if clip_rewards:
        env = ClipReward(env, -1.0, 1.0)

    return env


def make_mujoco_env(env_id, seed=0, normalize_obs=True):
    """创建MuJoCo环境并应用常用包装器"""
    from gymnasium.wrappers import NormalizeObservation, RecordEpisodeStatistics

    # 创建环境
    env = gym.make(env_id)

    # 设置随机种子
    env.reset(seed=seed)

    # 记录统计信息
    env = RecordEpisodeStatistics(env)

    # 观察归一化
    if normalize_obs:
        env = NormalizeObservation(env)

    return env


def make_env(
    env_id: str,
    seed: int = 0,
    frame_stack: Optional[int] = None,
    frame_skip: Optional[int] = None,
    normalize_obs: bool = False,
    clip_rewards: bool = False,
    time_limit: Optional[int] = None,
    monitor: bool = True,
    render_mode: Optional[str] = None,
    **kwargs
) -> gym.Env:
    """Create a Gymnasium environment with common wrappers.
    
    Args:
        env_id: Identifier of the environment
        seed: Random seed
        frame_stack: Number of frames to stack
        frame_skip: Number of frames to skip
        normalize_obs: Whether to normalize observations
        clip_rewards: Whether to clip rewards to [-1, 1]
        time_limit: Maximum number of steps per episode
        monitor: Whether to monitor episode statistics
        render_mode: Render mode for the environment
        **kwargs: Additional arguments for environment creation
    
    Returns:
        env: The created environment
    """
    # Create environment
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    
    # Seed environment
    env.reset(seed=seed)
    
    # Apply time limit wrapper
    if time_limit is not None:
        env = TimeLimit(env, max_steps=time_limit)
    
    # Apply frame skip wrapper
    if frame_skip is not None and frame_skip > 1:
        env = FrameSkip(env, skip=frame_skip)
    
    # Check if environment has image observations
    if len(env.observation_space.shape) == 3:
        # For Atari and other image-based environments
        # Apply grayscale wrapper
        if env.observation_space.shape[0] == 3:  # RGB image
            env = GrayScaleObservation(env, keep_dim=True)
        
        # Apply resize wrapper if needed
        if env.observation_space.shape[1] > 84 or env.observation_space.shape[2] > 84:
            env = ResizeObservation(env, shape=(84, 84))
    
    # Apply frame stack wrapper
    if frame_stack is not None and frame_stack > 1:
        env = FrameStack(env, n_frames=frame_stack)
    
    # Apply normalization wrapper
    if normalize_obs:
        env = NormalizeObservation(env)
    
    # Apply reward clipping wrapper
    if clip_rewards:
        env = ClipReward(env)
    
    # Apply monitoring wrapper
    if monitor:
        env = MonitorEpisodeStats(env)
    
    return env


def make_vec_env(
    env_id: str,
    num_envs: int = 4,
    seed: int = 0,
    frame_stack: Optional[int] = None,
    frame_skip: Optional[int] = None,
    normalize_obs: bool = False,
    clip_rewards: bool = False,
    time_limit: Optional[int] = None,
    monitor: bool = True,
    **kwargs
) -> gym.vector.VectorEnv:
    """Create a vectorized Gymnasium environment with common wrappers.
    
    Args:
        env_id: Identifier of the environment
        num_envs: Number of parallel environments
        seed: Random seed
        frame_stack: Number of frames to stack
        frame_skip: Number of frames to skip
        normalize_obs: Whether to normalize observations
        clip_rewards: Whether to clip rewards
        time_limit: Maximum number of steps per episode
        monitor: Whether to monitor episode statistics
        **kwargs: Additional arguments for environment creation
    
    Returns:
        vec_env: The created vectorized environment
    """
    # Create environment creation function
    def make_single_env(idx):
        env = make_env(
            env_id,
            seed=seed + idx,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            normalize_obs=normalize_obs,
            clip_rewards=clip_rewards,
            time_limit=time_limit,
            monitor=monitor,
            **kwargs
        )
        return env
    
    # Create vectorized environment
    vec_env = gym.vector.AsyncVectorEnv([
        lambda idx=i: make_single_env(idx) for i in range(num_envs)
    ])
    
    return vec_env


def visualize_env(env: gym.Env, num_episodes: int = 1, max_steps: int = 1000) -> None:
    """Visualize episodes from an environment.
    
    Args:
        env: The environment to visualize
        num_episodes: Number of episodes to run
        max_steps: Maximum number of steps per episode
    """
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        
        while not done and steps < max_steps:
            # Render environment
            env.render()
            
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Steps: {steps}")


def get_obs_stats(env: gym.Env, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """Get observation statistics from an environment.
    
    Args:
        env: The environment to get statistics from
        num_samples: Number of samples to collect
    
    Returns:
        mean: Mean of observations
        std: Standard deviation of observations
    """
    observations = []
    
    # Collect samples
    obs, _ = env.reset()
    observations.append(obs)
    
    for _ in range(num_samples - 1):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        
        if terminated or truncated:
            obs, _ = env.reset()
            observations.append(obs)
    
    # Convert to numpy array
    observations = np.array(observations)
    
    # Calculate statistics
    mean = np.mean(observations, axis=0)
    std = np.std(observations, axis=0)
    
    return mean, std


def get_action_stats(env: gym.Env, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """Get action statistics from an environment.
    
    Args:
        env: The environment to get statistics from
        num_samples: Number of samples to collect
    
    Returns:
        mean: Mean of actions
        std: Standard deviation of actions
    """
    # Check if action space is continuous
    if not isinstance(env.action_space, spaces.Box):
        raise ValueError("Action space must be continuous (Box)")
    
    # Generate random actions
    actions = []
    
    for _ in range(num_samples):
        action = env.action_space.sample()
        actions.append(action)
    
    # Convert to numpy array
    actions = np.array(actions)
    
    # Calculate statistics
    mean = np.mean(actions, axis=0)
    std = np.std(actions, axis=0)
    
    return mean, std


def evaluate_policy(
    env: gym.Env,
    policy: Callable[[np.ndarray], Union[int, np.ndarray]],
    num_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False
) -> Dict[str, float]:
    """Evaluate a policy in an environment.
    
    Args:
        env: The environment to evaluate in
        policy: Function that takes observations and returns actions
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic policy
        render: Whether to render the environment
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Initialize metrics
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            # Render if requested
            if render:
                env.render()
            
            # Select action
            action = policy(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
        
        # Store metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Steps: {steps}")
    
    # Calculate summary metrics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    return metrics


def record_video(
    env: gym.Env,
    policy: Callable[[np.ndarray], Union[int, np.ndarray]],
    video_path: str,
    num_episodes: int = 1,
    fps: int = 30,
    deterministic: bool = True
) -> None:
    """Record a video of a policy in an environment.
    
    Args:
        env: The environment to record in
        policy: Function that takes observations and returns actions
        video_path: Path to save the video
        num_episodes: Number of episodes to record
        fps: Frames per second
        deterministic: Whether to use deterministic policy
    """
    # Check if the environment supports recording
    if not hasattr(env, 'render_mode') or env.render_mode != 'rgb_array':
        raise ValueError("Environment must support 'rgb_array' render mode")
    
    try:
        import imageio
    except ImportError:
        raise ImportError("Package 'imageio' is required for recording videos")
    
    # Initialize video writer
    writer = imageio.get_writer(video_path, fps=fps)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        frames = []
        
        while not done:
            # Render frame
            frame = env.render()
            frames.append(frame)
            
            # Select action
            action = policy(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
        
        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}")
        
        # Write frames to video
        for frame in frames:
            writer.append_data(frame)
    
    # Close video writer
    writer.close()
    
    print(f"Video saved to {video_path}")


def plot_learning_curve(
    data: Dict[str, List[float]],
    x_key: str,
    y_keys: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    window: int = 10,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot learning curves from training data.
    
    Args:
        data: Dictionary with training data
        x_key: Key for x-axis data
        y_keys: Keys for y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        window: Window size for smoothing
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each y key
    for y_key in y_keys:
        x = data[x_key]
        y = data[y_key]
        
        # Apply smoothing if requested
        if window > 1:
            y_smooth = np.convolve(y, np.ones(window) / window, mode='valid')
            x_smooth = x[window-1:]
            
            # Plot both raw and smoothed data
            ax.plot(x, y, alpha=0.3, label=f"{y_key} (raw)")
            ax.plot(x_smooth, y_smooth, label=f"{y_key} (smoothed)")
        else:
            ax.plot(x, y, label=y_key)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig


def make_atari_env(env_id, seed=0, frame_stack=4, clip_rewards=True):
    """创建预处理过的Atari环境"""
    env = create_gym_env(env_id, seed=seed)

    # Atari预处理
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = FrameSkip(env, skip=4)
    env = FrameStack(env, n_frames=frame_stack)

    if clip_rewards:
        from environments.env_wrappers import ClipReward
        env = ClipReward(env)

    return env


def make_mujoco_env(env_id, seed=0, normalize_obs=True):
    """创建MuJoCo物理环境"""
    env = create_gym_env(env_id, seed=seed)

    if normalize_obs:
        env = NormalizeObservation(env)

    return env
