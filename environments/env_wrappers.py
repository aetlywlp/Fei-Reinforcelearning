"""
Environment wrappers for reinforcement learning.

This module provides various wrappers for Gymnasium environments that 
implement common preprocessing and normalization techniques.
"""

from typing import Dict, Any, Tuple, List, Optional, Union, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import cv2


class FrameStack(gym.Wrapper):
    """Stack frames for multiple timesteps.
    
    This wrapper stacks multiple frames together to provide a history
    of observations, which can help with learning temporal patterns.
    """
    
    def __init__(self, env: gym.Env, n_frames: int = 4):
        """Initialize the frame stacking wrapper.
        
        Args:
            env: The environment to wrap
            n_frames: Number of frames to stack
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = None
        
        # Modify observation space to account for stacked frames
        shape = env.observation_space.shape
        
        # Handle 1D and image observations differently
        if len(shape) == 1:
            # Vector observations
            self.observation_space = spaces.Box(
                low=np.repeat(env.observation_space.low, n_frames, axis=0),
                high=np.repeat(env.observation_space.high, n_frames, axis=0),
                dtype=env.observation_space.dtype
            )
        else:
            # Image observations - stack along first dimension
            self.observation_space = spaces.Box(
                low=env.observation_space.low[0],
                high=env.observation_space.high[0],
                shape=(n_frames, *shape),
                dtype=env.observation_space.dtype
            )
            
    def reset(self, **kwargs):
        """Reset the environment and initialize the frame stack.
        
        Returns:
            stacked_frames: Stacked initial frames
            info: Environment info
        """
        observation, info = self.env.reset(**kwargs)
        
        # Initialize frames buffer with copies of the initial observation
        if len(self.env.observation_space.shape) == 1:
            # Vector observations
            self.frames = np.tile(observation, (self.n_frames, 1))
        else:
            # Image observations
            self.frames = np.tile(observation[np.newaxis, ...], (self.n_frames, 1, 1, 1))
            
        return self._get_observation(), info
    
    def step(self, action):
        """Step the environment and update the frame stack.
        
        Args:
            action: Action to take
            
        Returns:
            stacked_frames: Stacked frames after taking the action
            reward: Reward from the environment
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated
            info: Environment info
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update the frame stack
        if len(self.env.observation_space.shape) == 1:
            # Vector observations - shift and replace the oldest frame
            self.frames = np.roll(self.frames, shift=-1, axis=0)
            self.frames[-1] = observation
        else:
            # Image observations - shift and replace the oldest frame
            self.frames = np.roll(self.frames, shift=-1, axis=0)
            self.frames[-1] = observation
            
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get the stacked frames as the current observation.
        
        Returns:
            stacked_frames: Current stacked frames
        """
        if len(self.env.observation_space.shape) == 1:
            # Vector observations - return flattened array
            return self.frames.flatten()
        else:
            # Image observations - return stacked frames
            return self.frames


class FrameSkip(gym.Wrapper):
    """Skip frames and repeat actions.
    
    This wrapper repeats the same action for multiple frames and
    returns only the last frame, which can speed up training.
    """
    
    def __init__(self, env: gym.Env, skip: int = 4):
        """Initialize the frame skipping wrapper.
        
        Args:
            env: The environment to wrap
            skip: Number of frames to skip
        """
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        """Step the environment multiple times with the same action.
        
        Args:
            action: Action to repeat
            
        Returns:
            observation: Observation after skipping frames
            total_reward: Sum of rewards across skipped frames
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated
            info: Environment info
        """
        total_reward = 0.0
        terminated = truncated = False
        
        # Repeat the action for skip frames
        for i in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
                
        return observation, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert observations to grayscale.
    
    This wrapper converts RGB images to grayscale to reduce
    input dimensionality for neural networks.
    """
    
    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """Initialize the grayscale wrapper.
        
        Args:
            env: The environment to wrap
            keep_dim: Whether to keep the channel dimension (1 channel)
        """
        super().__init__(env)
        self.keep_dim = keep_dim
        
        # Modify observation space to account for grayscale conversion
        assert len(env.observation_space.shape) == 3, "Observation space must be an image"
        
        obs_shape = env.observation_space.shape
        
        if self.keep_dim:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(1, obs_shape[1], obs_shape[2]),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(obs_shape[1], obs_shape[2]),
                dtype=np.uint8
            )
            
    def observation(self, observation):
        """Convert the observation to grayscale.
        
        Args:
            observation: RGB observation
            
        Returns:
            grayscale: Grayscale observation
        """
        grayscale = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        
        if self.keep_dim:
            grayscale = np.expand_dims(grayscale, axis=0)
            
        return grayscale


class ResizeObservation(gym.ObservationWrapper):
    """Resize observations to a specified size.
    
    This wrapper resizes image observations to a specified size,
    which can reduce input dimensionality for neural networks.
    """
    
    def __init__(self, env: gym.Env, shape: Tuple[int, int]):
        """Initialize the resize wrapper.
        
        Args:
            env: The environment to wrap
            shape: Target shape (height, width)
        """
        super().__init__(env)
        self.shape = shape
        
        # Modify observation space to account for resizing
        assert len(env.observation_space.shape) in [1, 3], "Observation space must be an image"
        
        obs_shape = env.observation_space.shape
        
        if len(obs_shape) == 3:
            # RGB image
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(obs_shape[0], *shape),
                dtype=np.uint8
            )
        else:
            # Grayscale image
            self.observation_space = spaces.Box(
                low=0, high=255, shape=shape,
                dtype=np.uint8
            )
            
    def observation(self, observation):
        """Resize the observation.
        
        Args:
            observation: Original observation
            
        Returns:
            resized: Resized observation
        """
        if len(observation.shape) == 3 and observation.shape[0] == 1:
            # Grayscale image with channel dimension
            observation = observation.squeeze(0)
            resized = cv2.resize(observation, (self.shape[1], self.shape[0]))
            return np.expand_dims(resized, axis=0)
        elif len(observation.shape) == 3 and observation.shape[0] == 3:
            # RGB image
            resized = np.zeros((3, self.shape[0], self.shape[1]), dtype=np.uint8)
            for i in range(3):
                resized[i] = cv2.resize(observation[i], (self.shape[1], self.shape[0]))
            return resized
        else:
            # Grayscale image without channel dimension
            return cv2.resize(observation, (self.shape[1], self.shape[0]))


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to a specified range.
    
    This wrapper normalizes observations to a specified range,
    which can help with neural network training stability.
    """
    
    def __init__(self, env: gym.Env, low: float = 0.0, high: float = 1.0):
        """Initialize the normalization wrapper.
        
        Args:
            env: The environment to wrap
            low: Lower bound of the target range
            high: Upper bound of the target range
        """
        super().__init__(env)
        self.low = low
        self.high = high
        
        # Get the range of the observation space
        assert isinstance(env.observation_space, spaces.Box), "Observation space must be continuous"
        
        self.orig_low = env.observation_space.low
        self.orig_high = env.observation_space.high
        
        # Modify observation space to account for normalization
        self.observation_space = spaces.Box(
            low=low, high=high, shape=env.observation_space.shape,
            dtype=np.float32
        )
        
    def observation(self, observation):
        """Normalize the observation.
        
        Args:
            observation: Original observation
            
        Returns:
            normalized: Normalized observation
        """
        # Handle cases where original range is zero
        range_mask = (self.orig_high - self.orig_low) > 0
        
        normalized = np.zeros_like(observation, dtype=np.float32)
        normalized[range_mask] = (
            (observation[range_mask] - self.orig_low[range_mask]) / 
            (self.orig_high[range_mask] - self.orig_low[range_mask])
        )
        
        # Scale to target range
        normalized = normalized * (self.high - self.low) + self.low
        
        return normalized


class ClipReward(gym.RewardWrapper):
    """Clip rewards to a specified range.
    
    This wrapper clips rewards to a specified range,
    which can help with training stability.
    """
    
    def __init__(self, env: gym.Env, min_reward: float = -1.0, max_reward: float = 1.0):
        """Initialize the reward clipping wrapper.
        
        Args:
            env: The environment to wrap
            min_reward: Minimum reward value
            max_reward: Maximum reward value
        """
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        
    def reward(self, reward):
        """Clip the reward.
        
        Args:
            reward: Original reward
            
        Returns:
            clipped: Clipped reward
        """
        return np.clip(reward, self.min_reward, self.max_reward)


class NormalizeReward(gym.RewardWrapper):
    """Normalize rewards using running statistics.
    
    This wrapper normalizes rewards using running mean and standard deviation,
    which can help with training stability.
    """
    
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        """Initialize the reward normalization wrapper.
        
        Args:
            env: The environment to wrap
            gamma: Discount factor for running statistics
            epsilon: Small constant for numerical stability
        """
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize running statistics
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
    def reward(self, reward):
        """Normalize the reward using running statistics.
        
        Args:
            reward: Original reward
            
        Returns:
            normalized: Normalized reward
        """
        # Update running statistics
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var = (self.reward_var * (self.reward_count - 1) + delta * delta2) / self.reward_count
        
        # Normalize reward
        normalized = reward / (np.sqrt(self.reward_var) + self.epsilon)
        
        return normalized
    
    def reset(self, **kwargs):
        """Reset the environment and the running statistics.
        
        Returns:
            observation: Initial observation
            info: Environment info
        """
        # Reset running statistics
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
        return self.env.reset(**kwargs)


class TimeLimit(gym.Wrapper):
    """Limit the maximum number of steps per episode.
    
    This wrapper limits the maximum number of steps per episode,
    which can prevent episodes from running indefinitely.
    """
    
    def __init__(self, env: gym.Env, max_steps: int = 1000):
        """Initialize the time limit wrapper.
        
        Args:
            env: The environment to wrap
            max_steps: Maximum number of steps per episode
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        
    def reset(self, **kwargs):
        """Reset the environment and the step counter.
        
        Returns:
            observation: Initial observation
            info: Environment info
        """
        self.current_step = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step the environment and check if the time limit has been reached.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Current observation
            reward: Reward from the environment
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated
            info: Environment info
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            
        return observation, reward, terminated, truncated, info


class ActionNormalization(gym.ActionWrapper):
    """Normalize continuous actions to a specified range.
    
    This wrapper normalizes continuous actions to a specified range,
    which can help with training stability.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        target_low: float = -1.0, 
        target_high: float = 1.0
    ):
        """Initialize the action normalization wrapper.
        
        Args:
            env: The environment to wrap
            target_low: Lower bound of the target range
            target_high: Upper bound of the target range
        """
        super().__init__(env)
        
        # Check if action space is continuous
        assert isinstance(env.action_space, spaces.Box), "Action space must be continuous"
        
        self.target_low = target_low
        self.target_high = target_high
        
        # Get the range of the action space
        self.orig_low = env.action_space.low
        self.orig_high = env.action_space.high
        
        # Modify action space to account for normalization
        self.action_space = spaces.Box(
            low=target_low, high=target_high, shape=env.action_space.shape,
            dtype=env.action_space.dtype
        )
        
    def action(self, action):
        """Normalize the action from the target range to the original range.
        
        Args:
            action: Action in the target range
            
        Returns:
            normalized: Action in the original range
        """
        # Clip action to target range
        clipped = np.clip(action, self.target_low, self.target_high)
        
        # Normalize from target range to [0, 1]
        normalized = (clipped - self.target_low) / (self.target_high - self.target_low)
        
        # Scale to original range
        scaled = normalized * (self.orig_high - self.orig_low) + self.orig_low
        
        return scaled


class GoalEnvWrapper(gym.Wrapper):
    """Wrapper for goal-based environments.
    
    This wrapper provides a standardized interface for goal-based environments,
    where the reward is computed based on the achieved goal and the desired goal.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        reward_fn: Callable[[np.ndarray, np.ndarray, Dict], float],
        distance_threshold: float = 0.05
    ):
        """Initialize the goal environment wrapper.
        
        Args:
            env: The environment to wrap
            reward_fn: Function that computes the reward given (achieved_goal, desired_goal, info)
            distance_threshold: Threshold for considering a goal achieved
        """
        super().__init__(env)
        self.reward_fn = reward_fn
        self.distance_threshold = distance_threshold
        
        # Check if the environment has the required observation components
        assert isinstance(env.observation_space, spaces.Dict), "Observation space must be a Dict"
        assert 'observation' in env.observation_space.spaces, "Observation space must contain 'observation'"
        assert 'desired_goal' in env.observation_space.spaces, "Observation space must contain 'desired_goal'"
        assert 'achieved_goal' in env.observation_space.spaces, "Observation space must contain 'achieved_goal'"
        
    def step(self, action):
        """Step the environment and compute the reward.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Current observation dict
            reward: Computed reward
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated
            info: Environment info
        """
        observation, _, terminated, truncated, info = self.env.step(action)
        
        # Compute reward based on achieved and desired goals
        reward = self.reward_fn(
            observation['achieved_goal'],
            observation['desired_goal'],
            info
        )
        
        # Check if the goal is achieved
        goal_achieved = np.linalg.norm(
            observation['achieved_goal'] - observation['desired_goal']
        ) < self.distance_threshold
        
        # Update terminated flag if goal is achieved
        terminated = terminated or goal_achieved
        
        return observation, reward, terminated, truncated, info


class MonitorEpisodeStats(gym.Wrapper):
    """Monitor and log episode statistics.
    
    This wrapper monitors and logs episode statistics such as
    return, length, and success rate.
    """
    
    def __init__(self, env: gym.Env, window_size: int = 100):
        """Initialize the episode monitoring wrapper.
        
        Args:
            env: The environment to wrap
            window_size: Number of episodes to use for running statistics
        """
        super().__init__(env)
        self.window_size = window_size
        
        # Initialize episode statistics
        self.episode_returns = np.zeros(window_size)
        self.episode_lengths = np.zeros(window_size)
        self.episode_successes = np.zeros(window_size)
        
        self.current_episode_idx = 0
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        
        # Total statistics
        self.total_episodes = 0
        self.total_steps = 0
        
    def reset(self, **kwargs):
        """Reset the environment and the episode statistics.
        
        Returns:
            observation: Initial observation
            info: Environment info with episode statistics
        """
        observation, info = self.env.reset(**kwargs)
        
        # Reset episode statistics
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        
        return observation, info
    
    def step(self, action):
        """Step the environment and update episode statistics.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Current observation
            reward: Reward from the environment
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated
            info: Environment info with episode statistics
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode statistics
        self.current_episode_return += reward
        self.current_episode_length += 1
        self.total_steps += 1
        
        # If episode ended, update running statistics
        if terminated or truncated:
            self.episode_returns[self.current_episode_idx] = self.current_episode_return
            self.episode_lengths[self.current_episode_idx] = self.current_episode_length
            
            # Check for success (if available in info)
            if 'is_success' in info:
                self.episode_successes[self.current_episode_idx] = info['is_success']
                
            # Update indices
            self.current_episode_idx = (self.current_episode_idx + 1) % self.window_size
            self.total_episodes += 1
            
            # Add statistics to info
            info['episode'] = {
                'return': self.current_episode_return,
                'length': self.current_episode_length,
                'mean_return': np.mean(self.episode_returns),
                'mean_length': np.mean(self.episode_lengths),
                'success_rate': np.mean(self.episode_successes),
                'total_episodes': self.total_episodes,
                'total_steps': self.total_steps
            }
            
        return observation, reward, terminated, truncated, info
