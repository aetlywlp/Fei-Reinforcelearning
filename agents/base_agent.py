"""
Base agent interface for reinforcement learning algorithms.

This module defines the abstract base class that all RL agents should inherit from.
It specifies the minimum interface that an agent must implement.
"""

import abc
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import torch
import gymnasium as gym


class BaseAgent(abc.ABC):
    """Abstract base class for all reinforcement learning agents.
    
    This class defines the interface that all agents should implement.
    It provides common functionality and enforces a consistent API.
    """
    
    def __init__(self, state_dim: Union[int, Tuple[int, ...]], action_dim: int, device: str = "auto"):
        """Initialize the agent.
        
        Args:
            state_dim: Dimension of the state space (int for discrete, tuple for continuous)
            action_dim: Dimension of the action space
            device: Device to use for tensor operations ('cpu', 'cuda', or 'auto')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set device for tensor operations
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Training statistics
        self.training_steps = 0
        self.episodes = 0
        self.best_reward = -float('inf')
        
    @abc.abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """Select an action given a state.
        
        Args:
            state: The current state
            training: Whether the agent is in training mode (affects exploration)
            
        Returns:
            action: The selected action
        """
        pass
    
    @abc.abstractmethod
    def update(self) -> Dict[str, float]:
        """Update the agent's parameters based on experience.
        
        Returns:
            info: Dictionary containing training metrics
        """
        pass
    
    @abc.abstractmethod
    def store_transition(self, state: np.ndarray, action: Union[int, np.ndarray], 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store a transition in the agent's memory.
        
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state
            done: Whether the episode has terminated
        """
        pass
    
    @abc.abstractmethod
    def save(self, path: str):
        """Save the agent's parameters to a file.
        
        Args:
            path: Path to save the parameters
        """
        pass
    
    @abc.abstractmethod
    def load(self, path: str):
        """Load the agent's parameters from a file.
        
        Args:
            path: Path to load the parameters from
        """
        pass
    
    def train_episode(self, env: gym.Env, max_steps: int = 1000) -> Dict[str, float]:
        """Train the agent for one episode.
        
        Args:
            env: The environment to train on
            max_steps: Maximum number of steps per episode
            
        Returns:
            info: Dictionary containing episode metrics
        """
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Select and perform an action
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store the transition in memory
            self.store_transition(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Update the agent
            update_info = self.update()
            
            if done:
                break
        
        # Update statistics
        self.episodes += 1
        self.training_steps += episode_steps
        self.best_reward = max(self.best_reward, episode_reward)
        
        # Return episode information
        info = {
            'episode': self.episodes,
            'reward': episode_reward,
            'steps': episode_steps,
            'best_reward': self.best_reward
        }
        if update_info:
            info.update(update_info)
            
        return info
    
    def evaluate(self, env: gym.Env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent's performance.
        
        Args:
            env: The environment to evaluate on
            num_episodes: Number of episodes to evaluate
            
        Returns:
            metrics: Dictionary containing evaluation metrics
        """
        rewards = []
        steps = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_steps': np.mean(steps)
        }
