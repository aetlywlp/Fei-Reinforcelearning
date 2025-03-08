"""
Implementation of Soft Actor-Critic (SAC) agent.

This module includes implementation of:
- SAC with automatic entropy tuning
- Double Q-learning for critic
- Delayed policy updates
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym

from agents.base_agent import BaseAgent
from networks.policy_networks import GaussianPolicy, QNetwork
from memory.replay_buffer import ReplayBuffer


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent.
    
    This class implements the SAC algorithm described in:
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning 
    with a Stochastic Actor" (Haarnoja et al., 2018).
    """
    
    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]], 
        action_dim: int,
        config: Dict[str, Any],
        device: str = "auto"
    ):
        """Initialize the SAC agent.
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            config: Configuration dictionary with hyperparameters
            device: Device to run the agent on ('cpu', 'cuda', or 'auto')
        """
        super().__init__(state_dim, action_dim, device)
        
        # Extract configuration
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)  # For soft target updates
        self.lr_actor = config.get('lr_actor', 0.0003)
        self.lr_critic = config.get('lr_critic', 0.0003)
        self.batch_size = config.get('batch_size', 256)
        self.buffer_size = config.get('buffer_size', 1000000)
        self.learning_starts = config.get('learning_starts', 10000)
        self.policy_update_freq = config.get('policy_update_freq', 2)
        
        # Action space bounds
        self.action_scale = config.get('action_scale', 1.0)
        self.action_bias = config.get('action_bias', 0.0)
        
        # Network architecture
        hidden_dims = config.get('hidden_dims', [256, 256])
        
        # Initialize networks
        self.actor = GaussianPolicy(
            state_dim,
            action_dim,
            hidden_dims,
            action_scale=self.action_scale,
            action_bias=self.action_bias
        ).to(self.device)
        
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(self.buffer_size, state_dim, action_dim=action_dim)
        
        # Automatic entropy tuning
        self.target_entropy = -torch.prod(torch.tensor(action_dim, dtype=torch.float32)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_actor)
        self.auto_entropy_tuning = config.get('auto_entropy_tuning', True)
        
        if not self.auto_entropy_tuning:
            self.alpha = config.get('alpha', 0.2)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select an action given the current state.
        
        Args:
            state: Current state
            training: Whether to add noise to the action
            
        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                action, _, _ = self.actor.sample(state_tensor)
            else:
                _, _, action = self.actor.sample(state_tensor)
                
        return action.cpu().numpy().flatten()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                       reward: float, next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.add(state, action, reward, next_state, done)
        
    def update(self) -> Dict[str, float]:
        """Update the networks based on stored experiences.
        
        Returns:
            info: Dictionary with training metrics
        """
        # Only update if we have enough samples
        if len(self.memory) < self.learning_starts:
            return {}
            
        # Sample experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Get current alpha value
        if self.auto_entropy_tuning:
            alpha = self.log_alpha.exp().item()
        else:
            alpha = self.alpha
        
        # Update critics
        with torch.no_grad():
            # Sample next action and get its log probability
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Compute target Q-values
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # Compute critic losses
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        actor_loss = None
        alpha_loss = None
        
        # Delayed policy updates
        if self.training_steps % self.policy_update_freq == 0:
            # Update actor
            # Freeze critics to save computational efforts
            for param in self.critic1.parameters():
                param.requires_grad = False
            for param in self.critic2.parameters():
                param.requires_grad = False
                
            # Sample actions and compute log probabilities
            pi, log_pi, _ = self.actor.sample(states)
            
            # Compute actor loss
            q1_pi = self.critic1(states, pi)
            q2_pi = self.critic2(states, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            actor_loss = (alpha * log_pi - min_q_pi).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Unfreeze critics
            for param in self.critic1.parameters():
                param.requires_grad = True
            for param in self.critic2.parameters():
                param.requires_grad = True
                
            # Update alpha (temperature parameter)
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            
            # Soft update of target networks
            self._soft_update(self.critic1, self.critic1_target, self.tau)
            self._soft_update(self.critic2, self.critic2_target, self.tau)
        
        # Return training metrics
        info = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'q_value': q1.mean().item()
        }
        
        if actor_loss is not None:
            info['actor_loss'] = actor_loss.item()
            
        if alpha_loss is not None:
            info['alpha_loss'] = alpha_loss.item()
            info['alpha'] = alpha
            
        return info
    
    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        """Soft update target network parameters.
        
        Args:
            source: Source network
            target: Target network
            tau: Interpolation parameter
        """
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
    
    def save(self, path: str):
        """Save the agent's parameters.
        
        Args:
            path: Path to save the parameters
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'best_reward': self.best_reward
        }, path)
    
    def load(self, path: str):
        """Load the agent's parameters.
        
        Args:
            path: Path to load the parameters from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']
        self.best_reward = checkpoint['best_reward']
