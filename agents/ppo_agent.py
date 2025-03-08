"""
Implementation of Proximal Policy Optimization (PPO) agent.

This module includes implementation of:
- PPO with clipped objective function
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy regularization
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gymnasium as gym

from agents.base_agent import BaseAgent
from networks.policy_networks import ActorCriticNetwork, ActorNetwork, CriticNetwork


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent.
    
    This class implements the PPO algorithm described in:
    "Proximal Policy Optimization Algorithms" (Schulman et al., 2017).
    """
    
    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]], 
        action_dim: int,
        config: Dict[str, Any],
        continuous_actions: bool = False,
        device: str = "auto"
    ):
        """Initialize the PPO agent.
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            config: Configuration dictionary with hyperparameters
            continuous_actions: Whether the action space is continuous
            device: Device to run the agent on ('cpu', 'cuda', or 'auto')
        """
        super().__init__(state_dim, action_dim, device)
        
        # Extract configuration
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.lr = config.get('learning_rate', 0.0003)
        
        # Learning parameters
        self.epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.continuous_actions = continuous_actions
        self.clip_value = config.get('clip_value', True)  # Whether to clip value function updates
        
        # Network architecture
        network_type = config.get('network_type', 'shared')
        hidden_dims = config.get('hidden_dims', [64, 64])
        
        # Initialize networks
        if network_type == 'shared':
            # Shared network for actor and critic
            self.network = ActorCriticNetwork(
                state_dim, 
                action_dim, 
                hidden_dims=hidden_dims,
                continuous_actions=continuous_actions
            ).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        else:
            # Separate networks for actor and critic
            self.actor = ActorNetwork(
                state_dim, 
                action_dim, 
                hidden_dims=hidden_dims,
                continuous_actions=continuous_actions
            ).to(self.device)
            self.critic = CriticNetwork(
                state_dim, 
                hidden_dims=hidden_dims
            ).to(self.device)
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=self.lr
            )
            
        # Initialize memory buffers for collecting experience
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # For continuous action spaces
        if continuous_actions:
            self.action_scale = config.get('action_scale', 1.0)
            self.action_bias = config.get('action_bias', 0.0)
            
    def select_action(self, state: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """Select an action given the current state.
        
        Args:
            state: Current state
            training: Whether to store additional info for training
            
        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if hasattr(self, 'network'):
                action_probs, value = self.network(state_tensor)
            else:
                action_probs = self.actor(state_tensor)
                value = self.critic(state_tensor)
                
            if self.continuous_actions:
                # For continuous actions, action_probs is (mean, std)
                mu, sigma = action_probs
                dist = Normal(mu, sigma)
                
                if training:
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                else:
                    # During evaluation just use the mean
                    action = mu
                    log_prob = dist.log_prob(mu).sum(dim=-1)
                
                # Scale the action to the action space
                action = self.action_scale * action + self.action_bias
                action = action.cpu().numpy().flatten()
            else:
                # For discrete actions, action_probs is a probability distribution
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.cpu().numpy().item()
                
        # Store info for training if needed
        if training:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob.cpu().numpy().item())
            self.values.append(value.cpu().numpy().item())
            
        return action
        
    def store_transition(self, state: np.ndarray, action: Union[int, np.ndarray], 
                       reward: float, next_state: np.ndarray, done: bool):
        """Store a transition for training.
        
        Args:
            state: Current state (already stored in select_action)
            action: Action taken (already stored in select_action)
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.rewards.append(reward)
        self.dones.append(done)
        
    def update(self) -> Dict[str, float]:
        """Update the policy and value networks.
        
        Returns:
            info: Dictionary with training metrics
        """
        # If we don't have enough experience, skip the update
        if len(self.states) < self.batch_size:
            return {}
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        
        if self.continuous_actions:
            # Scale actions back to network space
            unscaled_actions = (torch.FloatTensor(np.array(self.actions)) - self.action_bias) / self.action_scale
            actions = unscaled_actions.to(self.device)
        else:
            actions = torch.LongTensor(np.array(self.actions)).to(self.device)
            
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        
        # We need values for the next state for the last step
        with torch.no_grad():
            if hasattr(self, 'network'):
                _, next_value = self.network(torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device))
            else:
                next_value = self.critic(torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device))
            next_value = next_value.cpu().numpy().item()
            
        # Compute returns and advantages using GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
            
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Break experience into batches for multiple epochs
        num_samples = len(states)
        indices = np.arange(num_samples)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            # Process in batches
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current policy and value predictions
                if hasattr(self, 'network'):
                    action_probs, values = self.network(batch_states)
                else:
                    action_probs = self.actor(batch_states)
                    values = self.critic(batch_states)
                
                # Compute log probabilities and entropy
                if self.continuous_actions:
                    mu, sigma = action_probs
                    dist = Normal(mu, sigma)
                    current_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().mean()
                else:
                    dist = Categorical(action_probs)
                    if len(batch_actions.shape) > 1:
                        batch_actions = batch_actions.squeeze(-1)
                    current_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                
                # Compute policy loss with clipping
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss, optionally with clipping
                if self.clip_value:
                    # Compute and clip value function update
                    values_clipped = values.squeeze(-1)
                    old_values = torch.FloatTensor(values[batch_indices]).to(self.device)
                    values_clipped = old_values + torch.clamp(
                        values.squeeze(-1) - old_values,
                        -self.clip_ratio,
                        self.clip_ratio
                    )
                    value_loss1 = F.mse_loss(values.squeeze(-1), batch_returns)
                    value_loss2 = F.mse_loss(values_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    # Standard value loss
                    value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update the networks
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters() if hasattr(self, 'network') 
                    else list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Accumulate losses for reporting
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear memory buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Compute average losses
        n_updates = self.epochs * ((num_samples + self.batch_size - 1) // self.batch_size)
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }
            
    def save(self, path: str):
        """Save the agent's parameters.
        
        Args:
            path: Path to save the parameters
        """
        if hasattr(self, 'network'):
            torch.save({
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'training_steps': self.training_steps,
                'episodes': self.episodes,
                'best_reward': self.best_reward
            }, path)
        else:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'optimizer': self.optimizer.state_dict(),
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
        
        if hasattr(self, 'network'):
            self.network.load_state_dict(checkpoint['network'])
        else:
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']
        self.best_reward = checkpoint['best_reward']
