"""
Policy network implementations for reinforcement learning.

This module provides various policy network architectures including:
- Actor networks (discrete and continuous)
- Critic networks
- Actor-critic networks
- Gaussian policy networks for continuous control
"""

from typing import List, Tuple, Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import Normal, Categorical


class ActorNetwork(nn.Module):
    """Actor network for policy gradient methods.
    
    This implements a policy network that outputs action probabilities
    for discrete action spaces or means/stds for continuous action spaces.
    """
    
    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]], 
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: nn.Module = nn.Tanh(),
        continuous_actions: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """Initialize the actor network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use
            continuous_actions: Whether actions are continuous
            log_std_min: Minimum log standard deviation for continuous actions
            log_std_max: Maximum log standard deviation for continuous actions
        """
        super().__init__()
        
        if isinstance(state_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(state_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = state_dim
            self.input_preprocess = lambda x: x
            
        self.continuous_actions = continuous_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        layers = []
        in_dim = self.input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(activation)
            in_dim = dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layer
        if continuous_actions:
            # For continuous actions, output mean
            self.mean = nn.Linear(in_dim, action_dim)
            # And log std
            self.log_std = nn.Linear(in_dim, action_dim)
        else:
            # For discrete actions, output action probabilities
            self.action_head = nn.Linear(in_dim, action_dim)
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            If discrete actions:
                action_probs: Action probability distribution
            If continuous actions:
                mean: Mean of the action distribution
                std: Standard deviation of the action distribution
        """
        state = self.input_preprocess(state)
        features = self.shared_layers(state)
        
        if self.continuous_actions:
            mean = self.mean(features)
            log_std = self.log_std(features)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = log_std.exp()
            return mean, std
        else:
            action_logits = self.action_head(features)
            return F.softmax(action_logits, dim=-1)
            
    def sample(self, state: torch.Tensor) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Sample an action from the policy.
        
        Args:
            state: State tensor
            
        Returns:
            If discrete actions:
                action: Sampled action
                log_prob: Log probability of the action
            If continuous actions:
                action: Sampled action
                log_prob: Log probability of the action
                mean: Mean of the action distribution
        """
        if self.continuous_actions:
            mean, std = self.forward(state)
            dist = Normal(mean, std)
            action = dist.rsample()  # rsample() uses the reparameterization trick
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, mean
        else:
            action_probs = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob


class CriticNetwork(nn.Module):
    """Critic network for value-based and actor-critic methods.
    
    This implements a critic that estimates state values V(s).
    """
    
    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]],
        hidden_dims: List[int] = [64, 64],
        activation: nn.Module = nn.Tanh()
    ):
        """Initialize the critic network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use
        """
        super().__init__()
        
        if isinstance(state_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(state_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = state_dim
            self.input_preprocess = lambda x: x
        
        # Build the network
        layers = []
        in_dim = self.input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(activation)
            in_dim = dim
            
        # Value output
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
                
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            value: Estimated value of the state
        """
        state = self.input_preprocess(state)
        return self.network(state)


class QNetwork(nn.Module):
    """Q-Network for state-action value estimation.
    
    This implements a critic that estimates state-action values Q(s, a).
    """
    
    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU()
    ):
        """Initialize the Q-network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use
        """
        super().__init__()
        
        if isinstance(state_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(state_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = state_dim
            self.input_preprocess = lambda x: x
        
        # Build the network
        layers = []
        in_dim = self.input_dim + action_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(activation)
            in_dim = dim
            
        # Q-value output
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
                
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            q_value: Estimated Q-value of the state-action pair
        """
        state = self.input_preprocess(state)
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ActorCriticNetwork(nn.Module):
    """Combined actor-critic network with shared features.
    
    This implements a network that outputs both action probabilities
    and state values, sharing feature extraction layers.
    """
    
    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: nn.Module = nn.Tanh(),
        continuous_actions: bool = False,
        log_std_init: float = 0.0
    ):
        """Initialize the actor-critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use
            continuous_actions: Whether actions are continuous
            log_std_init: Initial log standard deviation for continuous actions
        """
        super().__init__()
        
        if isinstance(state_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(state_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = state_dim
            self.input_preprocess = lambda x: x
            
        self.continuous_actions = continuous_actions
        
        # Shared feature layers
        layers = []
        in_dim = self.input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(activation)
            in_dim = dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        if continuous_actions:
            self.actor_mean = nn.Linear(in_dim, action_dim)
            # Initialize log_std as a directly learnable parameter for stability
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        else:
            self.actor = nn.Linear(in_dim, action_dim)
            
        # Critic head (value function)
        self.critic = nn.Linear(in_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
                
        # Special initialization for the value output
        if hasattr(self, 'critic'):
            nn.init.orthogonal_(self.critic.weight, gain=1.0)
            nn.init.zeros_(self.critic.bias)
            
        # Special initialization for the policy output
        if hasattr(self, 'actor'):
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.zeros_(self.actor.bias)
        elif hasattr(self, 'actor_mean'):
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.zeros_(self.actor_mean.bias)
                
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            If discrete actions:
                action_probs: Action probability distribution
                value: Estimated value of the state
            If continuous actions:
                (mean, std): Parameters of the action distribution
                value: Estimated value of the state
        """
        state = self.input_preprocess(state)
        features = self.shared_layers(state)
        
        if self.continuous_actions:
            mean = self.actor_mean(features)
            std = torch.exp(self.log_std.expand_as(mean))
            value = self.critic(features)
            return (mean, std), value
        else:
            action_logits = self.actor(features)
            action_probs = F.softmax(action_logits, dim=-1)
            value = self.critic(features)
            return action_probs, value


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous action spaces.
    
    This implements a policy that outputs a Gaussian distribution
    over actions, with state-dependent mean and standard deviation.
    """
    
    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU(),
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """Initialize the Gaussian policy.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use
            action_scale: Scaling factor for actions
            action_bias: Bias for actions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        if isinstance(state_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(state_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = state_dim
            self.input_preprocess = lambda x: x
        
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        layers = []
        in_dim = self.input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(activation)
            in_dim = dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Mean and log std heads
        self.mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Linear(in_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
                
        # Special initialization for the policy output
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.zeros_(self.log_std.bias)
                
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            mean: Mean of the action distribution
            log_std: Log standard deviation of the action distribution
        """
        state = self.input_preprocess(state)
        features = self.shared_layers(state)
        
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
        
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.
        
        Args:
            state: State tensor
            
        Returns:
            action: Sampled action (scaled)
            log_prob: Log probability of the action
            mean: Mean of the action distribution (scaled)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from the Gaussian distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Use reparameterization trick
        y_t = torch.tanh(x_t)   # Squash to [-1, 1]
        
        # Scale and shift to the desired action range
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability, accounting for the tanh squashing
        # This formula comes from the SAC paper
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh squashing correction
        # log pi(a|s) = log pi(a'|s) - sum(log(1 - tanh(a')^2))
        # where a' = atanh(a)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
