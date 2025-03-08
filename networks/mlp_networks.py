"""
Multi-layer perceptron (MLP) network implementations.

This module provides various MLP architectures for reinforcement learning,
including standard MLPs and dueling network architectures.
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers.
    
    This class implements a standard MLP that maps states to action values.
    """
    
    def __init__(
        self, 
        input_dim: Union[int, Tuple[int, ...]], 
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = None
    ):
        """Initialize the MLP.
        
        Args:
            input_dim: Dimension of the input (state)
            output_dim: Dimension of the output (action values)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use for hidden layers
            output_activation: Activation function to use for output layer (None for no activation)
        """
        super().__init__()
        
        if isinstance(input_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(input_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = input_dim
            self.input_preprocess = lambda x: x
        
        # Build the network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(activation)
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch of states)
            
        Returns:
            output: Output tensor (action values)
        """
        x = self.input_preprocess(x)
        return self.network(x)


class DuelingMLP(nn.Module):
    """Dueling network architecture for DQN.
    
    This class implements a dueling architecture that separates
    state value and advantage functions.
    """
    
    def __init__(
        self, 
        input_dim: Union[int, Tuple[int, ...]], 
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU()
    ):
        """Initialize the dueling MLP.
        
        Args:
            input_dim: Dimension of the input (state)
            output_dim: Dimension of the output (action values)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use for hidden layers
        """
        super().__init__()
        
        if isinstance(input_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(input_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = input_dim
            self.input_preprocess = lambda x: x
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            activation,
            nn.Linear(hidden_dims[1] // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            activation,
            nn.Linear(hidden_dims[1] // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch of states)
            
        Returns:
            q_values: Action value tensor
        """
        x = self.input_preprocess(x)
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration.
    
    This implements a linear layer with parametric noise, as described in
    "Noisy Networks for Exploration" (Fortunato et al., 2018).
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialize the noisy linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial value for the noise standard deviation
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        """Reset the parameters (weights and biases)."""
        mu_range = 1 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
        
    def reset_noise(self):
        """Reset the noise parameters."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise.
        
        Args:
            size: Size of the tensor
            
        Returns:
            noise: Scaled noise tensor
        """
        noise = torch.randn(size)
        return noise.sign().mul(noise.abs().sqrt())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with added noise.
        
        Args:
            x: Input tensor
            
        Returns:
            output: Output tensor with noise added
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)


class NoisyMLP(nn.Module):
    """Multi-layer perceptron with noisy linear layers for exploration.
    
    This implements a network with noisy linear layers for parameterized
    exploration, as described in "Noisy Networks for Exploration".
    """
    
    def __init__(
        self, 
        input_dim: Union[int, Tuple[int, ...]], 
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU(),
        std_init: float = 0.5
    ):
        """Initialize the noisy MLP.
        
        Args:
            input_dim: Dimension of the input (state)
            output_dim: Dimension of the output (action values)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use for hidden layers
            std_init: Initial value for the noise standard deviation
        """
        super().__init__()
        
        if isinstance(input_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(input_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = input_dim
            self.input_preprocess = lambda x: x
        
        # Build the network architecture
        layers = []
        
        # Input layer (normal linear)
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)
        
        # Hidden layers (noisy)
        for i in range(len(hidden_dims) - 1):
            layers.append(NoisyLinear(hidden_dims[i], hidden_dims[i+1], std_init))
            layers.append(activation)
        
        # Output layer (noisy)
        layers.append(NoisyLinear(hidden_dims[-1], output_dim, std_init))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch of states)
            
        Returns:
            output: Output tensor (action values)
        """
        x = self.input_preprocess(x)
        return self.network(x)
        
    def reset_noise(self):
        """Reset the noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class NoisyDuelingMLP(nn.Module):
    """Dueling architecture with noisy linear layers.
    
    This combines the dueling architecture with noisy networks
    for parameterized exploration.
    """
    
    def __init__(
        self, 
        input_dim: Union[int, Tuple[int, ...]], 
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU(),
        std_init: float = 0.5
    ):
        """Initialize the noisy dueling MLP.
        
        Args:
            input_dim: Dimension of the input (state)
            output_dim: Dimension of the output (action values)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use for hidden layers
            std_init: Initial value for the noise standard deviation
        """
        super().__init__()
        
        if isinstance(input_dim, tuple):
            # Flatten input dimension if it's multi-dimensional
            self.input_dim = np.prod(input_dim)
            self.input_preprocess = lambda x: x.view(x.size(0), -1)
        else:
            self.input_dim = input_dim
            self.input_preprocess = lambda x: x
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            activation,
            NoisyLinear(hidden_dims[0], hidden_dims[1], std_init),
            activation
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dims[1], hidden_dims[1] // 2, std_init),
            activation,
            NoisyLinear(hidden_dims[1] // 2, 1, std_init)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dims[1], hidden_dims[1] // 2, std_init),
            activation,
            NoisyLinear(hidden_dims[1] // 2, output_dim, std_init)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch of states)
            
        Returns:
            q_values: Action value tensor
        """
        x = self.input_preprocess(x)
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling formula
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
        
    def reset_noise(self):
        """Reset the noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
