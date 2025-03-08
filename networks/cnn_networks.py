"""
Convolutional Neural Network (CNN) implementations.

This module provides CNN architectures for processing visual inputs
in reinforcement learning, including standard CNNs and dueling architectures.
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    """Convolutional neural network for visual inputs.
    
    This class implements a CNN architecture suitable for processing
    visual states, such as game screens or robot cameras.
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, ...],  # (channels, height, width)
        output_dim: int,
        feature_dim: int = 512,
        kernel_sizes: List[int] = [8, 4, 3],
        strides: List[int] = [4, 2, 1],
        channels: List[int] = [32, 64, 64]
    ):
        """Initialize the CNN.
        
        Args:
            input_shape: Shape of the input image (channels, height, width)
            output_dim: Dimension of the output (action values)
            feature_dim: Dimension of the feature representation
            kernel_sizes: List of kernel sizes for each convolutional layer
            strides: List of strides for each convolutional layer
            channels: List of output channels for each convolutional layer
        """
        super().__init__()
        
        assert len(kernel_sizes) == len(strides) == len(channels), \
            "kernel_sizes, strides, and channels must have the same length"
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        
        # Convolutional layers
        conv_layers = []
        in_channels = input_shape[0]
        
        for i in range(len(channels)):
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i]
            ))
            conv_layers.append(nn.ReLU())
            in_channels = channels[i]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate output shape of convolutional layers
        conv_out_shape = self._get_conv_output_shape(input_shape)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_shape, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _get_conv_output_shape(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate the output shape of the convolutional layers.
        
        Args:
            input_shape: Input shape (channels, height, width)
            
        Returns:
            flat_size: Flattened size of the convolutional output
        """
        # Create a dummy input to pass through the conv layers
        dummy_input = torch.zeros(1, *input_shape)
        conv_out = self.conv_layers(dummy_input)
        return int(np.prod(conv_out.shape[1:]))
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            output: Output tensor (action values)
        """
        # Ensure input has the right shape
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Make sure input is in the right format (N, C, H, W)
        if x.shape[1:] != self.input_shape and x.shape[2:] == self.input_shape[1:]:
            # Input is likely (N, H, W, C), convert to (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            
        # Normalize input to [0, 1] if it's not already
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.conv_layers(x)
        flat_features = features.view(features.size(0), -1)
        output = self.fc_layers(flat_features)
        
        return output


class DuelingCNN(nn.Module):
    """Dueling CNN architecture for DQN.
    
    This implements a dueling architecture with CNN feature extraction.
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, ...],  # (channels, height, width)
        output_dim: int,
        feature_dim: int = 512,
        kernel_sizes: List[int] = [8, 4, 3],
        strides: List[int] = [4, 2, 1],
        channels: List[int] = [32, 64, 64]
    ):
        """Initialize the dueling CNN.
        
        Args:
            input_shape: Shape of the input image (channels, height, width)
            output_dim: Dimension of the output (action values)
            feature_dim: Dimension of the feature representation
            kernel_sizes: List of kernel sizes for each convolutional layer
            strides: List of strides for each convolutional layer
            channels: List of output channels for each convolutional layer
        """
        super().__init__()
        
        assert len(kernel_sizes) == len(strides) == len(channels), \
            "kernel_sizes, strides, and channels must have the same length"
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        
        # Convolutional feature extractor
        conv_layers = []
        in_channels = input_shape[0]
        
        for i in range(len(channels)):
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i]
            ))
            conv_layers.append(nn.ReLU())
            in_channels = channels[i]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate output shape of convolutional layers
        conv_out_shape = self._get_conv_output_shape(input_shape)
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(conv_out_shape, feature_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _get_conv_output_shape(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate the output shape of the convolutional layers.
        
        Args:
            input_shape: Input shape (channels, height, width)
            
        Returns:
            flat_size: Flattened size of the convolutional output
        """
        # Create a dummy input to pass through the conv layers
        dummy_input = torch.zeros(1, *input_shape)
        conv_out = self.conv_layers(dummy_input)
        return int(np.prod(conv_out.shape[1:]))
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            q_values: Action value tensor
        """
        # Ensure input has the right shape
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Make sure input is in the right format (N, C, H, W)
        if x.shape[1:] != self.input_shape and x.shape[2:] == self.input_shape[1:]:
            # Input is likely (N, H, W, C), convert to (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            
        # Normalize input to [0, 1] if it's not already
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.conv_layers(x)
        flat_features = features.view(features.size(0), -1)
        
        features = self.feature_layer(flat_features)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling formula
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class NatureCNN(nn.Module):
    """CNN architecture from the DQN Nature paper.
    
    This implements the CNN architecture described in the paper:
    "Human-level control through deep reinforcement learning" (Mnih et al., 2015).
    """
    
    def __init__(
        self, 
        input_shape: Tuple[int, ...],  # (channels, height, width)
        output_dim: int
    ):
        """Initialize the Nature CNN.
        
        Args:
            input_shape: Shape of the input image (channels, height, width)
            output_dim: Dimension of the output (action values)
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # Convolutional layers exactly as described in the Nature paper
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output shape of convolutional layers
        conv_out_shape = self._get_conv_output_shape(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_shape, 512)
        self.fc2 = nn.Linear(512, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _get_conv_output_shape(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate the output shape of the convolutional layers.
        
        Args:
            input_shape: Input shape (channels, height, width)
            
        Returns:
            flat_size: Flattened size of the convolutional output
        """
        # Create a dummy input to pass through the conv layers
        dummy_input = torch.zeros(1, *input_shape)
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.shape[1:]))
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            output: Output tensor (action values)
        """
        # Ensure input has the right shape
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Make sure input is in the right format (N, C, H, W)
        if x.shape[1:] != self.input_shape and x.shape[2:] == self.input_shape[1:]:
            # Input is likely (N, H, W, C), convert to (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            
        # Normalize input to [0, 1] if it's not already
        if x.max() > 1.0:
            x = x / 255.0
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class R3MCNN(nn.Module):
    """CNN architecture for R3M visual features.
    
    This implements a network that uses pretrained visual features
    from the R3M model, which is useful for robotics applications.
    """
    
    def __init__(
        self,
        r3m_feature_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [512, 256]
    ):
        """Initialize the R3M CNN.
        
        Args:
            r3m_feature_dim: Dimension of the R3M feature vector
            output_dim: Dimension of the output (action values)
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()
        
        # MLP layers for processing R3M features
        layers = []
        in_dim = r3m_feature_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
                
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            features: Input tensor of R3M features (batch_size, r3m_feature_dim)
            
        Returns:
            output: Output tensor (action values)
        """
        return self.network(features)
