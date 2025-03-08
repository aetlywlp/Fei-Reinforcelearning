"""
Implementation of Deep Q-Network (DQN) agent and its variants.

This module includes implementations of:
- Original DQN (Mnih et al., 2015)
- Double DQN (DDQN)
- Dueling DQN
- Noisy DQN
- Categorical DQN (C51)
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym


from agents.base_agent import BaseAgent
from networks.mlp_networks import MLP, DuelingMLP
from networks.cnn_networks import CNN, DuelingCNN
from memory.replay_buffer import ReplayBuffer
from memory.prioritized_replay import  ProportionalPrioritizedReplayBuffer


class DQNAgent(BaseAgent):
    """Deep Q-Network agent implementation.
    
    This class implements the DQN algorithm described in:
    "Human-level control through deep reinforcement learning" (Mnih et al., 2015).
    """
    
    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]], 
        action_dim: int,
        config: Dict[str, Any],
        device: str = "auto"
    ):
        """Initialize the DQN agent.
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            config: Configuration dictionary with hyperparameters
            device: Device to run the agent on ('cpu', 'cuda', or 'auto')
        """
        super().__init__(state_dim, action_dim, device)
        
        # Extract configuration
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('learning_rate', 0.0001)
        self.target_update_freq = config.get('target_update_frequency', 1000)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 100000)
        self.learning_starts = config.get('learning_starts', 1000)
        self.train_freq = config.get('train_frequency', 4)
        
        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.999)
        
        # Network architecture
        network_type = config.get('network_type', 'mlp')
        hidden_dims = config.get('hidden_dims', [256, 256])
        dueling = config.get('dueling', False)
        
        # Initialize networks
        if network_type == 'mlp':
            if dueling:
                self.q_network = DuelingMLP(state_dim, action_dim, hidden_dims).to(self.device)
                self.target_network = DuelingMLP(state_dim, action_dim, hidden_dims).to(self.device)
            else:
                self.q_network = MLP(state_dim, action_dim, hidden_dims).to(self.device)
                self.target_network = MLP(state_dim, action_dim, hidden_dims).to(self.device)
        elif network_type == 'cnn':
            if dueling:
                self.q_network = DuelingCNN(state_dim, action_dim).to(self.device)
                self.target_network = DuelingCNN(state_dim, action_dim).to(self.device)
            else:
                self.q_network = CNN(state_dim, action_dim).to(self.device)
                self.target_network = CNN(state_dim, action_dim).to(self.device)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Initialize replay buffer
        use_prioritized = config.get('prioritized_replay', False)
        if use_prioritized:
            alpha = config.get('prioritized_replay_alpha', 0.6)
            beta = config.get('prioritized_replay_beta', 0.4)
            self.memory = ProportionalPrioritizedReplayBuffer(
                self.buffer_size, 
                state_dim, 
                alpha=alpha, 
                beta=beta
            )
        else:
            self.memory = ReplayBuffer(self.buffer_size, state_dim)
            
        # DQN variants
        self.use_double = config.get('double_q', False)
        self.n_step = config.get('n_step', 1)
        
        # Training info
        self.last_target_update = 0
            
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether to use exploration or not
            
        Returns:
            action: Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, 
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
        # Only update if we have enough samples and it's time to update
        if (len(self.memory) < self.learning_starts or
            self.training_steps % self.train_freq != 0):
            return {}
            
        # Sample experiences from the replay buffer
        if isinstance(self.memory, ProportionalPrioritizedReplayBuffer):
            (states, actions, rewards, next_states, dones, 
             weights, indices) = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
            
        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            if self.use_double:
                # Double DQN: Select actions using the online network
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                # Evaluate the actions using the target network
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Regular DQN: Just use maximum Q-value from the target network
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update priorities if using prioritized replay
        td_errors = target_q_values - q_values
        if indices is not None:
            self.memory.update_priorities(indices, np.abs(td_errors.detach().cpu().numpy()) + 1e-6)
            
        # Use Huber loss for stability
        loss = F.smooth_l1_loss(q_values, target_q_values, reduction='none')
        loss = (loss * weights).mean()  # Apply importance sampling weights
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Periodically update the target network
        if self.training_steps - self.last_target_update >= self.target_update_freq:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.last_target_update = self.training_steps
            
        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': loss.item(),
            'q_values': q_values.mean().item(),
            'epsilon': self.epsilon
        }
    
    def save(self, path: str):
        """Save the agent's parameters.
        
        Args:
            path: Path to save the parameters
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
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
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']
        self.best_reward = checkpoint['best_reward']
