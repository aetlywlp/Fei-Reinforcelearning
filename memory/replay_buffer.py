"""
Replay buffer implementations for experience replay.

This module provides various replay buffer implementations for storing
and sampling transitions in reinforcement learning algorithms.
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Basic replay buffer for storing transitions.
    
    This class implements a simple circular buffer for storing transitions
    and sampling random batches for off-policy learning.
    """
    
    def __init__(
        self, 
        capacity: int,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: Optional[int] = None,
        n_step: int = 1,
        gamma: float = 0.99
    ):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space (None for discrete actions)
            n_step: Number of steps for n-step returns (1 for regular returns)
            gamma: Discount factor for n-step returns
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma
        
        # Convert state_dim to a consistent format
        if isinstance(state_dim, int):
            self.state_shape = (state_dim,)
        else:
            self.state_shape = state_dim
            
        # Determine action shape based on action_dim
        if action_dim is None:
            # Discrete actions (integers)
            self.action_shape = ()
        elif isinstance(action_dim, int) and action_dim == 1:
            # Continuous actions (scalars)
            self.action_shape = ()
        else:
            # Continuous actions (vectors)
            self.action_shape = (action_dim,)
        
        # Initialize buffers
        self.reset()
        
        # n-step return buffer
        if n_step > 1:
            self.n_step_buffer = deque(maxlen=n_step)
            
    def reset(self):
        """Reset the buffer."""
        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        
        if not self.action_shape:
            # Discrete actions or scalar continuous actions
            self.actions = np.zeros(self.capacity, dtype=np.float32 if self.action_dim is not None else np.int64)
        else:
            # Vector continuous actions
            self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.float32)
            
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        
        self.size = 0
        self.position = 0
        
    def add(
        self, 
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if self.n_step > 1:
            # Add to n-step buffer
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            # Only add to replay buffer if we have enough transitions
            if len(self.n_step_buffer) < self.n_step:
                return
                
            # Calculate n-step return
            state, action = self.n_step_buffer[0][:2]
            next_state, done = self.n_step_buffer[-1][3:]
            reward = self._get_n_step_return()
            
            # If any transition in the n-step buffer is done, mark as done
            for _, _, _, _, d in self.n_step_buffer:
                if d:
                    done = True
                    break
        
        # Store transition
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def _get_n_step_return(self) -> float:
        """Calculate the n-step return.
        
        Returns:
            n_step_return: The n-step return
        """
        n_step_return = 0
        for i in range(len(self.n_step_buffer)):
            reward = self.n_step_buffer[i][2]
            n_step_return += reward * (self.gamma ** i)
            
        return n_step_return
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
        
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return self.size


class PrioritizedReplayBuffer:
    """Prioritized replay buffer for storing transitions.
    
    This class implements a prioritized replay buffer as described in
    "Prioritized Experience Replay" (Schaul et al., 2015).
    """
    
    def __init__(
        self, 
        capacity: int,
        state_dim: Union[int, Tuple[int, ...]],
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        action_dim: Optional[int] = None,
        n_step: int = 1,
        gamma: float = 0.99
    ):
        """Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimensions of the state space
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
            beta_increment: Increment for beta parameter per sampling
            epsilon: Small constant to add to priorities to ensure non-zero probabilities
            action_dim: Dimensions of the action space (None for discrete actions)
            n_step: Number of steps for n-step returns (1 for regular returns)
            gamma: Discount factor for n-step returns
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma
        
        # Tree for storing priorities
        self.tree = SumTree(capacity)
        
        # Convert state_dim to a consistent format
        if isinstance(state_dim, int):
            self.state_shape = (state_dim,)
        else:
            self.state_shape = state_dim
            
        # Determine action shape based on action_dim
        if action_dim is None:
            # Discrete actions (integers)
            self.action_shape = ()
        elif isinstance(action_dim, int) and action_dim == 1:
            # Continuous actions (scalars)
            self.action_shape = ()
        else:
            # Continuous actions (vectors)
            self.action_shape = (action_dim,)
        
        # Initialize buffers
        self.states = np.zeros((capacity, *self.state_shape), dtype=np.float32)
        
        if not self.action_shape:
            # Discrete actions or scalar continuous actions
            self.actions = np.zeros(capacity, dtype=np.float32 if action_dim is not None else np.int64)
        else:
            # Vector continuous actions
            self.actions = np.zeros((capacity, *self.action_shape), dtype=np.float32)
            
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *self.state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # n-step return buffer
        if n_step > 1:
            self.n_step_buffer = deque(maxlen=n_step)
            
        self.max_priority = 1.0
        
    def add(
        self, 
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if self.n_step > 1:
            # Add to n-step buffer
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            # Only add to replay buffer if we have enough transitions
            if len(self.n_step_buffer) < self.n_step:
                return
                
            # Calculate n-step return
            state, action = self.n_step_buffer[0][:2]
            next_state, done = self.n_step_buffer[-1][3:]
            reward = self._get_n_step_return()
            
            # If any transition in the n-step buffer is done, mark as done
            for _, _, _, _, d in self.n_step_buffer:
                if d:
                    done = True
                    break
        
        # Get the current position in the buffer
        idx = self.tree.add(self.max_priority)
        
        # Store transition
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
    def _get_n_step_return(self) -> float:
        """Calculate the n-step return.
        
        Returns:
            n_step_return: The n-step return
        """
        n_step_return = 0
        for i in range(len(self.n_step_buffer)):
            reward = self.n_step_buffer[i][2]
            n_step_return += reward * (self.gamma ** i)
            
        return n_step_return
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions with priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            weights: Importance sampling weights
            indices: Indices of sampled transitions
        """
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate segment size
        segment_size = self.tree.total() / batch_size
        
        # Increase beta over time to reduce importance sampling bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate min priority for normalization
        min_priority = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        
        # Sample from each segment
        for i in range(batch_size):
            # Get a value from this segment
            a = segment_size * i
            b = segment_size * (i + 1)
            value = np.random.uniform(a, b)
            
            # Get the corresponding index and priority
            idx, priority = self.tree.get(value)
            
            # Calculate weight
            sampling_probability = priority / self.tree.total()
            weights[i] = (sampling_probability * self.tree.size) ** -self.beta
            batch_indices[i] = idx
            
        # Normalize weights
        weights /= weights.max()
        
        return (
            self.states[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_states[batch_indices],
            self.dones[batch_indices],
            weights,
            batch_indices
        )
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled transitions.
        
        Args:
            indices: Indices of sampled transitions
            priorities: New priorities for the transitions
        """
        for idx, priority in zip(indices, priorities):
            # Add a small positive value to avoid zero probabilities
            priority = (priority + self.epsilon) ** self.alpha
            
            # Update priority in the tree
            self.tree.update(idx, priority)
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return self.tree.size


class SumTree:
    """Sum tree data structure for efficient sampling with priorities.
    
    This class implements a binary tree where internal nodes store the
    sum of their children and leaf nodes store the priorities.
    """
    
    def __init__(self, capacity: int):
        """Initialize the sum tree.
        
        Args:
            capacity: Maximum number of leaf nodes
        """
        self.capacity = capacity
        
        # Complete binary tree with 2*capacity-1 nodes
        # Tree structure: [0, 1, 2, ..., 2*capacity-2]
        # Leaf nodes: [capacity-1, capacity, ..., 2*capacity-2]
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        
        # Data storage
        self.size = 0
        self.next_idx = 0
        
    def add(self, priority: float) -> int:
        """Add a new priority to the tree.
        
        Args:
            priority: Priority value
            
        Returns:
            idx: Index of the new leaf in the original data storage
        """
        # Get the leaf index in the tree
        tree_idx = self.next_idx + self.capacity - 1
        
        # Store the priority value in the tree
        self.update(tree_idx, priority)
        
        # Get the corresponding data index
        data_idx = self.next_idx
        
        # Update next index
        self.next_idx = (self.next_idx + 1) % self.capacity
        
        # Update size
        self.size = min(self.size + 1, self.capacity)
        
        return data_idx
        
    def update(self, tree_idx: int, priority: float):
        """Update a priority value in the tree.
        
        Args:
            tree_idx: Index of the leaf in the tree
            priority: New priority value
        """
        # Calculate the change in priority
        change = priority - self.tree[tree_idx]
        
        # Update the leaf node
        self.tree[tree_idx] = priority
        
        # Update parent nodes
        self._propagate(tree_idx, change)
        
    def _propagate(self, tree_idx: int, change: float):
        """Propagate the change in priority up the tree.
        
        Args:
            tree_idx: Index of the leaf in the tree
            change: Change in priority
        """
        # Get parent index
        parent = (tree_idx - 1) // 2
        
        # Update parent
        self.tree[parent] += change
        
        # Continue propagation if not at root
        if parent > 0:
            self._propagate(parent, change)
            
    def get(self, value: float) -> Tuple[int, float]:
        """Get the leaf index and priority for a given value.
        
        Args:
            value: Value to search for
            
        Returns:
            idx: Index of the leaf in the original data storage
            priority: Priority value at the leaf
        """
        # Start from root
        idx = self._retrieve(0, value)
        
        # Get the data index
        data_idx = idx - self.capacity + 1
        
        return data_idx, self.tree[idx]
        
    def _retrieve(self, idx: int, value: float) -> int:
        """Retrieve the leaf index for a given value.
        
        Args:
            idx: Current tree index
            value: Value to search for
            
        Returns:
            leaf_idx: Index of the leaf in the tree
        """
        # Check if we reached a leaf
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        # Go left if value is smaller than left child
        if value <= self.tree[left]:
            return self._retrieve(left, value)
            
        # Otherwise go right and update value
        return self._retrieve(right, value - self.tree[left])
        
    def total(self) -> float:
        """Get the total priority in the tree.
        
        Returns:
            total: Total priority
        """
        return self.tree[0]


class EpisodeBuffer:
    """Buffer for storing complete episodes for on-policy algorithms.
    
    This class implements a buffer for storing complete episodes,
    which is useful for on-policy algorithms like PPO and TRPO.
    """
    
    def __init__(self):
        """Initialize the episode buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(
        self, 
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        next_state: np.ndarray, 
        done: bool, 
        log_prob: Optional[float] = None,
        value: Optional[float] = None
    ):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            log_prob: Log probability of the action
            value: Estimated value of the state
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if log_prob is not None:
            self.log_probs.append(log_prob)
            
        if value is not None:
            self.values.append(value)
            
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def get(self) -> Dict[str, Union[List, np.ndarray]]:
        """Get the current contents of the buffer.
        
        Returns:
            data: Dictionary containing the buffer data
        """
        data = {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'next_states': np.array(self.next_states),
            'dones': np.array(self.dones)
        }
        
        if self.log_probs:
            data['log_probs'] = np.array(self.log_probs)
            
        if self.values:
            data['values'] = np.array(self.values)
            
        return data
        
    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """Compute the returns for each timestep.
        
        Args:
            gamma: Discount factor
            
        Returns:
            returns: Array of returns
        """
        returns = []
        G = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            G = reward + gamma * G * (1 - done)
            returns.insert(0, G)
            
        return np.array(returns)
        
    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
        """Compute Generalized Advantage Estimation (GAE).
        
        Args:
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            advantages: Array of advantages
        """
        if not self.values:
            raise ValueError("Cannot compute GAE without state values")
            
        # Create a copy of values with an additional value for the terminal state
        values = np.array(self.values + [0.0])
        
        # Initialize advantages
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
            
        return advantages
        
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.states)
