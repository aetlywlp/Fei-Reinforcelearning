"""
Prioritized Experience Replay implementations.

This module provides specialized prioritized replay buffer implementations
that can be used with various reinforcement learning algorithms.
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import torch
from memory.replay_buffer import SumTree


class ProportionalPrioritizedReplayBuffer:
    """Prioritized Replay Buffer based on proportional prioritization.
    
    This class implements a proportional prioritized replay buffer as described in
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
        action_dim: Optional[int] = None
    ):
        """Initialize the proportional prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimensions of the state space
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
            beta_increment: Increment for beta parameter per sampling
            epsilon: Small constant to add to priorities to ensure non-zero probabilities
            action_dim: Dimensions of the action space (None for discrete actions)
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.action_dim = action_dim
        
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
        # Get the current position in the buffer
        idx = self.tree.add(self.max_priority)
        
        # Store transition
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
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


class RankBasedPrioritizedReplayBuffer:
    """Prioritized Replay Buffer based on rank-based prioritization.
    
    This class implements a rank-based prioritized replay buffer as described in
    "Prioritized Experience Replay" (Schaul et al., 2015).
    """
    
    def __init__(
        self, 
        capacity: int,
        state_dim: Union[int, Tuple[int, ...]],
        alpha: float = 0.7,
        beta: float = 0.5,
        beta_increment: float = 0.001,
        action_dim: Optional[int] = None
    ):
        """Initialize the rank-based prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimensions of the state space
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
            beta_increment: Increment for beta parameter per sampling
            action_dim: Dimensions of the action space (None for discrete actions)
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.action_dim = action_dim
        
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
        
        # Store TD errors for each transition
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        self.size = 0
        self.next_idx = 0
        
    def add(
        self, 
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        priority: Optional[float] = None
    ):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            priority: Priority of the transition (if None, max priority is used)
        """
        # Store transition
        self.states[self.next_idx] = state
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.next_states[self.next_idx] = next_state
        self.dones[self.next_idx] = done
        
        # Store priority (default to max priority if not provided)
        if priority is None:
            if self.size == 0:
                priority = 1.0
            else:
                priority = np.max(self.priorities[:self.size])
                
        self.priorities[self.next_idx] = priority
        
        # Update next index
        self.next_idx = (self.next_idx + 1) % self.capacity
        
        # Update size
        self.size = min(self.size + 1, self.capacity)
        
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
        # Increase beta over time to reduce importance sampling bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get the valid size
        size = min(self.size, self.capacity)
        
        # Sort priorities (only the first 'size' elements)
        sorted_priorities_idx = np.argsort(self.priorities[:size])
        
        # Compute rank-based probabilities
        # P(i) = 1/rank(i)^alpha / sum_j(1/rank(j)^alpha)
        ranks = size - np.arange(size)
        probs = (1.0 / ranks) ** self.alpha
        probs /= np.sum(probs)
        
        # Sample batch of indices
        batch_indices = np.random.choice(
            sorted_priorities_idx,
            size=batch_size,
            p=probs,
            replace=False
        )
        
        # Compute importance sampling weights
        weights = (size * probs[batch_indices]) ** -self.beta
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
        self.priorities[indices] = priorities
            
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return self.size


class HindsightExperienceReplayBuffer:
    """Hindsight Experience Replay (HER) buffer for goal-conditioned RL.
    
    This class implements a replay buffer that supports Hindsight Experience Replay
    as described in "Hindsight Experience Replay" (Andrychowicz et al., 2017).
    """
    
    def __init__(
        self, 
        capacity: int,
        state_dim: Union[int, Tuple[int, ...]],
        goal_dim: Union[int, Tuple[int, ...]],
        strategy: str = 'future',
        k_goals: int = 4,
        action_dim: Optional[int] = None
    ):
        """Initialize the HER buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimensions of the state space
            goal_dim: Dimensions of the goal space
            strategy: Strategy for selecting additional goals ('future', 'episode', 'random')
            k_goals: Number of additional goals to sample for each transition
            action_dim: Dimensions of the action space (None for discrete actions)
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.strategy = strategy
        self.k_goals = k_goals
        self.action_dim = action_dim
        
        # Convert state_dim and goal_dim to a consistent format
        if isinstance(state_dim, int):
            self.state_shape = (state_dim,)
        else:
            self.state_shape = state_dim
            
        if isinstance(goal_dim, int):
            self.goal_shape = (goal_dim,)
        else:
            self.goal_shape = goal_dim
            
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
        self.goals = np.zeros((capacity, *self.goal_shape), dtype=np.float32)
        self.achieved_goals = np.zeros((capacity, *self.goal_shape), dtype=np.float32)
        
        if not self.action_shape:
            # Discrete actions or scalar continuous actions
            self.actions = np.zeros(capacity, dtype=np.float32 if action_dim is not None else np.int64)
        else:
            # Vector continuous actions
            self.actions = np.zeros((capacity, *self.action_shape), dtype=np.float32)
            
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *self.state_shape), dtype=np.float32)
        self.next_achieved_goals = np.zeros((capacity, *self.goal_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # For tracking episodes
        self.episode_start_indices = []
        self.current_episode_start = 0
        
        self.size = 0
        self.next_idx = 0
        
    def add(
        self, 
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        goal: np.ndarray,
        achieved_goal: np.ndarray,
        next_achieved_goal: np.ndarray,
        compute_reward_func: Optional[callable] = None
    ):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            goal: Goal for the transition
            achieved_goal: Achieved goal for the current state
            next_achieved_goal: Achieved goal for the next state
            compute_reward_func: Function to compute reward given (achieved_goal, goal, info)
        """
        # Store transition
        self.states[self.next_idx] = state
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.next_states[self.next_idx] = next_state
        self.dones[self.next_idx] = done
        self.goals[self.next_idx] = goal
        self.achieved_goals[self.next_idx] = achieved_goal
        self.next_achieved_goals[self.next_idx] = next_achieved_goal
        
        # Update next index
        self.next_idx = (self.next_idx + 1) % self.capacity
        
        # Update size
        self.size = min(self.size + 1, self.capacity)
        
        # If episode ended, store the episode start index
        if done:
            self.episode_start_indices.append(
                (self.current_episode_start, self.next_idx - 1 if self.next_idx > 0 else self.capacity - 1)
            )
            self.current_episode_start = self.next_idx
            
            # Keep only the most recent episodes that fit in the buffer
            while len(self.episode_start_indices) > 0 and self.episode_start_indices[0][0] < self.next_idx - self.capacity:
                self.episode_start_indices.pop(0)
                
    def sample(self, batch_size: int, compute_reward_func: callable) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Sample a batch of transitions with HER.
        
        Args:
            batch_size: Number of transitions to sample
            compute_reward_func: Function to compute reward given (achieved_goal, goal, info)
            
        Returns:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        if self.size < batch_size:
            # Not enough samples in the buffer
            return None
            
        # Sample regular transitions
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Prepare batch data
        states_batch = {
            'observation': self.states[indices].copy(),
            'achieved_goal': self.achieved_goals[indices].copy(),
            'desired_goal': self.goals[indices].copy()
        }
        
        actions_batch = self.actions[indices].copy()
        
        next_states_batch = {
            'observation': self.next_states[indices].copy(),
            'achieved_goal': self.next_achieved_goals[indices].copy(),
            'desired_goal': self.goals[indices].copy()
        }
        
        dones_batch = self.dones[indices].copy()
        
        # Compute rewards
        rewards_batch = compute_reward_func(
            next_states_batch['achieved_goal'],
            next_states_batch['desired_goal'],
            None
        )
        
        # Apply HER: replace some of the goals
        her_indices = np.random.randint(0, batch_size, size=int(batch_size * 0.8))
        future_offset = np.random.randint(1, 10, size=len(her_indices))
        
        # Find valid future goals for each selected transition
        for i, (orig_idx, offset) in enumerate(zip(her_indices, future_offset)):
            idx = indices[orig_idx]
            
            # Find the episode this transition belongs to
            episode_start, episode_end = None, None
            for start, end in reversed(self.episode_start_indices):
                if start <= idx <= end:
                    episode_start, episode_end = start, end
                    break
                    
            if episode_start is None:
                # This transition is from the current episode
                episode_start = self.current_episode_start
                episode_end = (self.next_idx - 1) % self.capacity
                
            # Compute the index of the future state to use as goal
            if idx + offset <= episode_end:
                future_idx = (idx + offset) % self.capacity
            else:
                future_idx = episode_end
                
            # Replace the goal with the achieved goal from the future state
            future_ag = self.next_achieved_goals[future_idx]
            states_batch['desired_goal'][orig_idx] = future_ag
            next_states_batch['desired_goal'][orig_idx] = future_ag
            
            # Recompute the reward with the new goal
            rewards_batch[orig_idx] = compute_reward_func(
                next_states_batch['achieved_goal'][orig_idx],
                future_ag,
                None
            )
            
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
        
    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return self.size
