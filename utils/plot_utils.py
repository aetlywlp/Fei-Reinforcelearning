"""
Plotting utilities for reinforcement learning.

This module provides functions for visualizing reinforcement learning
training progress, policies, and environment interactions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
import torch


def plot_learning_curves(
    data: Dict[str, List[float]],
    x_key: str = 'steps',
    y_keys: Optional[List[str]] = None,
    window_size: int = 10,
    title: str = 'Learning Curves',
    xlabel: str = 'Steps',
    ylabel: str = 'Value',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot learning curves from training data.
    
    Args:
        data: Dictionary containing training metrics
        x_key: Key for x-axis values
        y_keys: Keys for y-axis values (if None, plot all except x_key)
        window_size: Window size for smoothing
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If y_keys is None, use all keys except x_key
    if y_keys is None:
        y_keys = [key for key in data.keys() if key != x_key]
    
    # Get x values
    x = data[x_key]
    
    # Plot each y key
    for key in y_keys:
        # Get y values
        y = data[key]
        
        # Ensure x and y have the same length
        length = min(len(x), len(y))
        x_plot = x[:length]
        y_plot = y[:length]
        
        # Plot raw data
        ax.plot(x_plot, y_plot, alpha=0.3, label=f"{key} (raw)")
        
        # Plot smoothed data
        if window_size > 1 and length > window_size:
            # Apply rolling window
            y_smooth = np.convolve(y_plot, np.ones(window_size) / window_size, mode='valid')
            x_smooth = x_plot[window_size-1:]
            
            # Plot smoothed data
            ax.plot(x_smooth, y_smooth, label=f"{key} (smoothed)")
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_episode_rewards(
    rewards: List[float],
    window_size: int = 10,
    title: str = 'Episode Rewards',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot episode rewards.
    
    Args:
        rewards: List of episode rewards
        window_size: Window size for smoothing
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate episode numbers
    episodes = np.arange(1, len(rewards) + 1)
    
    # Plot raw rewards
    ax.plot(episodes, rewards, alpha=0.3, label='Raw rewards')
    
    # Plot smoothed rewards
    if window_size > 1 and len(rewards) > window_size:
        # Apply rolling window
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        # Plot smoothed rewards
        ax.plot(smoothed_episodes, smoothed_rewards, label='Smoothed rewards')
    
    # Set labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_action_distribution(
    actions: Union[List[int], np.ndarray],
    num_actions: Optional[int] = None,
    title: str = 'Action Distribution',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot the distribution of actions.
    
    Args:
        actions: List or array of actions
        num_actions: Number of possible actions (if None, infer from actions)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert actions to numpy array
    actions = np.array(actions)
    
    # Infer number of actions if not specified
    if num_actions is None:
        num_actions = int(np.max(actions)) + 1
    
    # Count actions
    action_counts = np.zeros(num_actions)
    for action in actions:
        action_counts[action] += 1
    
    # Normalize to probabilities
    action_probs = action_counts / len(actions)
    
    # Create bar plot
    ax.bar(np.arange(num_actions), action_probs)
    
    # Set labels and title
    ax.set_xlabel('Action')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.set_xticks(np.arange(num_actions))
    ax.grid(alpha=0.3, axis='y')
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_value_function(
    state_values: Union[List[float], np.ndarray],
    state_labels: Optional[List[str]] = None,
    title: str = 'State Values',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot state values.
    
    Args:
        state_values: List or array of state values
        state_labels: Labels for states (if None, use indices)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert state values to numpy array
    state_values = np.array(state_values)
    
    # Generate state indices if labels not specified
    if state_labels is None:
        state_labels = [str(i) for i in range(len(state_values))]
    
    # Create bar plot
    ax.bar(state_labels, state_values)
    
    # Set labels and title
    ax.set_xlabel('State')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_q_values(
    q_values: Union[List[List[float]], np.ndarray],
    state_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    title: str = 'Q-Values',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot Q-values as a heatmap.
    
    Args:
        q_values: 2D list or array of Q-values (state x action)
        state_labels: Labels for states (if None, use indices)
        action_labels: Labels for actions (if None, use indices)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert Q-values to numpy array
    q_values = np.array(q_values)
    
    # Generate state and action labels if not specified
    num_states, num_actions = q_values.shape
    
    if state_labels is None:
        state_labels = [f"S{i}" for i in range(num_states)]
    
    if action_labels is None:
        action_labels = [f"A{i}" for i in range(num_actions)]
    
    # Create heatmap
    im = ax.imshow(q_values, cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Q-Value')
    
    # Set labels and title
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_title(title)
    
    # Set tick labels
    ax.set_xticks(np.arange(num_actions))
    ax.set_yticks(np.arange(num_states))
    ax.set_xticklabels(action_labels)
    ax.set_yticklabels(state_labels)
    
    # Rotate x tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(num_states):
        for j in range(num_actions):
            text = ax.text(j, i, f"{q_values[i, j]:.2f}",
                           ha="center", va="center", color="w")
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_policy(
    policy: Union[List[int], np.ndarray],
    state_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    title: str = 'Policy',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot a deterministic policy.
    
    Args:
        policy: List or array of actions for each state
        state_labels: Labels for states (if None, use indices)
        action_labels: Labels for actions (if None, use indices)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert policy to numpy array
    policy = np.array(policy)
    
    # Get number of states and actions
    num_states = len(policy)
    num_actions = int(np.max(policy)) + 1
    
    # Generate state and action labels if not specified
    if state_labels is None:
        state_labels = [f"S{i}" for i in range(num_states)]
    
    if action_labels is None:
        action_labels = [f"A{i}" for i in range(num_actions)]
    
    # Create a policy matrix (one-hot encoded)
    policy_matrix = np.zeros((num_states, num_actions))
    for i, action in enumerate(policy):
        policy_matrix[i, action] = 1
    
    # Create heatmap
    im = ax.imshow(policy_matrix, cmap='Blues')
    
    # Set labels and title
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_title(title)
    
    # Set tick labels
    ax.set_xticks(np.arange(num_actions))
    ax.set_yticks(np.arange(num_states))
    ax.set_xticklabels(action_labels)
    ax.set_yticklabels(state_labels)
    
    # Rotate x tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_stochastic_policy(
    policy: Union[List[List[float]], np.ndarray],
    state_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    title: str = 'Stochastic Policy',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot a stochastic policy as a heatmap.
    
    Args:
        policy: 2D list or array of action probabilities (state x action)
        state_labels: Labels for states (if None, use indices)
        action_labels: Labels for actions (if None, use indices)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert policy to numpy array
    policy = np.array(policy)
    
    # Generate state and action labels if not specified
    num_states, num_actions = policy.shape
    
    if state_labels is None:
        state_labels = [f"S{i}" for i in range(num_states)]
    
    if action_labels is None:
        action_labels = [f"A{i}" for i in range(num_actions)]
    
    # Create heatmap
    im = ax.imshow(policy, cmap='Blues', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Probability')
    
    # Set labels and title
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_title(title)
    
    # Set tick labels
    ax.set_xticks(np.arange(num_actions))
    ax.set_yticks(np.arange(num_states))
    ax.set_xticklabels(action_labels)
    ax.set_yticklabels(state_labels)
    
    # Rotate x tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(num_states):
        for j in range(num_actions):
            text = ax.text(j, i, f"{policy[i, j]:.2f}",
                           ha="center", va="center", 
                           color="black" if policy[i, j] < 0.5 else "white")
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_grid_world_policy(
    policy: Union[List[int], np.ndarray],
    grid_shape: Tuple[int, int],
    action_to_arrow: Optional[Dict[int, str]] = None,
    title: str = 'Grid World Policy',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot a grid world policy with arrows.
    
    Args:
        policy: List or array of actions for each state
        grid_shape: Shape of the grid (rows, cols)
        action_to_arrow: Mapping from action to arrow symbol (if None, use default)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert policy to numpy array
    policy = np.array(policy)
    
    # Default action to arrow mapping (Up, Right, Down, Left)
    if action_to_arrow is None:
        action_to_arrow = {
            0: "↑",
            1: "→",
            2: "↓",
            3: "←"
        }
    
    # Create grid
    rows, cols = grid_shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    
    # Draw grid lines
    for i in range(cols + 1):
        ax.axvline(i, color='black', alpha=0.3)
    for i in range(rows + 1):
        ax.axhline(i, color='black', alpha=0.3)
    
    # Plot arrows for each cell
    for i in range(rows):
        for j in range(cols):
            state_idx = i * cols + j
            if state_idx < len(policy):
                action = policy[state_idx]
                arrow = action_to_arrow.get(action, "?")
                ax.text(j + 0.5, rows - i - 0.5, arrow, 
                        ha='center', va='center', fontsize=20)
    
    # Set labels and title
    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker=arrow, color='w', label=f'Action {action}',
                  markersize=15, markeredgecolor='black')
        for action, arrow in action_to_arrow.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_grid_world_values(
    values: Union[List[float], np.ndarray],
    grid_shape: Tuple[int, int],
    title: str = 'Grid World Values',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot state values for a grid world.
    
    Args:
        values: List or array of state values
        grid_shape: Shape of the grid (rows, cols)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert values to numpy array
    values = np.array(values)
    
    # Reshape values to grid
    rows, cols = grid_shape
    grid_values = values.reshape((rows, cols))
    
    # Create heatmap
    im = ax.imshow(grid_values, cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Value')
    
    # Set labels and title
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_title(title)
    
    # Loop over data dimensions and create text annotations
    for i in range(rows):
        for j in range(cols):
            text = ax.text(j, i, f"{grid_values[i, j]:.2f}",
                           ha="center", va="center", color="w")
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_frames_animation(
    frames: List[np.ndarray],
    fps: int = 30,
    title: str = 'Animation',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create an animation from frames.
    
    Args:
        frames: List of frames (numpy arrays)
        fps: Frames per second
        title: Plot title
        figsize: Figure size
        save_path: Path to save the animation (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create initial frame
    im = ax.imshow(frames[0])
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(title)
    
    # Create animation
    def update(i):
        im.set_array(frames[i])
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    
    # Save animation if requested
    if save_path is not None:
        ani.save(save_path, fps=fps)
    
    return fig, ani


def plot_network_weights(
    model: torch.nn.Module,
    layer_indices: Optional[List[int]] = None,
    title: str = 'Network Weights',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot the weights of a neural network.
    
    Args:
        model: PyTorch neural network
        layer_indices: Indices of layers to plot (if None, plot all linear layers)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    # Get all linear layers
    linear_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(module)
    
    # Filter layers by indices
    if layer_indices is not None:
        linear_layers = [linear_layers[i] for i in layer_indices if i < len(linear_layers)]
    
    # Check if we have any layers to plot
    if not linear_layers:
        raise ValueError("No linear layers to plot")
    
    # Create figure
    fig, axes = plt.subplots(len(linear_layers), 1, figsize=figsize)
    if len(linear_layers) == 1:
        axes = [axes]
    
    # Plot each layer
    for i, layer in enumerate(linear_layers):
        # Get weights
        weights = layer.weight.data.cpu().numpy()
        
        # Create heatmap
        im = axes[i].imshow(weights, cmap='viridis')
        
        # Add colorbar
        fig.colorbar(im, ax=axes[i])
        
        # Set labels
        axes[i].set_xlabel('Input Neuron')
        axes[i].set_ylabel('Output Neuron')
        axes[i].set_title(f'Layer {i+1} Weights ({weights.shape[0]} x {weights.shape[1]})')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_reward_landscape(
    env_fn: Callable[[], Any],
    policy_fn: Callable[[np.ndarray, float, float], np.ndarray],
    param1_range: Tuple[float, float],
    param2_range: Tuple[float, float],
    n_params: int = 20,
    episodes_per_param: int = 5,
    title: str = 'Reward Landscape',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot the reward landscape for a parameterized policy.
    
    Args:
        env_fn: Function that creates an environment
        policy_fn: Function that takes (observation, param1, param2) and returns an action
        param1_range: Range for parameter 1
        param2_range: Range for parameter 2
        n_params: Number of parameter values to try
        episodes_per_param: Number of episodes to run per parameter setting
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, don't save)
    
    Returns:
        fig: Matplotlib figure
    """
    # Generate parameter grids
    param1_values = np.linspace(param1_range[0], param1_range[1], n_params)
    param2_values = np.linspace(param2_range[0], param2_range[1], n_params)
    param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
    
    # Initialize reward grid
    reward_grid = np.zeros_like(param1_grid)
    
    # Evaluate policy for each parameter setting
    for i in range(n_params):
        for j in range(n_params):
            # Get parameter values
            param1 = param1_values[i]
            param2 = param2_values[j]
            
            # Run episodes
            total_reward = 0
            
            for _ in range(episodes_per_param):
                # Create environment
                env = env_fn()
                
                # Run episode
                observation, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # Get action from policy
                    action = policy_fn(observation, param1, param2)
                    
                    # Step environment
                    observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                
                total_reward += episode_reward
            
            # Average reward
            reward_grid[j, i] = total_reward / episodes_per_param
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.pcolormesh(param1_grid, param2_grid, reward_grid, cmap='viridis', shading='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Average Reward')
    
    # Set labels and title
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_title(title)
    
    # Add contour lines
    contour = ax.contour(param1_grid, param2_grid, reward_grid, colors='white', alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Save figure if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
