"""
Miscellaneous utility functions for reinforcement learning.

This module provides various utility functions that don't fit into
other categories.
"""

import os
import random
import numpy as np
import torch
import yaml
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import gymnasium as gym


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_output_dir(base_dir: str, experiment_name: str) -> str:
    """Create output directory for an experiment.
    
    Args:
        base_dir: Base directory for outputs
        experiment_name: Name of the experiment
        
    Returns:
        output_dir: Path to output directory
    """
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def linear_schedule(
    initial_value: float,
    final_value: float,
    current_step: int,
    total_steps: int
) -> float:
    """Linear schedule for hyperparameter annealing.
    
    Args:
        initial_value: Initial value
        final_value: Final value
        current_step: Current step
        total_steps: Total steps
        
    Returns:
        value: Current value
    """
    fraction = max(1.0 - float(current_step) / float(total_steps), 0.0)
    return final_value + fraction * (initial_value - final_value)


def exponential_schedule(
    initial_value: float,
    final_value: float,
    current_step: int,
    total_steps: int,
    decay_rate: float = 0.01
) -> float:
    """Exponential schedule for hyperparameter annealing.
    
    Args:
        initial_value: Initial value
        final_value: Final value
        current_step: Current step
        total_steps: Total steps
        decay_rate: Decay rate
        
    Returns:
        value: Current value
    """
    decay = np.exp(-decay_rate * current_step / total_steps)
    return final_value + (initial_value - final_value) * decay


def discount_rewards(
    rewards: List[float],
    gamma: float
) -> np.ndarray:
    """Calculate discounted rewards.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        discounted_rewards: Array of discounted rewards
    """
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_sum = 0
    
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + gamma * running_sum
        discounted_rewards[i] = running_sum
    
    return discounted_rewards


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    dones: List[bool],
    gamma: float,
    lam: float
) -> np.ndarray:
    """Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of state values
        next_value: Value of the next state
        dones: List of done flags
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: Array of advantages
    """
    # Convert inputs to numpy arrays
    rewards = np.array(rewards)
    values = np.array(values)
    dones = np.array(dones, dtype=np.float32)
    
    # Append next_value to values
    values_extended = np.append(values, next_value)
    
    # Calculate temporal difference errors
    deltas = rewards + gamma * values_extended[1:] * (1 - dones) - values
    
    # Calculate advantages
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0
    
    for i in reversed(range(len(rewards))):
        gae = deltas[i] + gamma * lam * (1 - dones[i]) * gae
        advantages[i] = gae
    
    return advantages


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        explained_var: Explained variance
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    
    return 1 - np.var(y_true - y_pred) / var_y


def soft_update(
    source_network: torch.nn.Module,
    target_network: torch.nn.Module,
    tau: float
) -> None:
    """Soft update for target networks.
    
    Args:
        source_network: Source network
        target_network: Target network
        tau: Interpolation parameter
    """
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def hard_update(
    source_network: torch.nn.Module,
    target_network: torch.nn.Module
) -> None:
    """Hard update for target networks.
    
    Args:
        source_network: Source network
        target_network: Target network
    """
    target_network.load_state_dict(source_network.state_dict())


def compute_returns(
    rewards: List[float],
    dones: List[bool],
    gamma: float
) -> np.ndarray:
    """Compute returns (discounted sum of rewards).
    
    Args:
        rewards: List of rewards
        dones: List of done flags
        gamma: Discount factor
        
    Returns:
        returns: Array of returns
    """
    returns = []
    R = 0
    
    for reward, done in zip(reversed(rewards), reversed(dones)):
        R = reward + gamma * R * (1 - done)
        returns.insert(0, R)
    
    return np.array(returns)


def huber_loss(
    x: torch.Tensor,
    delta: float = 1.0
) -> torch.Tensor:
    """Compute Huber loss.
    
    Args:
        x: Input tensor
        delta: Threshold parameter
        
    Returns:
        loss: Huber loss
    """
    abs_x = torch.abs(x)
    return torch.where(
        abs_x < delta,
        0.5 * x ** 2,
        delta * (abs_x - 0.5 * delta)
    )


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    separator: str = '/'
) -> Dict[str, Any]:
    """Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Base key for flattened keys
        separator: Separator between nested keys
        
    Returns:
        flattened: Flattened dictionary
    """
    flattened = {}
    
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, new_key, separator))
        else:
            flattened[new_key] = v
    
    return flattened


def unflatten_dict(
    d: Dict[str, Any],
    separator: str = '/'
) -> Dict[str, Any]:
    """Unflatten a flattened dictionary.
    
    Args:
        d: Flattened dictionary
        separator: Separator between nested keys
        
    Returns:
        unflattened: Unflattened dictionary
    """
    unflattened = {}
    
    for key, value in d.items():
        parts = key.split(separator)
        
        current = unflattened
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return unflattened


def normalize(
    x: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize an array.
    
    Args:
        x: Array to normalize
        mean: Mean for normalization (if None, compute from x)
        std: Standard deviation for normalization (if None, compute from x)
        epsilon: Small constant for numerical stability
        
    Returns:
        normalized: Normalized array
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    
    if std is None:
        std = np.std(x, axis=0) + epsilon
    
    normalized = (x - mean) / std
    
    return normalized, mean, std


def standardize_rewards(
    rewards: List[float]
) -> np.ndarray:
    """Standardize rewards to have zero mean and unit variance.
    
    Args:
        rewards: List of rewards
        
    Returns:
        standardized: Standardized rewards
    """
    rewards = np.array(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def get_device() -> torch.device:
    """Get the device to use for PyTorch.
    
    Returns:
        device: PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_tensor(
    x: Union[np.ndarray, List],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convert a numpy array or list to a PyTorch tensor.
    
    Args:
        x: Numpy array or list
        device: PyTorch device (if None, use GPU if available)
        
    Returns:
        tensor: PyTorch tensor
    """
    if device is None:
        device = get_device()
    
    if isinstance(x, list):
        x = np.array(x)
    
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        return x.to(device)


def to_numpy(
    x: Union[torch.Tensor, np.ndarray]
) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array.
    
    Args:
        x: PyTorch tensor or numpy array
        
    Returns:
        array: Numpy array
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x


def batch_from_dict(
    batch: Dict[str, Any],
    keys: List[str],
    device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """Convert a batch dictionary to a list of tensors.
    
    Args:
        batch: Batch dictionary
        keys: Keys to extract
        device: PyTorch device (if None, use GPU if available)
        
    Returns:
        tensors: List of tensors
    """
    if device is None:
        device = get_device()
    
    return [to_tensor(batch[key], device) for key in keys]


def get_env_properties(env: gym.Env) -> Dict[str, Any]:
    """Get environment properties.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        properties: Dictionary of environment properties
    """
    properties = {
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'reward_range': env.reward_range
    }
    
    # Add observation space properties
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        properties['observation_dim'] = obs_space.shape
        properties['observation_shape'] = obs_space.shape
        properties['observation_low'] = obs_space.low
        properties['observation_high'] = obs_space.high
    elif isinstance(obs_space, gym.spaces.Discrete):
        properties['observation_dim'] = (obs_space.n,)
        properties['observation_shape'] = (obs_space.n,)
    elif isinstance(obs_space, gym.spaces.Dict):
        properties['observation_spaces'] = obs_space.spaces
    
    # Add action space properties
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        properties['action_dim'] = action_space.shape[0]
        properties['action_shape'] = action_space.shape
        properties['action_low'] = action_space.low
        properties['action_high'] = action_space.high
        properties['continuous_actions'] = True
    elif isinstance(action_space, gym.spaces.Discrete):
        properties['action_dim'] = action_space.n
        properties['action_shape'] = (action_space.n,)
        properties['continuous_actions'] = False
    
    return properties


def save_metrics(
    metrics: Dict[str, List[float]],
    save_path: str
) -> None:
    """Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save metrics
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    # Save metrics
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)


def load_metrics(
    load_path: str
) -> Dict[str, List[float]]:
    """Load metrics from a JSON file.
    
    Args:
        load_path: Path to load metrics from
        
    Returns:
        metrics: Dictionary of metrics
    """
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base
        
    Returns:
        merged_config: Merged configuration
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            merged_config[key] = merge_configs(base_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config
