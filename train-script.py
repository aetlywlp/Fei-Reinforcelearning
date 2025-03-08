"""
Training script for reinforcement learning algorithms.

This script provides a unified interface for training various
reinforcement learning algorithms on Gymnasium environments.
"""

import os
import time
import argparse
import yaml
import torch
import gymnasium as gym
import numpy as np
from typing import Dict, Any

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from environments.env_utils import make_env, make_vec_env,make_atari_env,make_mujoco_env
from utils.logger import Logger
from utils.misc import set_random_seed, create_output_dir, get_env_properties
from utils.plot_utils import plot_learning_curves, plot_episode_rewards



def create_environment(config):
    """根据配置创建适当的环境"""
    env_type = config.get('env_type', 'standard')
    env_id = config['env_id']
    seed = config.get('seed', 0)

    if env_type == 'atari':
        env = make_atari_env(
            env_id=env_id,
            seed=seed,
            frame_stack=config.get('frame_stack', 4),
            clip_rewards=config.get('clip_rewards', True)
        )
    elif env_type == 'mujoco':
        env = make_mujoco_env(
            env_id=env_id,
            seed=seed,
            normalize_obs=config.get('normalize_obs', True)
        )
    else:  # 标准环境
        env = make_env(
            env_id=env_id,
            seed=seed,
            frame_stack=config.get('frame_stack'),
            normalize_obs=config.get('normalize_obs', False)
        )

    return env


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a reinforcement learning agent')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory to save logs')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_agent(config: Dict[str, Any], env_props: Dict[str, Any], device: torch.device):
    """Create an agent based on the configuration.
    
    Args:
        config: Configuration dictionary
        env_props: Environment properties
        device: PyTorch device
    
    Returns:
        agent: Reinforcement learning agent
    """
    algorithm = config['algorithm'].lower()
    
    if algorithm == 'dqn':
        agent = DQNAgent(
            state_dim=env_props['observation_dim'],
            action_dim=env_props['action_dim'],
            config=config,
            device=device
        )
    elif algorithm == 'ppo':
        agent = PPOAgent(
            state_dim=env_props['observation_dim'],
            action_dim=env_props['action_dim'],
            config=config,
            continuous_actions=env_props.get('continuous_actions', False),
            device=device
        )
    elif algorithm == 'sac':
        if not env_props.get('continuous_actions', False):
            raise ValueError("SAC only supports continuous action spaces")
        
        agent = SACAgent(
            state_dim=env_props['observation_dim'],
            action_dim=env_props['action_dim'],
            config=config,
            device=device
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return agent


def train(config: Dict[str, Any], output_dir: str, device: str):
    """Train an agent based on the configuration.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save outputs
        device: Device to use ('cpu' or 'cuda')
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    
    # Get algorithm
    algorithm = config['algorithm'].lower()
    
    # Create environment
    env_id = config['env_id']
    
    # Check if the algorithm requires vectorized environments
    if algorithm == 'ppo' and config.get('num_envs', 1) > 1:
        env = make_vec_env(
            env_id=env_id,
            num_envs=config.get('num_envs', 1),
            seed=seed,
            frame_stack=config.get('frame_stack'),
            frame_skip=config.get('frame_skip'),
            normalize_obs=config.get('normalize_obs', False),
            clip_rewards=config.get('clip_rewards', False),
            time_limit=config.get('time_limit'),
            monitor=config.get('monitor', True)
        )
        
        # For PPO with vectorized environments, we'll need a separate environment for evaluation
        eval_env = make_env(
            env_id=env_id,
            seed=seed + 1000,
            frame_stack=config.get('frame_stack'),
            frame_skip=config.get('frame_skip'),
            normalize_obs=config.get('normalize_obs', False),
            clip_rewards=config.get('clip_rewards', False),
            time_limit=config.get('time_limit'),
            monitor=config.get('monitor', True)
        )
    else:
        env = make_env(
            env_id=env_id,
            seed=seed,
            frame_stack=config.get('frame_stack'),
            frame_skip=config.get('frame_skip'),
            normalize_obs=config.get('normalize_obs', False),
            clip_rewards=config.get('clip_rewards', False),
            time_limit=config.get('time_limit'),
            monitor=config.get('monitor', True)
        )
        eval_env = env
    
    # Get environment properties
    env_props = get_env_properties(eval_env)
    
    # Create agent
    agent = create_agent(config, env_props, device)
    
    # Create logger
    logger = Logger(
        log_dir=config.get('log_dir', 'logs'),
        experiment_name=config.get('experiment_name', f"{algorithm}_{env_id}"),
        use_tensorboard=config.get('use_tensorboard', True),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        wandb_entity=config.get('wandb_entity'),
        log_frequency=config.get('log_frequency', 1),
        verbose=config.get('verbose', True)
    )
    
    # Log configuration
    logger.log_hyperparams(config)
    
    # Load checkpoint if provided
    checkpoint_path = config.get('checkpoint')
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Training loop
    total_timesteps = config.get('total_timesteps', 100000)
    eval_frequency = config.get('eval_frequency', 10000)
    eval_episodes = config.get('eval_episodes', 10)
    checkpoint_frequency = config.get('checkpoint_frequency', 10000)
    
    episode = 0
    timestep = 0
    start_time = time.time()
    
    # For tracking metrics
    episode_rewards = []
    episode_lengths = []
    
    # Algorithm-specific training
    if algorithm == 'dqn' or algorithm == 'sac':
        # For off-policy algorithms
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while timestep < total_timesteps:
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            episode_reward += reward
            episode_length += 1
            timestep += 1
            
            # Update the agent
            loss_info = agent.update()
            
            # Log training info
            if timestep % config.get('log_frequency', 1) == 0 and loss_info:
                logger.log_metrics(loss_info, timestep)
            
            # End of episode
            if done:
                episode += 1
                
                # Log episode info
                episode_info = {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length
                }
                logger.log_episode(episode_info, episode)
                
                # Track metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Evaluation
            if timestep % eval_frequency == 0:
                eval_metrics = agent.evaluate(eval_env, num_episodes=eval_episodes)
                logger.log_metrics(
                    {'eval/' + k: v for k, v in eval_metrics.items()},
                    timestep
                )
                print(f"Evaluation at timestep {timestep}: {eval_metrics}")
            
            # Save checkpoint
            if timestep % checkpoint_frequency == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{timestep}.pt")
                agent.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Also save the latest version
                latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
                agent.save(latest_path)
    
    elif algorithm == 'ppo':
        # For on-policy algorithms with vectorized environments
        if hasattr(env, 'num_envs'):  # Vectorized environment
            num_envs = env.num_envs
            states, _ = env.reset()
            
            while timestep < total_timesteps:
                # Collect rollout
                rollout_timesteps = 0
                rollout_steps = config.get('rollout_steps', 2048)
                
                while rollout_timesteps < rollout_steps:
                    # Select actions
                    with torch.no_grad():
                        actions = []
                        for state in states:
                            action = agent.select_action(state)
                            actions.append(action)
                    
                    # Step environments
                    next_states, rewards, terminated, truncated, infos = env.step(actions)
                    dones = np.logical_or(terminated, truncated)
                    
                    # Store transitions
                    for i in range(num_envs):
                        agent.store_transition(
                            states[i], actions[i], rewards[i], next_states[i], dones[i]
                        )
                    
                    # Update states
                    states = next_states
                    
                    # Update counters
                    rollout_timesteps += num_envs
                    timestep += num_envs
                    
                    # Log episode info from vectorized environments
                    for i in range(num_envs):
                        if dones[i] and 'episode' in infos[i]:
                            episode += 1
                            ep_info = infos[i]['episode']
                            
                            # Track metrics
                            episode_rewards.append(ep_info['return'])
                            episode_lengths.append(ep_info['length'])
                            
                            # Log episode info
                            logger.log_episode(
                                {'episode_reward': ep_info['return'],
                                 'episode_length': ep_info['length']},
                                episode
                            )
                
                # Update agent
                loss_info = agent.update()
                
                # Log training info
                if loss_info:
                    logger.log_metrics(loss_info, timestep)
                
                # Evaluation
                if timestep % eval_frequency < num_envs:
                    eval_metrics = agent.evaluate(eval_env, num_episodes=eval_episodes)
                    logger.log_metrics(
                        {'eval/' + k: v for k, v in eval_metrics.items()},
                        timestep
                    )
                    print(f"Evaluation at timestep {timestep}: {eval_metrics}")
                
                # Save checkpoint
                if timestep % checkpoint_frequency < num_envs:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_{timestep}.pt")
                    agent.save(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Also save the latest version
                    latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
                    agent.save(latest_path)
        
        else:  # Single environment
            while timestep < total_timesteps:
                # Train for one episode
                info = agent.train_episode(env)
                
                # Update counters
                episode += 1
                timestep += info['steps']
                
                # Track metrics
                episode_rewards.append(info['reward'])
                episode_lengths.append(info['steps'])
                
                # Log episode info
                logger.log_episode(info, episode)
                
                # Evaluation
                if timestep % eval_frequency < info['steps']:
                    eval_metrics = agent.evaluate(eval_env, num_episodes=eval_episodes)
                    logger.log_metrics(
                        {'eval/' + k: v for k, v in eval_metrics.items()},
                        timestep
                    )
                    print(f"Evaluation at timestep {timestep}: {eval_metrics}")
                
                # Save checkpoint
                if timestep % checkpoint_frequency < info['steps']:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_{timestep}.pt")
                    agent.save(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Also save the latest version
                    latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
                    agent.save(latest_path)
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Final evaluation
    eval_metrics = agent.evaluate(eval_env, num_episodes=eval_episodes)
    logger.log_metrics(
        {'eval/' + k: v for k, v in eval_metrics.items()},
        timestep
    )
    print(f"Final evaluation: {eval_metrics}")
    
    # Save final model
    final_path = os.path.join(output_dir, "model_final.pt")
    agent.save(final_path)
    print(f"Saved final model to {final_path}")
    
    # Close environments
    env.close()
    if eval_env is not env:
        eval_env.close()
    
    # Close logger
    logger.close()
    
    # Plot results
    if episode_rewards:
        fig = plot_episode_rewards(
            episode_rewards,
            window_size=10,
            title=f"{algorithm.upper()} on {env_id}",
            figsize=(10, 6),
            save_path=os.path.join(output_dir, "rewards.png")
        )
    
    # Training summary
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f}s")
    print(f"Total episodes: {episode}")
    print(f"Total timesteps: {timestep}")
    if episode_rewards:
        print(f"Mean reward: {np.mean(episode_rewards):.2f}")
        print(f"Max reward: {np.max(episode_rewards):.2f}")


def evaluate(config: Dict[str, Any], output_dir: str, device: str, render: bool):
    """Evaluate an agent based on the configuration.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save outputs
        device: Device to use ('cpu' or 'cuda')
        render: Whether to render the environment
    """
    # Set random seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    
    # Get algorithm
    algorithm = config['algorithm'].lower()
    
    # Create environment
    env_id = config['env_id']
    env = make_env(
        env_id=env_id,
        seed=seed,
        frame_stack=config.get('frame_stack'),
        frame_skip=config.get('frame_skip'),
        normalize_obs=config.get('normalize_obs', False),
        clip_rewards=config.get('clip_rewards', False),
        time_limit=config.get('time_limit'),
        monitor=config.get('monitor', True),
        render_mode='human' if render else None
    )
    
    # Get environment properties
    env_props = get_env_properties(env)
    
    # Create agent
    agent = create_agent(config, env_props, device)
    
    # Load checkpoint
    checkpoint_path = config.get('checkpoint')
    if checkpoint_path is None:
        # Try to find the latest checkpoint
        checkpoint_path = os.path.join(output_dir, "model_final.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pt")
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}")
    
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Evaluate agent
    num_episodes = config.get('eval_episodes', 10)
    eval_metrics = agent.evaluate(env, num_episodes=num_episodes)
    
    print(f"Evaluation results:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")
    
    # Close environment
    env.close()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Override log directory if provided
    if args.log_dir is not None:
        config['log_dir'] = args.log_dir
    
    # Override checkpoint if provided
    if args.checkpoint is not None:
        config['checkpoint'] = args.checkpoint
    
    # Determine device
    device = 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(
        args.output_dir,
        config.get('experiment_name', f"{config['algorithm']}_{config['env_id']}")
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training or evaluation
    if args.eval_only:
        print("Running evaluation...")
        evaluate(config, output_dir, device, args.render)
    else:
        print("Running training...")
        train(config, output_dir, device)


if __name__ == '__main__':
    main()
