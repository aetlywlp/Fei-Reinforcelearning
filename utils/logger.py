"""
Logging utilities for reinforcement learning.

This module provides logging utilities for tracking and visualizing
training progress in reinforcement learning experiments.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    """Generic logger for reinforcement learning experiments.
    
    This class provides methods for logging metrics, saving checkpoints,
    and visualizing training progress.
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        log_frequency: int = 1,
        verbose: bool = True
    ):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard for logging
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity name
            log_frequency: Frequency of logging (in steps)
            verbose: Whether to print logs to console
        """
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.log_frequency = log_frequency
        self.verbose = verbose
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logging
        log_file = os.path.join(self.log_dir, 'log.txt')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        
        # Initialize tensorboard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                self.logger.warning("TensorBoard not available. Disabling TensorBoard logging.")
                self.use_tensorboard = False
        
        # Initialize wandb
        if self.use_wandb:
            if wandb is None:
                self.logger.warning("Weights & Biases not available. Disabling W&B logging.")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    dir=self.log_dir
                )
        
        # Initialize metrics
        self.metrics = {}
        self.smoothed_metrics = {}
        self.step = 0
        self.start_time = time.time()
        
        # Log initial info
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Logs saved to: {self.log_dir}")
    
    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        # Save hyperparameters to file
        hp_file = os.path.join(self.log_dir, 'hyperparams.json')
        with open(hp_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        
        # Log hyperparameters to tensorboard
        if self.use_tensorboard:
            self.tb_writer.add_hparams(hyperparams, {})
        
        # Log hyperparameters to wandb
        if self.use_wandb:
            wandb.config.update(hyperparams)
            
        # Log hyperparameters to console
        if self.verbose:
            self.logger.info("Hyperparameters:")
            for key, value in hyperparams.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step (if None, use internal counter)
        """
        if step is not None:
            self.step = step
        
        # Update metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.smoothed_metrics[key] = 0.0
            
            self.metrics[key].append(value)
            
            # Update smoothed metrics with exponential moving average
            alpha = 0.1
            self.smoothed_metrics[key] = alpha * value + (1 - alpha) * self.smoothed_metrics[key]
        
        # Log metrics to tensorboard
        if self.use_tensorboard and self.step % self.log_frequency == 0:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.step)
                self.tb_writer.add_scalar(f"{key}_smoothed", self.smoothed_metrics[key], self.step)
        
        # Log metrics to wandb
        if self.use_wandb and self.step % self.log_frequency == 0:
            wandb_metrics = {**metrics}
            for key, value in self.smoothed_metrics.items():
                wandb_metrics[f"{key}_smoothed"] = value
            wandb_metrics['step'] = self.step
            wandb.log(wandb_metrics)
            
        # Log metrics to console
        if self.verbose and self.step % self.log_frequency == 0:
            time_elapsed = time.time() - self.start_time
            self.logger.info(f"Step {self.step}, Time: {time_elapsed:.2f}s")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value:.4f} (smoothed: {self.smoothed_metrics[key]:.4f})")
        
        # Increment step
        self.step += 1
    
    def log_episode(self, episode_metrics: Dict[str, float], episode: int) -> None:
        """Log episode metrics.
        
        Args:
            episode_metrics: Dictionary of episode metrics
            episode: Episode number
        """
        # Add episode number to metrics
        metrics = {**episode_metrics, 'episode': episode}
        
        # Log metrics
        self.log_metrics(metrics)
    
    def save_checkpoint(self, state: Dict[str, Any], filename: str = 'checkpoint.pt') -> None:
        """Save a checkpoint.
        
        Args:
            state: Dictionary of state variables to save
            filename: Filename for the checkpoint
        """
        checkpoint_path = os.path.join(self.log_dir, filename)
        torch.save(state, checkpoint_path)
        
        if self.verbose:
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pt') -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            filename: Filename of the checkpoint
            
        Returns:
            state: Dictionary of loaded state variables
        """
        checkpoint_path = os.path.join(self.log_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} does not exist")
            return None
        
        state = torch.load(checkpoint_path, map_location='cpu')
        
        if self.verbose:
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return state
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple) -> None:
        """Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (without batch dimension)
        """
        if not self.use_tensorboard:
            return
        
        try:
            # Create dummy input
            dummy_input = torch.zeros(1, *input_shape, device=next(model.parameters()).device)
            
            # Log model graph
            self.tb_writer.add_graph(model, dummy_input)
            
            if self.verbose:
                self.logger.info(f"Model graph logged to TensorBoard")
        except Exception as e:
            self.logger.warning(f"Failed to log model graph: {e}")
    
    def log_image(self, tag: str, image_tensor: torch.Tensor, step: Optional[int] = None) -> None:
        """Log an image to TensorBoard and W&B.
        
        Args:
            tag: Tag for the image
            image_tensor: Image tensor (BCHW or CHW format)
            step: Current step (if None, use internal counter)
        """
        if step is not None:
            self.step = step
        
        # Add batch dimension if missing
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Log image to tensorboard
        if self.use_tensorboard:
            self.tb_writer.add_images(tag, image_tensor, self.step)
        
        # Log image to wandb
        if self.use_wandb:
            wandb.log({tag: [wandb.Image(img) for img in image_tensor]}, step=self.step)
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None) -> None:
        """Log a histogram to TensorBoard.
        
        Args:
            tag: Tag for the histogram
            values: Values for the histogram
            step: Current step (if None, use internal counter)
        """
        if step is not None:
            self.step = step
        
        # Log histogram to tensorboard
        if self.use_tensorboard:
            self.tb_writer.add_histogram(tag, values, self.step)
    
    def log_video(self, tag: str, video_tensor: torch.Tensor, fps: int = 30, step: Optional[int] = None) -> None:
        """Log a video to TensorBoard and W&B.
        
        Args:
            tag: Tag for the video
            video_tensor: Video tensor (BCTHW format)
            fps: Frames per second
            step: Current step (if None, use internal counter)
        """
        if step is not None:
            self.step = step
        
        # Log video to tensorboard
        if self.use_tensorboard:
            self.tb_writer.add_video(tag, video_tensor, self.step, fps=fps)
        
        # Log video to wandb
        if self.use_wandb:
            wandb.log({tag: wandb.Video(video_tensor, fps=fps, format="mp4")}, step=self.step)
    
    def close(self) -> None:
        """Close the logger."""
        # Save metrics to file
        metrics_file = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Close tensorboard writer
        if self.use_tensorboard:
            self.tb_writer.close()
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
        
        if self.verbose:
            self.logger.info("Logger closed")


class ConsoleLogger:
    """Simple console logger for reinforcement learning experiments.
    
    This class provides a simplified interface for logging to console,
    without dependencies on TensorBoard or W&B.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_frequency: int = 1
    ):
        """Initialize the console logger.
        
        Args:
            experiment_name: Name of the experiment
            log_frequency: Frequency of logging (in steps)
        """
        self.experiment_name = experiment_name
        self.log_frequency = log_frequency
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        
        # Initialize metrics
        self.metrics = {}
        self.smoothed_metrics = {}
        self.step = 0
        self.start_time = time.time()
        
        # Log initial info
        self.logger.info(f"Starting experiment: {experiment_name}")
    
    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step (if None, use internal counter)
        """
        if step is not None:
            self.step = step
        
        # Update metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.smoothed_metrics[key] = 0.0
            
            self.metrics[key].append(value)
            
            # Update smoothed metrics with exponential moving average
            alpha = 0.1
            self.smoothed_metrics[key] = alpha * value + (1 - alpha) * self.smoothed_metrics[key]
            
        # Log metrics to console
        if self.step % self.log_frequency == 0:
            time_elapsed = time.time() - self.start_time
            self.logger.info(f"Step {self.step}, Time: {time_elapsed:.2f}s")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value:.4f} (smoothed: {self.smoothed_metrics[key]:.4f})")
        
        # Increment step
        self.step += 1
    
    def log_episode(self, episode_metrics: Dict[str, float], episode: int) -> None:
        """Log episode metrics.
        
        Args:
            episode_metrics: Dictionary of episode metrics
            episode: Episode number
        """
        # Add episode number to metrics
        metrics = {**episode_metrics, 'episode': episode}
        
        # Log metrics
        self.log_metrics(metrics)
    
    def close(self) -> None:
        """Close the logger."""
        self.logger.info("Experiment completed")
