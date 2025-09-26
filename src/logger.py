"""
Logging utilities for curiosity-driven RL training.

Supports both Weights & Biases and TensorBoard logging with configurable options.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from constants import LOGGING_CONFIG


class Logger:
    """Unified logger for both Weights & Biases and TensorBoard."""

    def __init__(
        self,
        project_name: str = "noreward-rl",
        experiment_name: Optional[str] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        log_dir: str = "logs",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize logger.

        Args:
            project_name: Name of the project for logging
            experiment_name: Name of the current experiment
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            log_dir: Directory for TensorBoard logs
            config: Configuration dictionary to log
        """
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.config = config or {}

        # Initialize Weights & Biases
        if self.use_wandb:
            self._init_wandb(project_name, experiment_name)

        # Initialize TensorBoard
        if self.use_tensorboard:
            self._init_tensorboard()

    def _init_wandb(self, project_name: str, experiment_name: Optional[str]) -> None:
        """Initialize Weights & Biases."""
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available, skipping Weights & Biases logging")
            self.use_wandb = False
            return

        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = "curiosity-{int(time.time())}"

        # Initialize wandb
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=self.config,
            reinit=True,
        )

        print("Weights & Biases initialized: {project_name}/{experiment_name}")

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard writer."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_writer = tf.summary.create_file_writer(self.log_dir)
        print("TensorBoard logging to: {self.log_dir}")

    def log_scalar(self, key: str, value: Union[float, int], step: Optional[int] = None) -> None:
        """Log a scalar value.

        Args:
            key: Key for the metric
            value: Value to log
            step: Step number (optional)
        """
        if self.use_wandb:
            wandb.log({key: value}, step=step)

        if self.use_tensorboard:
            with self.tb_writer.as_default():
                tf.summary.scalar(key, value, step=step)

    def log_scalars(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log multiple scalar values.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        if self.use_wandb:
            wandb.log(metrics, step=step)

        if self.use_tensorboard:
            with self.tb_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(key, value, step=step)

    def log_image(self, key: str, image: np.ndarray, step: Optional[int] = None) -> None:
        """Log an image.

        Args:
            key: Key for the image
            image: Image array (H, W, C) or (H, W)
            step: Step number (optional)
        """
        if self.use_wandb:
            wandb.log({key: wandb.Image(image)}, step=step)

        if self.use_tensorboard:
            with self.tb_writer.as_default():
                tf.summary.image(key, image[np.newaxis, ...], step=step)

    def log_video(self, key: str, video: np.ndarray, fps: int = 4, step: Optional[int] = None) -> None:
        """Log a video.

        Args:
            key: Key for the video
            video: Video array (T, H, W, C)
            fps: Frames per second
            step: Step number (optional)
        """
        if self.use_wandb:
            wandb.log({key: wandb.Video(video, fps=fps)}, step=step)

        if self.use_tensorboard:
            with self.tb_writer.as_default():
                tf.summary.image(key, video, step=step)

    def log_histogram(self, key: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log a histogram.

        Args:
            key: Key for the histogram
            values: Values to create histogram from
            step: Step number (optional)
        """
        if self.use_wandb:
            wandb.log({key: wandb.Histogram(values)}, step=step)

        if self.use_tensorboard:
            with self.tb_writer.as_default():
                tf.summary.histogram(key, values, step=step)

    def log_curiosity_metrics(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        extrinsic_reward: float,
        intrinsic_reward: float,
        episode_length: int,
        prediction_loss: Optional[float] = None,
        inverse_loss: Optional[float] = None,
        forward_loss: Optional[float] = None,
    ) -> None:
        """Log curiosity-specific metrics.

        Args:
            step: Current step number
            policy_loss: Policy network loss
            value_loss: Value network loss
            entropy: Policy entropy
            extrinsic_reward: External environment reward
            intrinsic_reward: Intrinsic curiosity reward
            episode_length: Length of current episode
            prediction_loss: Total prediction loss (optional)
            inverse_loss: Inverse model loss (optional)
            forward_loss: Forward model loss (optional)
        """
        metrics = {
            "loss/policy": policy_loss,
            "loss/value": value_loss,
            "loss/entropy": entropy,
            "reward/extrinsic": extrinsic_reward,
            "reward/intrinsic": intrinsic_reward,
            "reward/total": extrinsic_reward + intrinsic_reward,
            "episode/length": episode_length,
        }

        if prediction_loss is not None:
            metrics["loss/prediction"] = prediction_loss
        if inverse_loss is not None:
            metrics["loss/inverse"] = inverse_loss
        if forward_loss is not None:
            metrics["loss/forward"] = forward_loss

        self.log_scalars(metrics, step=step)

    def log_episode_summary(
        self,
        step: int,
        episode_reward: float,
        episode_length: int,
        episode_time: float,
        distance: Optional[float] = None,
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
    ) -> None:
        """Log episode summary.

        Args:
            step: Current step number
            episode_reward: Total episode reward
            episode_length: Episode length
            episode_time: Episode duration
            distance: Distance covered (Mario)
            position_x: X position (Doom)
            position_y: Y position (Doom)
        """
        metrics = {
            "episode/reward": episode_reward,
            "episode/length": episode_length,
            "episode/time": episode_time,
            "episode/reward_per_time": episode_reward / episode_time if episode_time > 0 else 0,
        }

        if distance is not None:
            metrics["episode/distance"] = distance
        if position_x is not None:
            metrics["episode/position_x"] = position_x
        if position_y is not None:
            metrics["episode/position_y"] = position_y

        self.log_scalars(metrics, step=step)

    def watch_model(self, model: tf.keras.Model) -> None:
        """Watch model for gradients and parameters.

        Args:
            model: TensorFlow model to watch
        """
        if self.use_wandb:
            wandb.watch(model, log="all", log_freq=100)

    def save_model(self, model: tf.keras.Model, path: str) -> None:
        """Save model checkpoint.

        Args:
            model: Model to save
            path: Path to save the model
        """
        if self.use_wandb:
            wandb.save(path)

        # Also save locally
        model.save(path)

    def finish(self) -> None:
        """Finish logging and cleanup."""
        if self.use_wandb:
            wandb.finish()

        if self.use_tensorboard:
            self.tb_writer.close()


def create_logger(
    project_name: str = "noreward-rl",
    experiment_name: Optional[str] = None,
    use_wandb: Optional[bool] = None,
    use_tensorboard: Optional[bool] = None,
    log_dir: str = "logs",
    config: Optional[Dict[str, Any]] = None,
) -> Logger:
    """Create a logger instance with default configuration.

    Args:
        project_name: Name of the project for logging
        experiment_name: Name of the current experiment
        use_wandb: Whether to use Weights & Biases (defaults to config)
        use_tensorboard: Whether to use TensorBoard (defaults to config)
        log_dir: Directory for TensorBoard logs
        config: Configuration dictionary to log

    Returns:
        Configured logger instance
    """
    if use_wandb is None:
        use_wandb = LOGGING_CONFIG.get("USE_WANDB", True)
    if use_tensorboard is None:
        use_tensorboard = LOGGING_CONFIG.get("USE_TENSORBOARD", True)

    return Logger(
        project_name=project_name,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        log_dir=log_dir,
        config=config,
    )

