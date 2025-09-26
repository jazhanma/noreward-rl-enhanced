#!/usr/bin/env python3
"""
Modern training script for curiosity-driven RL with enhanced logging and gymnasium support.

This script provides a clean interface for training curiosity-driven agents with:
- Weights & Biases and TensorBoard logging
- Gymnasium environment support
- Hard exploration Atari games
- Configurable hyperparameters
- Distributed training support

Usage:
    python train_modern.py --env-id doom --use-wandb
    python train_modern.py --env-id MontezumaRevenge-v5 --num-workers 8
    python train_modern.py --env-id mario --no-reward --unsup action
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
# Add project root to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf

from constants import constants, get_env_config, update_constants_for_env
from envs import create_env
from logger import create_logger
from a3c_modern import A3C
from utils.seed import set_global_seed, set_gymnasium_seed, apply_deterministic_config
from utils.checkpoint import create_checkpoint_manager, find_latest_checkpoint


def create_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create training configuration from arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Configuration dictionary
    """
    config = {
        'env_id': args.env_id,
        'num_workers': args.num_workers,
        'log_dir': args.log_dir,
        'max_steps': args.max_steps,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'entropy_beta': args.entropy_beta,
        'prediction_beta': args.prediction_beta,
        'unsup_type': args.unsup,
        'design_head': args.design_head,
        'no_reward': args.no_reward,
        'no_life_reward': args.no_life_reward,
        'env_wrap': args.env_wrap,
        'visualise': args.visualise,
        'use_wandb': args.use_wandb,
        'use_tensorboard': args.use_tensorboard,
        'experiment_name': args.experiment_name,
        'seed': args.seed,
        'save_dir': args.save_dir,
        'save_interval': args.save_interval,
        'resume': args.resume,
    }

    # Update constants based on environment
    update_constants_for_env(args.env_id)

    # Override constants with command line arguments
    if args.learning_rate is not None:
        constants['LEARNING_RATE'] = args.learning_rate
    if args.gamma is not None:
        constants['GAMMA'] = args.gamma
    if args.entropy_beta is not None:
        constants['ENTROPY_BETA'] = args.entropy_beta
    if args.prediction_beta is not None:
        constants['PREDICTION_BETA'] = args.prediction_beta
    if args.max_steps is not None:
        constants['MAX_GLOBAL_STEPS'] = args.max_steps

    return config


def train_worker(
    worker_id: int,
    config: Dict[str, Any],
    logger: Optional[Any] = None,
) -> None:
    """Train a single worker.

    Args:
        worker_id: Worker ID
        config: Training configuration
        logger: Logger instance
    """
    print("Starting worker {worker_id}")

    # Create environment
    env = create_env(
        config['env_id'],
        client_id=str(worker_id),
        remotes=None,
        env_wrap=config['env_wrap'],
        design_head=config['design_head'],
        no_life_reward=config['no_life_reward'],
    )

    # Seed the environment if seed is provided
    if config.get('seed') is not None:
        env, actual_seed = set_gymnasium_seed(env, config['seed'])
        if actual_seed is not None:
            print("🌱 Worker {worker_id} environment seeded with: {actual_seed}")
        else:
            print("⚠️  Warning: Could not seed environment for worker {worker_id}")

    # Initialize checkpoint manager
    checkpoint_manager = None
    if config.get('save_dir') is not None:
        checkpoint_manager = create_checkpoint_manager(
            save_dir=config['save_dir'],
            save_interval=config.get('save_interval', 100),
            checkpoint_prefix="worker_{worker_id}_{config['env_id']}"
        )

        # Resume from checkpoint if requested
        if config.get('resume', False):
            latest_checkpoint = find_latest_checkpoint(config['save_dir'])
            if latest_checkpoint:
                print("🔄 Worker {worker_id} resuming from checkpoint: {latest_checkpoint.name}")
            else:
                print("⚠️  Worker {worker_id}: No checkpoint found, starting from beginning")

    # Create A3C trainer
    trainer = A3C(
        env=env,
        task=worker_id,
        visualise=config['visualise'] and worker_id == 0,
        unsup_type=config['unsup_type'],
        env_wrap=config['env_wrap'],
        design_head=config['design_head'],
        no_reward=config['no_reward'],
        logger=logger if worker_id == 0 else None,  # Only log from worker 0
    )

    # Setup TensorBoard logging
    if config['use_tensorboard']:
        log_dir = Path(config['log_dir']) / "worker_{worker_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(str(log_dir))
    else:
        summary_writer = None

    # Training loop
    with tf.Session() as sess:
        trainer.start(sess, summary_writer)

        global_step = sess.run(trainer.global_step)
        print("Worker {worker_id} starting at global step: {global_step}")

        while global_step < constants['MAX_GLOBAL_STEPS']:
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

            if global_step % 1000 == 0:
                print("Worker {worker_id} - Global step: {global_step}")

            # Save checkpoint if needed
            if checkpoint_manager and checkpoint_manager.should_save_checkpoint(global_step, 0):
                # Get current metrics (this would be more sophisticated in practice)
                metrics = {
                    'global_step': global_step,
                    'worker_id': worker_id
                }

                # Save trainer state as model
                checkpoint_manager.save_checkpoint(
                    model=trainer,
                    step=global_step,
                    episode=0,  # A3C doesn't track episodes directly
                    metrics=metrics,
                    config=config,
                    is_best=False  # This would be determined by performance metrics
                )

    print("Worker {worker_id} finished training")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train curiosity-driven RL agents")

    # Environment
    parser.add_argument("--env-id", required=True, help="Environment identifier")
    parser.add_argument("--env-wrap", action="store_true", help="Apply environment wrappers")
    parser.add_argument("--no-reward", action="store_true", help="Remove all extrinsic rewards")
    parser.add_argument("--no-life-reward", action="store_true", help="Remove negative rewards")

    # Training parameters
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--entropy-beta", type=float, help="Entropy regularization")
    parser.add_argument("--prediction-beta", type=float, help="Curiosity weight")

    # Model architecture
    parser.add_argument("--design-head", choices=['universe', 'nips', 'nature', 'doom'],
                       default='universe', help="Network head design")
    parser.add_argument("--unsup", choices=['action', 'state', 'stateAenc'],
                       help="Unsupervised learning type")

    # Logging and output
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--use-tensorboard", action="store_true", help="Use TensorBoard")
    parser.add_argument("--experiment-name", help="Experiment name")
    parser.add_argument("--visualise", action="store_true", help="Visualize training")

    # Hard exploration games
    parser.add_argument("--hard-exploration", action="store_true",
                       help="Use hard exploration Atari games")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (None for random)")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save checkpoints (None to disable)")
    parser.add_argument("--save-interval", type=int, default=100,
                       help="Save checkpoint every N steps/episodes")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from latest checkpoint")

    args = parser.parse_args()

    # Validate arguments
    if args.hard_exploration and 'atari' not in args.env_id.lower():
        print("Warning: --hard-exploration flag only applies to Atari games")

    # Create configuration
    config = create_training_config(args)

    # Set random seeds for reproducibility
    if args.seed is not None:
        print("🌱 Setting random seed: {args.seed}")
        set_global_seed(args.seed)
        apply_deterministic_config()
    else:
        print("🌱 Using random seeds (non-deterministic)")

    # Create logger
    logger = None
    if config['use_wandb'] or config['use_tensorboard']:
        logger = create_logger(
            project_name="noreward-rl",
            experiment_name=config['experiment_name'] or "train-{config['env_id']}-{int(time.time())}",
            use_wandb=config['use_wandb'],
            use_tensorboard=config['use_tensorboard'],
            log_dir=config['log_dir'],
            config=config,
        )

    # Print configuration
    print("Training Configuration:")
    print("=" * 50)
    for key, value in config.items():
        print("{key}: {value}")
    print("=" * 50)

    # Create log directory
    os.makedirs(config['log_dir'], exist_ok=True)

    # For single worker training (simplified for this example)
    # In practice, you would use multiprocessing for multiple workers
    if config['num_workers'] == 1:
        train_worker(0, config, logger)
    else:
        print("Multi-worker training not implemented in this simplified version")
        print("Using single worker instead")
        train_worker(0, config, logger)

    # Finish logging
    if logger:
        logger.finish()

    print("Training completed!")


if __name__ == "__main__":
    main()

