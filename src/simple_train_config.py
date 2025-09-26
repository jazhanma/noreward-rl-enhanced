#!/usr/bin/env python3
"""
Configuration-driven simple training script for NoReward-RL.

This script demonstrates how to use YAML configuration files
for training experiments with the refactored implementation.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
# Add project root to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import constants, get_env_config, update_constants_for_env
from envs import create_env
from logger import create_logger
from utils.seed import set_global_seed, set_gymnasium_seed, apply_deterministic_config
from utils.checkpoint import create_checkpoint_manager, find_latest_checkpoint
from utils.config import ConfigManager, load_config, create_config_from_args


def simple_training_loop_with_config(
    config: Dict[str, Any],
    num_episodes: Optional[int] = None
):
    """Simple training loop using configuration.

    Args:
        config: Configuration dictionary
        num_episodes: Override number of episodes (if None, uses config)
    """
    # Extract configuration values
    exp_config = config.get('experiment', {})
    training_config = config.get('training', {})
    env_config = config.get('environment', {})
    logging_config = config.get('logging', {})
    checkpoint_config = config.get('checkpointing', {})
    repro_config = config.get('reproducibility', {})

    env_id = env_config.get('env_id', 'CartPole-v1')
    seed = repro_config.get('seed')
    use_wandb = logging_config.get('use_wandb', False)
    save_dir = checkpoint_config.get('save_dir')
    save_interval = checkpoint_config.get('save_interval', 10)
    resume = checkpoint_config.get('resume', False)

    # Override episodes if specified
    if num_episodes is None:
        num_episodes = training_config.get('max_episodes', 10)

    print("🚀 Starting configuration-driven training on {env_id}")
    print("=" * 60)
    print("Experiment: {exp_config.get('name', 'unknown')}")
    print("Description: {exp_config.get('description', 'No description')}")
    print("Episodes: {num_episodes}")
    print("Use W&B: {use_wandb}")
    print("Seed: {seed if seed is not None else 'Random'}")
    print("Save Directory: {save_dir if save_dir else 'Disabled'}")
    print("=" * 60)

    # Set random seeds for reproducibility
    if seed is not None:
        print("🌱 Setting random seed: {seed}")
        set_global_seed(seed)
        apply_deterministic_config()
    else:
        print("🌱 Using random seeds (non-deterministic)")

    # Update constants for environment
    update_constants_for_env(env_id)
    env_config_dict = get_env_config(env_id)

    # Override constants with config values if available
    if 'learning_rate' in training_config:
        constants['LEARNING_RATE'] = training_config['learning_rate']
    if 'gamma' in training_config:
        constants['GAMMA'] = training_config['gamma']
    if 'entropy_beta' in training_config:
        constants['ENTROPY_BETA'] = training_config['entropy_beta']
    if 'prediction_beta' in training_config:
        constants['PREDICTION_BETA'] = training_config['prediction_beta']

    print("Environment config: {env_config_dict}")
    print("Global constants: GAMMA={constants['GAMMA']}, LR={constants['LEARNING_RATE']}")
    print("Entropy Beta: {constants['ENTROPY_BETA']}, Prediction Beta: {constants['PREDICTION_BETA']}")

    # Create logger
    experiment_name = logging_config.get('experiment_name', "config-{env_id}-{int(time.time())}")
    if '{timestamp}' in experiment_name:
        experiment_name = experiment_name.replace('{timestamp}', str(int(time.time())))

    logger = create_logger(
        project_name="noreward-rl-config",
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        use_tensorboard=logging_config.get('use_tensorboard', False),
    )

    # Initialize checkpoint manager
    checkpoint_manager = None
    start_episode = 0
    if save_dir is not None:
        checkpoint_manager = create_checkpoint_manager(
            save_dir=save_dir,
            save_interval=save_interval,
            checkpoint_prefix="config_{env_id}"
        )

        # Resume from checkpoint if requested
        if resume:
            latest_checkpoint = find_latest_checkpoint(save_dir)
            if latest_checkpoint:
                print("🔄 Resuming from checkpoint: {latest_checkpoint.name}")
                # For simple training, we'll just track the episode number
                # In a real implementation, you'd load the actual model state
                start_episode = 0  # This would be loaded from checkpoint metadata
                print("   Starting from episode: {start_episode}")
            else:
                print("⚠️  No checkpoint found, starting from beginning")

    # Create environment
    print("\nCreating environment: {env_id}")
    env = create_env(env_id, client_id='0', remotes=None, env_wrap=env_config.get('env_wrap', True))

    # Seed the environment if seed is provided
    if seed is not None:
        env, actual_seed = set_gymnasium_seed(env, seed)
        if actual_seed is not None:
            print("🌱 Environment seeded with: {actual_seed}")
        else:
            print("⚠️  Warning: Could not seed environment")

    print("Environment created: {env}")
    print("Observation space: {env.observation_space}")
    print("Action space: {env.action_space}")

    # Simple training loop
    print("\nStarting training for {num_episodes} episodes...")
    if start_episode > 0:
        print("Resuming from episode {start_episode}")

    episode_rewards = []
    episode_lengths = []

    # Get logging intervals from config
    log_interval = logging_config.get('log_interval', 10)

    for episode in range(start_episode, start_episode + num_episodes):
        if episode % log_interval == 0 or episode == start_episode:
            print("\nEpisode {episode + 1}/{num_episodes}")

        # Reset environment
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_start_time = time.time()

        # Run episode
        while True:
            # Random action (for demonstration)
            action = env.action_space.sample()

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Check if episode is done
            if terminated or truncated:
                break

        episode_time = time.time() - episode_start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Log episode results
        if episode % log_interval == 0 or episode == start_episode:
            print("  Reward: {episode_reward:.2f}, Length: {episode_length}, Time: {episode_time:.2f}s")

        if logger:
            logger.log_episode_summary(
                step=episode,
                episode_reward=episode_reward,
                episode_length=episode_length,
                episode_time=episode_time,
            )

        # Save checkpoint if needed
        if checkpoint_manager and checkpoint_manager.should_save_checkpoint(episode, episode):
            metrics = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'episode_time': episode_time,
                'mean_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
                'mean_length': sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
            }

            checkpoint_config_dict = {
                'env_id': env_id,
                'seed': seed,
                'env_config': env_config_dict,
                'constants': dict(constants),
                'experiment_config': config
            }

            # For simple training, we create a dummy model state
            # In a real implementation, you'd save the actual model
            dummy_model = {'episode': episode, 'rewards': episode_rewards}

            checkpoint_manager.save_checkpoint(
                model=dummy_model,
                step=episode,
                episode=episode,
                metrics=metrics,
                config=checkpoint_config_dict,
                is_best=episode_reward == max(episode_rewards) if episode_rewards else False
            )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print("Experiment: {exp_config.get('name', 'unknown')}")
    print("Environment: {env_id}")
    print("Episodes: {num_episodes}")
    print("Mean Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print("Mean Length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print("Best Reward: {max(episode_rewards):.2f}")
    print("Worst Reward: {min(episode_rewards):.2f}")

    # Check against performance targets if available
    targets = config.get('targets', {})
    if targets:
        print("\nPerformance Targets:")
        solved_reward = targets.get('solved_reward')
        if solved_reward and max(episode_rewards) >= solved_reward:
            print("✅ Solved! (Target: {solved_reward}, Best: {max(episode_rewards):.2f})")
        elif solved_reward:
            print("❌ Not solved (Target: {solved_reward}, Best: {max(episode_rewards):.2f})")

    # Cleanup
    env.close()
    if logger:
        logger.finish()

    print("\n✅ Configuration-driven training completed successfully!")
    return episode_rewards, episode_lengths


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Configuration-driven simple training script for NoReward-RL")

    # Configuration options
    parser.add_argument("--config", type=str, help="Path to experiment configuration file")
    parser.add_argument("--experiment", type=str, help="Name of experiment config (e.g., 'curiosity_baseline')")
    parser.add_argument("--environment", type=str, help="Name of environment config (e.g., 'cartpole')")

    # Override options
    parser.add_argument("--env-id", type=str, help="Override environment ID")
    parser.add_argument("--num-episodes", type=int, help="Override number of episodes")
    parser.add_argument("--use-wandb", action="store_true", help="Override: Use Weights & Biases logging")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--save-dir", type=str, help="Override checkpoint save directory")

    # Legacy mode (for backward compatibility)
    parser.add_argument("--legacy", action="store_true", help="Use legacy argument-based configuration")

    args = parser.parse_args()

    print("🎯 NoReward-RL Configuration-Driven Training")
    print("=" * 60)

    try:
        # Load configuration
        if args.legacy or not (args.config or args.experiment):
            print("📝 Using legacy argument-based configuration")
            config = create_config_from_args(args)
        else:
            if args.config:
                print("📁 Loading configuration from file: {args.config}")
                config_manager = ConfigManager()
                with open(args.config, 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
            else:
                print("📁 Loading experiment config: {args.experiment}")
                config = load_config(args.experiment, args.environment)

        # Apply overrides
        if args.env_id:
            config['environment']['env_id'] = args.env_id
            print("🔄 Override: Environment ID = {args.env_id}")

        if args.num_episodes:
            config['training']['max_episodes'] = args.num_episodes
            print("🔄 Override: Episodes = {args.num_episodes}")

        if args.use_wandb:
            config['logging']['use_wandb'] = True
            print("🔄 Override: Use W&B = True")

        if args.seed is not None:
            config['reproducibility']['seed'] = args.seed
            print("🔄 Override: Seed = {args.seed}")

        if args.save_dir:
            config['checkpointing']['save_dir'] = args.save_dir
            print("🔄 Override: Save Directory = {args.save_dir}")

        # Run training
        episode_rewards, episode_lengths = simple_training_loop_with_config(
            config=config,
            num_episodes=args.num_episodes
        )

        print("\n🎉 Training completed! Final mean reward: {sum(episode_rewards) / len(episode_rewards):.2f}")

    except Exception as e:
        print("\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
