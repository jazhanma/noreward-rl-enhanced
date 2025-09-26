#!/usr/bin/env python3
"""
Simple training script for testing the refactored NoReward-RL implementation.

This script provides a basic training loop that works with the current setup
without requiring complex TensorFlow 1.x compatibility.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
# Add project root to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import constants, get_env_config, update_constants_for_env
from envs import create_env
from logger import create_logger
from utils.seed import set_global_seed, set_gymnasium_seed, apply_deterministic_config
from utils.checkpoint import create_checkpoint_manager, find_latest_checkpoint


def simple_training_loop(
    env_id: str,
    num_episodes: int = 10,
    use_wandb: bool = False,
    seed: int = None,
    save_dir: Optional[str] = None,
    save_interval: int = 10,
    resume: bool = False
):
    """Simple training loop for demonstration.

    Args:
        env_id: Environment identifier
        num_episodes: Number of episodes to run
        use_wandb: Whether to use Weights & Biases logging
        seed: Random seed for reproducibility (None for random)
        save_dir: Directory to save checkpoints (None to disable)
        save_interval: Save checkpoint every N episodes
        resume: Whether to resume from latest checkpoint
    """
    print("🚀 Starting simple training on {env_id}")
    print("=" * 50)

    # Set random seeds for reproducibility
    if seed is not None:
        print("🌱 Setting random seed: {seed}")
        set_global_seed(seed)
        apply_deterministic_config()
    else:
        print("🌱 Using random seeds (non-deterministic)")

    # Update constants for environment
    update_constants_for_env(env_id)
    env_config = get_env_config(env_id)

    print("Environment config: {env_config}")
    print("Global constants: GAMMA={constants['GAMMA']}, LR={constants['LEARNING_RATE']}")

    # Create logger
    logger = create_logger(
        project_name="noreward-rl-simple",
        experiment_name="simple-{env_id}-{int(time.time())}",
        use_wandb=use_wandb,
        use_tensorboard=False,
    )

    # Initialize checkpoint manager
    checkpoint_manager = None
    start_episode = 0
    if save_dir is not None:
        checkpoint_manager = create_checkpoint_manager(
            save_dir=save_dir,
            save_interval=save_interval,
            checkpoint_prefix="simple_{env_id}"
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
    env = create_env(env_id, client_id='0', remotes=None, env_wrap=True)

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

    for episode in range(start_episode, start_episode + num_episodes):
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

            config = {
                'env_id': env_id,
                'seed': seed,
                'env_config': env_config,
                'constants': dict(constants)
            }

            # For simple training, we create a dummy model state
            # In a real implementation, you'd save the actual model
            dummy_model = {'episode': episode, 'rewards': episode_rewards}

            checkpoint_manager.save_checkpoint(
                model=dummy_model,
                step=episode,
                episode=episode,
                metrics=metrics,
                config=config,
                is_best=episode_reward == max(episode_rewards) if episode_rewards else False
            )

    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print("Environment: {env_id}")
    print("Episodes: {num_episodes}")
    print("Mean Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print("Mean Length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print("Best Reward: {max(episode_rewards):.2f}")
    print("Worst Reward: {min(episode_rewards):.2f}")

    # Cleanup
    env.close()
    if logger:
        logger.finish()

    print("\n✅ Training completed successfully!")
    return episode_rewards, episode_lengths


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple training script for NoReward-RL")

    parser.add_argument("--env-id", required=True, help="Environment identifier")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (None for random)")

    # Checkpointing arguments
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save checkpoints (None to disable)")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N episodes")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    print("🎯 NoReward-RL Simple Training")
    print("=" * 50)
    print("Environment: {args.env_id}")
    print("Episodes: {args.num_episodes}")
    print("Use W&B: {args.use_wandb}")
    print("Seed: {args.seed if args.seed is not None else 'Random'}")
    print("Save Directory: {args.save_dir if args.save_dir else 'Disabled'}")
    print("Save Interval: {args.save_interval}")
    print("Resume: {args.resume}")
    print("=" * 50)

    try:
        episode_rewards, episode_lengths = simple_training_loop(
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            use_wandb=args.use_wandb,
            seed=args.seed,
            save_dir=args.save_dir,
            save_interval=args.save_interval,
            resume=args.resume,
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

