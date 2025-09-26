#!/usr/bin/env python3
"""
Evaluation and recording script for trained curiosity-driven RL agents.

This script provides a clean interface for:
- Loading trained models
- Running evaluation episodes
- Recording videos/GIFs of agent behavior
- Generating performance reports

Usage:
    python scripts/eval_and_record.py --env-id doom --model-path models/doom_ICM --record
    python scripts/eval_and_record.py --env-id mario --model-path models/mario_ICM --num-episodes 10
    python scripts/eval_and_record.py --env-id MontezumaRevenge-v5 --model-path models/atari_ICM --greedy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import tensorflow as tf
from PIL import Image

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from constants import get_env_config, update_constants_for_env
from envs import create_env
from logger import create_logger
from model import LSTMPolicy


class AgentEvaluator:
    """Agent evaluator with recording capabilities."""

    def __init__(
        self,
        env_id: str,
        model_path: str,
        output_dir: str = "evaluation_results",
        use_wandb: bool = False,
        experiment_name: Optional[str] = None,
    ):
        """Initialize evaluator.

        Args:
            env_id: Environment identifier
            model_path: Path to trained model
            output_dir: Directory for output files
            use_wandb: Whether to use Weights & Biases logging
            experiment_name: Name for the experiment
        """
        self.env_id = env_id
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Update constants for environment
        update_constants_for_env(env_id)

        # Create logger
        self.logger = create_logger(
            project_name="noreward-rl-eval",
            experiment_name=experiment_name or "eval-{env_id}-{int(time.time())}",
            use_wandb=use_wandb,
            use_tensorboard=False,
        )

        # Create environment
        self.env = create_env(env_id, client_id='0', remotes=None, env_wrap=True)
        self.num_actions = self.env.action_space.n

        # Load model
        self.policy = self._load_model()

        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_times: List[float] = []

    def _load_model(self) -> LSTMPolicy:
        """Load the trained model.

        Returns:
            Loaded policy network
        """
        print("Loading model from: {self.model_path}")

        # Create policy network
        with tf.variable_scope("global"):
            policy = LSTMPolicy(
                self.env.observation_space.shape,
                self.num_actions,
                design_head='universe'
            )
            policy.global_step = tf.get_variable(
                "global_step",
                [],
                tf.int32,
                initializer=tf.constant_initializer(0, dtype=tf.int32),
                trainable=False
            )

        # Load checkpoint
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, self.model_path)

        print("Model loaded successfully. Global step: {sess.run(policy.global_step)}")

        return policy

    def evaluate(
        self,
        num_episodes: int = 5,
        render: bool = False,
        record: bool = False,
        greedy: bool = False,
        random: bool = False,
        save_frames: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate the agent.

        Args:
            num_episodes: Number of episodes to run
            render: Whether to render the environment
            record: Whether to record videos
            greedy: Whether to use greedy policy (argmax)
            random: Whether to use random policy
            save_frames: Whether to save individual frames

        Returns:
            Dictionary with evaluation results
        """
        print("Starting evaluation: {num_episodes} episodes")
        print("Policy: {'greedy' if greedy else 'random' if random else 'sampled'}")

        # Setup recording if requested
        if record:
            self.env = gym.wrappers.RecordVideo(
                self.env,
                self.output_dir / "videos",
                episode_trigger=lambda x: True,
                name_prefix="{self.env_id}_episode"
            )

        # Setup frame saving if requested
        if save_frames:
            frames_dir = self.output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

        # Run episodes
        for episode in range(num_episodes):
            print("\nEpisode {episode + 1}/{num_episodes}")

            episode_reward, episode_length, episode_time = self._run_episode(
                render=render,
                greedy=greedy,
                random=random,
                save_frames=save_frames,
                episode_num=episode,
            )

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_times.append(episode_time)

            # Log episode results
            if self.logger:
                self.logger.log_episode_summary(
                    step=episode,
                    episode_reward=episode_reward,
                    episode_length=episode_length,
                    episode_time=episode_time,
                )

        # Compute statistics
        results = self._compute_statistics()

        # Save results
        self._save_results(results)

        # Print summary
        self._print_summary(results)

        return results

    def _run_episode(
        self,
        render: bool = False,
        greedy: bool = False,
        random: bool = False,
        save_frames: bool = False,
        episode_num: int = 0,
    ) -> Tuple[float, int, float]:
        """Run a single episode.

        Args:
            render: Whether to render
            greedy: Whether to use greedy policy
            random: Whether to use random policy
            save_frames: Whether to save frames
            episode_num: Episode number for saving

        Returns:
            Tuple of (reward, length, time)
        """
        obs, _ = self.env.reset()
        last_features = self.policy.get_initial_features()

        episode_reward = 0.0
        episode_length = 0
        episode_start_time = time.time()

        if save_frames:
            frames_dir = self.output_dir / "frames" / "episode_{episode_num:03d}"
            frames_dir.mkdir(exist_ok=True)
            frame_count = 0

        while True:
            # Get action
            if random:
                action = np.random.randint(0, self.num_actions)
            else:
                with tf.Session() as sess:
                    if greedy:
                        probs, _, features = self.policy.act_inference(obs, *last_features)
                        action = probs.argmax()
                    else:
                        _, action_one_hot, _, features = self.policy.act_inference(obs, *last_features)
                        action = action_one_hot.argmax()
                last_features = features

            # Take step
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            episode_length += 1

            # Render if requested
            if render:
                self.env.render()

            # Save frame if requested
            if save_frames:
                frame = self.env.render(mode='rgb_array')
                if frame is not None:
                    Image.fromarray(frame).save(frames_dir / "frame_{frame_count:06d}.png")
                    frame_count += 1

            # Check if episode is done
            if terminated or truncated:
                break

        episode_time = time.time() - episode_start_time

        print("  Reward: {episode_reward:.2f}, Length: {episode_length}, Time: {episode_time:.2f}s")
        if 'distance' in info:
            print("  Distance: {info['distance']}")

        return episode_reward, episode_length, episode_time

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute evaluation statistics.

        Returns:
            Dictionary with statistics
        """
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        times = np.array(self.episode_times)

        stats = {
            'num_episodes': len(rewards),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'mean_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths)),
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'rewards_per_second': float(np.mean(rewards / times)),
        }

        return stats

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results.

        Args:
            results: Results dictionary to save
        """
        # Save as numpy arrays
        np.save(self.output_dir / "episode_rewards.npy", np.array(self.episode_rewards))
        np.save(self.output_dir / "episode_lengths.npy", np.array(self.episode_lengths))
        np.save(self.output_dir / "episode_times.npy", np.array(self.episode_times))

        # Save statistics
        with open(self.output_dir / "results.txt", "w") as f:
            f.write("Evaluation Results for {self.env_id}\n")
            f.write("=" * 50 + "\n")
            for key, value in results.items():
                f.write("{key}: {value}\n")

        print("Results saved to: {self.output_dir}")

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation summary.

        Args:
            results: Results dictionary
        """
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY - {self.env_id.upper()}")
        print("=" * 50)
        print("Episodes: {results['num_episodes']}")
        print("Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print("Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
        print("Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        print("Mean Time: {results['mean_time']:.2f}s ± {results['std_time']:.2f}s")
        print("Rewards/Second: {results['rewards_per_second']:.2f}")
        print("=" * 50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained curiosity-driven RL agents")

    # Environment and model
    parser.add_argument("--env-id", required=True, help="Environment identifier")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")

    # Evaluation parameters
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--record", action="store_true", help="Record videos")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames")

    # Policy options
    parser.add_argument("--greedy", action="store_true", help="Use greedy policy (argmax)")
    parser.add_argument("--random", action="store_true", help="Use random policy")

    # Logging
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--experiment-name", help="Experiment name for logging")

    args = parser.parse_args()

    # Validate arguments
    if args.greedy and args.random:
        print("Error: Cannot use both greedy and random policies")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print("Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Create evaluator
    evaluator = AgentEvaluator(
        env_id=args.env_id,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name,
    )

    # Run evaluation
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        render=args.render,
        record=args.record,
        greedy=args.greedy,
        random=args.random,
        save_frames=args.save_frames,
    )

    # Cleanup
    evaluator.env.close()
    if evaluator.logger:
        evaluator.logger.finish()


if __name__ == "__main__":
    main()

