#!/usr/bin/env python3
"""
Benchmark script for hard exploration Atari games.

This script provides a standardized way to evaluate curiosity-driven agents
on hard exploration Atari games like Montezuma's Revenge, Pitfall, etc.

Usage:
    python scripts/benchmark_hard_exploration.py --env-id MontezumaRevenge-v5 --model-path models/montezuma_ICM
    python scripts/benchmark_hard_exploration.py --env-id Pitfall-v5 --model-path models/pitfall_ICM --num-episodes 20
    python scripts/benchmark_hard_exploration.py --all-games --model-path models/atari_ICM
"""
from __future__ import python3

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from constants import HARD_EXPLORATION_GAMES
from scripts.eval_and_record import AgentEvaluator


class HardExplorationBenchmark:
    """Benchmark for hard exploration Atari games."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "benchmark_results",
        use_wandb: bool = False,
    ):
        """Initialize benchmark.
        
        Args:
            model_path: Path to trained model
            output_dir: Directory for benchmark results
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def benchmark_game(
        self,
        env_id: str,
        num_episodes: int = 10,
        render: bool = False,
        record: bool = False,
    ) -> Dict[str, Any]:
        """Benchmark a single game.
        
        Args:
            env_id: Environment identifier
            num_episodes: Number of episodes to run
            render: Whether to render
            record: Whether to record videos
            
        Returns:
            Benchmark results for the game
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {env_id.upper()}")
        print(f"{'='*60}")
        
        # Create evaluator
        evaluator = AgentEvaluator(
            env_id=env_id,
            model_path=self.model_path,
            output_dir=str(self.output_dir / env_id),
            use_wandb=self.use_wandb,
            experiment_name=f"benchmark-{env_id}",
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            num_episodes=num_episodes,
            render=render,
            record=record,
            greedy=True,  # Use greedy policy for benchmarking
        )
        
        # Store results
        self.results[env_id] = results
        
        # Cleanup
        evaluator.env.close()
        if evaluator.logger:
            evaluator.logger.finish()
        
        return results
    
    def benchmark_all_games(
        self,
        num_episodes: int = 10,
        render: bool = False,
        record: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark all hard exploration games.
        
        Args:
            num_episodes: Number of episodes per game
            render: Whether to render
            record: Whether to record videos
            
        Returns:
            Results for all games
        """
        print(f"BENCHMARKING ALL HARD EXPLORATION GAMES")
        print(f"Games: {len(HARD_EXPLORATION_GAMES)}")
        print(f"Episodes per game: {num_episodes}")
        print(f"Total episodes: {len(HARD_EXPLORATION_GAMES) * num_episodes}")
        
        for i, game in enumerate(HARD_EXPLORATION_GAMES):
            print(f"\nProgress: {i+1}/{len(HARD_EXPLORATION_GAMES)}")
            try:
                self.benchmark_game(
                    env_id=game,
                    num_episodes=num_episodes,
                    render=render,
                    record=record,
                )
            except Exception as e:
                print(f"Error benchmarking {game}: {e}")
                self.results[game] = {"error": str(e)}
        
        return self.results
    
    def save_benchmark_results(self) -> None:
        """Save benchmark results to files."""
        # Save individual game results
        for game, results in self.results.items():
            game_dir = self.output_dir / game
            game_dir.mkdir(exist_ok=True)
            
            with open(game_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
        
        # Save summary
        summary = self._compute_summary()
        with open(self.output_dir / "benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save human-readable report
        self._save_report(summary)
        
        print(f"\nBenchmark results saved to: {self.output_dir}")
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute benchmark summary.
        
        Returns:
            Summary statistics
        """
        successful_games = {k: v for k, v in self.results.items() if "error" not in v}
        failed_games = {k: v for k, v in self.results.items() if "error" in v}
        
        if not successful_games:
            return {
                "total_games": len(self.results),
                "successful_games": 0,
                "failed_games": len(failed_games),
                "errors": failed_games,
            }
        
        # Compute aggregate statistics
        all_rewards = []
        all_lengths = []
        game_stats = {}
        
        for game, results in successful_games.items():
            rewards = results.get('episode_rewards', [])
            lengths = results.get('episode_lengths', [])
            
            all_rewards.extend(rewards)
            all_lengths.extend(lengths)
            
            game_stats[game] = {
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "mean_length": float(np.mean(lengths)) if lengths else 0.0,
                "num_episodes": len(rewards),
            }
        
        summary = {
            "total_games": len(self.results),
            "successful_games": len(successful_games),
            "failed_games": len(failed_games),
            "overall_stats": {
                "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "std_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
                "mean_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
                "total_episodes": len(all_rewards),
            },
            "game_stats": game_stats,
            "errors": failed_games,
        }
        
        return summary
    
    def _save_report(self, summary: Dict[str, Any]) -> None:
        """Save human-readable benchmark report.
        
        Args:
            summary: Summary statistics
        """
        report_path = self.output_dir / "benchmark_report.txt"
        
        with open(report_path, "w") as f:
            f.write("HARD EXPLORATION ATARI BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Games: {summary['total_games']}\n")
            f.write(f"Successful: {summary['successful_games']}\n")
            f.write(f"Failed: {summary['failed_games']}\n")
            
            if summary['successful_games'] > 0:
                overall = summary['overall_stats']
                f.write(f"Mean Reward: {overall['mean_reward']:.2f} ± {overall['std_reward']:.2f}\n")
                f.write(f"Mean Length: {overall['mean_length']:.1f}\n")
                f.write(f"Total Episodes: {overall['total_episodes']}\n")
            
            f.write("\nGAME-BY-GAME RESULTS\n")
            f.write("-" * 25 + "\n")
            
            for game, stats in summary['game_stats'].items():
                f.write(f"\n{game}:\n")
                f.write(f"  Episodes: {stats['num_episodes']}\n")
                f.write(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n")
                f.write(f"  Mean Length: {stats['mean_length']:.1f}\n")
            
            if summary['errors']:
                f.write("\nERRORS\n")
                f.write("-" * 10 + "\n")
                for game, error in summary['errors'].items():
                    f.write(f"{game}: {error['error']}\n")
        
        print(f"Benchmark report saved to: {report_path}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark hard exploration Atari games")
    
    # Model and output
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    
    # Games to benchmark
    parser.add_argument("--env-id", help="Single environment to benchmark")
    parser.add_argument("--all-games", action="store_true", help="Benchmark all hard exploration games")
    
    # Evaluation parameters
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes per game")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--record", action="store_true", help="Record videos")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.env_id and not args.all_games:
        print("Error: Must specify either --env-id or --all-games")
        sys.exit(1)
    
    if args.env_id and args.all_games:
        print("Error: Cannot specify both --env-id and --all-games")
        sys.exit(1)
    
    # Create benchmark
    benchmark = HardExplorationBenchmark(
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )
    
    # Run benchmark
    if args.all_games:
        benchmark.benchmark_all_games(
            num_episodes=args.num_episodes,
            render=args.render,
            record=args.record,
        )
    else:
        benchmark.benchmark_game(
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            render=args.render,
            record=args.record,
        )
    
    # Save results
    benchmark.save_benchmark_results()
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()

