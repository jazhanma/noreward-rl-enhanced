"""
Checkpointing utilities for saving and loading training state.

This module provides comprehensive checkpointing functionality for:
- Model weights and optimizer state
- Training progress (steps, episodes, metrics)
- Configuration and hyperparameters
- Seamless resume functionality
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import tensorflow as tf


class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    This class handles the complete state of a training run including
    model weights, optimizer state, training progress, and configuration.
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_interval: int = 100,
        checkpoint_prefix: str = "checkpoint"
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_interval: Save checkpoint every N steps/episodes
            checkpoint_prefix: Prefix for checkpoint files
        """
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval
        self.checkpoint_prefix = checkpoint_prefix

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoint metadata
        self.checkpoint_metadata_file = self.save_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()

        print(f"📁 Checkpoint manager initialized: {self.save_dir}")
        print("   Save interval: {self.save_interval}")
        print("   Max checkpoints: {self.max_checkpoints}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata from file."""
        if self.checkpoint_metadata_file.exists():
            try:
                with open(self.checkpoint_metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print("⚠️  Warning: Could not load checkpoint metadata: {e}")

        return {
            'checkpoints': [],
            'latest_checkpoint': None,
            'best_checkpoint': None,
            'training_start_time': None
        }

    def _save_metadata(self) -> None:
        """Save checkpoint metadata to file."""
        try:
            with open(self.checkpoint_metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            print("⚠️  Warning: Could not save checkpoint metadata: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        checkpoints = self.metadata.get('checkpoints', [])
        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            checkpoints_to_remove = checkpoints[:-self.max_checkpoints]
            for checkpoint in checkpoints_to_remove:
                checkpoint_path = self.save_dir / checkpoint['filename']
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print("🗑️  Removed old checkpoint: {checkpoint['filename']}")

            # Update metadata
            self.metadata['checkpoints'] = checkpoints[-self.max_checkpoints:]
            self._save_metadata()

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        step: int = 0,
        episode: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            model: Model to save (TensorFlow model or checkpointable object)
            optimizer: Optimizer state to save
            step: Current training step
            episode: Current episode
            metrics: Training metrics to save
            config: Training configuration
            is_best: Whether this is the best checkpoint so far
            additional_data: Any additional data to save

        Returns:
            Path to saved checkpoint
        """
        timestamp = int(time.time())
        checkpoint_name = "{self.checkpoint_prefix}_step_{step}_ep_{episode}_{timestamp}"
        checkpoint_path = self.save_dir / "{checkpoint_name}.ckpt"

        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'episode': episode,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'config': config or {},
            'additional_data': additional_data or {}
        }

        # Save TensorFlow checkpoint
        if hasattr(model, 'save_weights'):
            # For Keras models
            model.save_weights(str(checkpoint_path))
        elif hasattr(model, 'save'):
            # For TensorFlow checkpoints
            model.save(str(checkpoint_path))
        else:
            # For custom models, try to save as pickle
            try:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print("⚠️  Warning: Could not save model: {e}")
                return None

        # Save optimizer state if provided
        if optimizer is not None:
            optimizer_path = self.save_dir / "{checkpoint_name}_optimizer.pkl"
            try:
                with open(optimizer_path, 'wb') as f:
                    pickle.dump(optimizer, f)
            except Exception as e:
                print("⚠️  Warning: Could not save optimizer: {e}")

        # Save checkpoint metadata
        checkpoint_info = {
            'filename': "{checkpoint_name}.ckpt",
            'step': step,
            'episode': episode,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'config': config or {},
            'is_best': is_best
        }

        # Update metadata
        self.metadata['checkpoints'].append(checkpoint_info)
        self.metadata['latest_checkpoint'] = checkpoint_info

        if is_best:
            self.metadata['best_checkpoint'] = checkpoint_info

        # Set training start time if not set
        if self.metadata['training_start_time'] is None:
            self.metadata['training_start_time'] = timestamp

        self._save_metadata()
        self._cleanup_old_checkpoints()

        print("💾 Checkpoint saved: {checkpoint_name}")
        print("   Step: {step}, Episode: {episode}")
        if metrics:
            print("   Metrics: {metrics}")

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Load a training checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            checkpoint_path: Specific checkpoint to load (if None, loads latest)
            load_best: Whether to load the best checkpoint instead of latest

        Returns:
            Tuple of (checkpoint_data, success_flag)
        """
        # Determine which checkpoint to load
        if checkpoint_path is not None:
            checkpoint_file = Path(checkpoint_path)
        elif load_best and self.metadata.get('best_checkpoint'):
            checkpoint_file = self.save_dir / self.metadata['best_checkpoint']['filename']
        elif self.metadata.get('latest_checkpoint'):
            checkpoint_file = self.save_dir / self.metadata['latest_checkpoint']['filename']
        else:
            print("❌ No checkpoint found to load")
            return {}, False

        if not checkpoint_file.exists():
            print("❌ Checkpoint file not found: {checkpoint_file}")
            return {}, False

        try:
            # Load model weights
            if hasattr(model, 'load_weights'):
                # For Keras models
                model.load_weights(str(checkpoint_file))
            elif hasattr(model, 'restore'):
                # For TensorFlow checkpoints
                model.restore(str(checkpoint_file))
            else:
                # For custom models, try to load as pickle
                with open(checkpoint_file, 'rb') as f:
                    loaded_model = pickle.load(f)
                    # Copy weights or state to the provided model
                    if hasattr(loaded_model, 'get_weights') and hasattr(model, 'set_weights'):
                        model.set_weights(loaded_model.get_weights())

            # Load optimizer state if provided
            if optimizer is not None:
                optimizer_file = checkpoint_file.with_suffix('.pkl').with_name(
                    checkpoint_file.stem.replace('.ckpt', '_optimizer.pkl')
                )
                if optimizer_file.exists():
                    with open(optimizer_file, 'rb') as f:
                        optimizer_state = pickle.load(f)
                        # Restore optimizer state
                        if hasattr(optimizer, 'set_weights'):
                            optimizer.set_weights(optimizer_state)

            # Load checkpoint metadata
            checkpoint_data = self._get_checkpoint_data(checkpoint_file)

            print("✅ Checkpoint loaded: {checkpoint_file.name}")
            print("   Step: {checkpoint_data.get('step', 'Unknown')}")
            print("   Episode: {checkpoint_data.get('episode', 'Unknown')}")

            return checkpoint_data, True

        except Exception as e:
            print("❌ Error loading checkpoint: {e}")
            return {}, False

    def _get_checkpoint_data(self, checkpoint_file: Path) -> Dict[str, Any]:
        """Get metadata for a specific checkpoint file."""
        checkpoint_name = checkpoint_file.stem

        # Find checkpoint in metadata
        for checkpoint in self.metadata.get('checkpoints', []):
            if checkpoint['filename'] == checkpoint_file.name:
                return checkpoint

        # If not found in metadata, return basic info
        return {
            'step': 0,
            'episode': 0,
            'timestamp': int(checkpoint_file.stat().st_mtime),
            'metrics': {},
            'config': {},
            'is_best': False
        }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.metadata.get('checkpoints', [])

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint info."""
        return self.metadata.get('latest_checkpoint')

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the best checkpoint info."""
        return self.metadata.get('best_checkpoint')

    def should_save_checkpoint(self, step: int, episode: int) -> bool:
        """Check if a checkpoint should be saved at this step/episode."""
        return (step > 0 and step % self.save_interval == 0) or (episode > 0 and episode % self.save_interval == 0)

    def cleanup(self) -> None:
        """Clean up checkpoint manager resources."""
        self._save_metadata()
        print("🧹 Checkpoint manager cleanup completed: {self.save_dir}")


def create_checkpoint_manager(
    save_dir: Union[str, Path],
    max_checkpoints: int = 5,
    save_interval: int = 100,
    checkpoint_prefix: str = "checkpoint"
) -> CheckpointManager:
    """
    Create a checkpoint manager instance.

    Args:
        save_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        save_interval: Save checkpoint every N steps/episodes
        checkpoint_prefix: Prefix for checkpoint files

    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(
        save_dir=save_dir,
        max_checkpoints=max_checkpoints,
        save_interval=save_interval,
        checkpoint_prefix=checkpoint_prefix
    )


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Look for checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

    if not checkpoint_files:
        return None

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return checkpoint_files[0]


def cleanup_checkpoint_directory(
    checkpoint_dir: Union[str, Path],
    keep_latest: int = 5
) -> None:
    """
    Clean up old checkpoints in a directory.

    Args:
        checkpoint_dir: Directory to clean up
        keep_latest: Number of latest checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

    if len(checkpoint_files) <= keep_latest:
        return

    # Sort by modification time (oldest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime)

    # Remove oldest checkpoints
    files_to_remove = checkpoint_files[:-keep_latest]
    for file_path in files_to_remove:
        file_path.unlink()
        print("🗑️  Removed old checkpoint: {file_path.name}")


if __name__ == "__main__":
    # Test checkpoint manager
    print("🧪 Testing checkpoint manager...")

    # Create test checkpoint manager
    test_dir = Path("test_checkpoints")
    manager = create_checkpoint_manager(
        save_dir=test_dir,
        max_checkpoints=3,
        save_interval=10
    )

    # Test metadata operations
    print("Checkpoints: {manager.list_checkpoints()}")
    print("Latest: {manager.get_latest_checkpoint()}")
    print("Best: {manager.get_best_checkpoint()}")

    # Test cleanup
    manager.cleanup()

    # Clean up test directory
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print("✅ Checkpoint manager test completed!")
