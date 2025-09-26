"""
Tests for checkpointing functionality.

This module tests the checkpointing utilities to ensure proper saving,
loading, and resuming of training state.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add project root to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint import (
    CheckpointManager,
    create_checkpoint_manager,
    find_latest_checkpoint,
    cleanup_checkpoint_directory
)


class TestCheckpointManager:
    """Test checkpoint manager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "test_checkpoints"
        self.manager = create_checkpoint_manager(
            save_dir=self.checkpoint_dir,
            max_checkpoints=3,
            save_interval=5
        )

    def teardown_method(self):
        """Clean up test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_manager_initialization(self):
        """Test checkpoint manager initialization."""
        assert self.manager.save_dir == self.checkpoint_dir
        assert self.manager.max_checkpoints == 3
        assert self.manager.save_interval == 5
        assert self.checkpoint_dir.exists()

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        # Create a dummy model
        dummy_model = {'weights': [1, 2, 3], 'bias': [0.1, 0.2]}

        # Save checkpoint
        checkpoint_path = self.manager.save_checkpoint(
            model=dummy_model,
            step=10,
            episode=5,
            metrics={'reward': 100.0, 'loss': 0.5},
            config={'learning_rate': 0.001},
            is_best=True
        )

        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()

        # Check metadata
        checkpoints = self.manager.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]['step'] == 10
        assert checkpoints[0]['episode'] == 5
        assert checkpoints[0]['is_best'] is True

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        # Create and save a dummy model
        dummy_model = {'weights': [1, 2, 3], 'bias': [0.1, 0.2]}
        self.manager.save_checkpoint(
            model=dummy_model,
            step=10,
            episode=5,
            metrics={'reward': 100.0}
        )

        # Load checkpoint
        loaded_model = {'weights': [], 'bias': []}
        checkpoint_data, success = self.manager.load_checkpoint(loaded_model)

        assert success is True
        assert checkpoint_data['step'] == 10
        assert checkpoint_data['episode'] == 5
        assert checkpoint_data['metrics']['reward'] == 100.0

    def test_checkpoint_cleanup(self):
        """Test automatic cleanup of old checkpoints."""
        # Save more checkpoints than max_checkpoints
        for i in range(5):
            dummy_model = {'weights': [i], 'bias': [i * 0.1]}
            self.manager.save_checkpoint(
                model=dummy_model,
                step=i * 10,
                episode=i * 5,
                metrics={'reward': i * 10.0}
            )

        # Should only keep max_checkpoints (3) checkpoints
        checkpoints = self.manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Should keep the most recent checkpoints
        steps = [cp['step'] for cp in checkpoints]
        assert max(steps) == 40  # Most recent checkpoint
        assert min(steps) == 20  # Oldest kept checkpoint

    def test_should_save_checkpoint(self):
        """Test checkpoint saving logic."""
        # Should save at save_interval
        assert self.manager.should_save_checkpoint(5, 0) is True
        assert self.manager.should_save_checkpoint(10, 0) is True
        assert self.manager.should_save_checkpoint(15, 0) is True

        # Should not save between intervals
        assert self.manager.should_save_checkpoint(3, 0) is False
        assert self.manager.should_save_checkpoint(7, 0) is False
        assert self.manager.should_save_checkpoint(12, 0) is False

    def test_best_checkpoint_tracking(self):
        """Test best checkpoint tracking."""
        # Save multiple checkpoints with different metrics
        for i, reward in enumerate([50.0, 75.0, 100.0, 60.0]):
            dummy_model = {'weights': [i]}
            self.manager.save_checkpoint(
                model=dummy_model,
                step=i * 10,
                episode=i * 5,
                metrics={'reward': reward},
                is_best=(reward == 100.0)
            )

        # Check best checkpoint
        best_checkpoint = self.manager.get_best_checkpoint()
        assert best_checkpoint is not None
        assert best_checkpoint['metrics']['reward'] == 100.0
        assert best_checkpoint['is_best'] is True

    def test_latest_checkpoint_tracking(self):
        """Test latest checkpoint tracking."""
        # Save multiple checkpoints
        for i in range(3):
            dummy_model = {'weights': [i]}
            self.manager.save_checkpoint(
                model=dummy_model,
                step=i * 10,
                episode=i * 5,
                metrics={'reward': i * 10.0}
            )

        # Check latest checkpoint
        latest_checkpoint = self.manager.get_latest_checkpoint()
        assert latest_checkpoint is not None
        assert latest_checkpoint['step'] == 20  # Most recent
        assert latest_checkpoint['episode'] == 10


class TestCheckpointUtilities:
    """Test checkpoint utility functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "test_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint."""
        # Create some checkpoint files with different timestamps
        checkpoint_files = []
        for i in range(3):
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{i}.ckpt"
            checkpoint_file.touch()
            checkpoint_files.append(checkpoint_file)

        # Find latest checkpoint
        latest = find_latest_checkpoint(self.checkpoint_dir)
        assert latest is not None
        assert latest in checkpoint_files

    def test_find_latest_checkpoint_empty_dir(self):
        """Test finding latest checkpoint in empty directory."""
        latest = find_latest_checkpoint(self.checkpoint_dir)
        assert latest is None

    def test_find_latest_checkpoint_nonexistent_dir(self):
        """Test finding latest checkpoint in non-existent directory."""
        nonexistent_dir = self.checkpoint_dir / "nonexistent"
        latest = find_latest_checkpoint(nonexistent_dir)
        assert latest is None

    def test_cleanup_checkpoint_directory(self):
        """Test cleaning up checkpoint directory."""
        # Create more checkpoint files than keep_latest
        checkpoint_files = []
        for i in range(10):
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{i}.ckpt"
            checkpoint_file.touch()
            checkpoint_files.append(checkpoint_file)

        # Clean up, keeping only 3 latest
        cleanup_checkpoint_directory(self.checkpoint_dir, keep_latest=3)

        # Should only have 3 files left
        remaining_files = list(self.checkpoint_dir.glob("*.ckpt"))
        assert len(remaining_files) == 3


class TestCheckpointIntegration:
    """Test checkpointing integration with training."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "integration_test"

    def teardown_method(self):
        """Clean up test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_metadata_persistence(self):
        """Test that checkpoint metadata persists across manager instances."""
        # Create first manager and save checkpoint
        manager1 = create_checkpoint_manager(
            save_dir=self.checkpoint_dir,
            max_checkpoints=5,
            save_interval=10
        )

        dummy_model = {'weights': [1, 2, 3]}
        manager1.save_checkpoint(
            model=dummy_model,
            step=10,
            episode=5,
            metrics={'reward': 100.0},
            is_best=True
        )

        # Create second manager and check metadata
        manager2 = create_checkpoint_manager(
            save_dir=self.checkpoint_dir,
            max_checkpoints=5,
            save_interval=10
        )

        checkpoints = manager2.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]['step'] == 10
        assert checkpoints[0]['is_best'] is True

        latest = manager2.get_latest_checkpoint()
        assert latest is not None
        assert latest['step'] == 10

    def test_checkpoint_with_complex_data(self):
        """Test checkpointing with complex data structures."""
        manager = create_checkpoint_manager(
            save_dir=self.checkpoint_dir,
            max_checkpoints=3,
            save_interval=5
        )

        # Complex model with nested data
        complex_model = {
            'layers': [
                {'weights': [[1, 2], [3, 4]], 'bias': [0.1, 0.2]},
                {'weights': [[5, 6], [7, 8]], 'bias': [0.3, 0.4]}
            ],
            'optimizer_state': {
                'momentum': [0.9, 0.8],
                'learning_rate': 0.001
            }
        }

        # Complex metrics
        complex_metrics = {
            'episode_rewards': [10.0, 20.0, 30.0],
            'losses': {'policy': 0.5, 'value': 0.3, 'curiosity': 0.2},
            'performance': {'mean_reward': 20.0, 'std_reward': 8.16}
        }

        # Complex config
        complex_config = {
            'env_id': 'CartPole-v1',
            'hyperparameters': {
                'learning_rate': 0.001,
                'gamma': 0.99,
                'entropy_beta': 0.01
            },
            'architecture': {
                'hidden_sizes': [64, 32],
                'activation': 'relu'
            }
        }

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=complex_model,
            step=100,
            episode=50,
            metrics=complex_metrics,
            config=complex_config,
            is_best=True,
            additional_data={'custom_field': 'test_value'}
        )

        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()

        # Load checkpoint
        loaded_model = {}
        checkpoint_data, success = manager.load_checkpoint(loaded_model)

        assert success is True
        assert checkpoint_data['step'] == 100
        assert checkpoint_data['episode'] == 50
        assert checkpoint_data['is_best'] is True
        assert checkpoint_data['metrics']['performance']['mean_reward'] == 20.0
        assert checkpoint_data['config']['env_id'] == 'CartPole-v1'


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
