"""
Tests for configuration management utilities.

This module tests the ConfigManager class and related functionality
for loading, validating, and managing YAML configuration files.
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager, ConfigError, load_config, create_config_from_args


class TestConfigManager:
    """Test cases for ConfigManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.env_dir = self.config_dir / "environments"
        self.exp_dir = self.config_dir / "experiments"

        # Create directories
        self.env_dir.mkdir(parents=True)
        self.exp_dir.mkdir(parents=True)

        self.manager = ConfigManager(str(self.config_dir))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test ConfigManager initialization."""
        assert self.manager.config_dir == self.config_dir
        assert self.manager.environments_dir == self.env_dir
        assert self.manager.experiments_dir == self.exp_dir
        assert self.env_dir.exists()
        assert self.exp_dir.exists()

    def test_load_environment_config_success(self):
        """Test successful environment config loading."""
        # Create test environment config
        env_config = {
            'name': 'TestEnv',
            'type': 'test',
            'description': 'Test environment',
            'training': {
                'learning_rate': 0.001,
                'max_episodes': 100
            }
        }

        config_file = self.env_dir / "test_env.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(env_config, f)

        # Load config
        loaded_config = self.manager.load_environment_config("test_env")

        assert loaded_config['name'] == 'TestEnv'
        assert loaded_config['type'] == 'test'
        assert loaded_config['training']['learning_rate'] == 0.001

    def test_load_environment_config_not_found(self):
        """Test environment config not found error."""
        with pytest.raises(ConfigError, match="Environment config not found"):
            self.manager.load_environment_config("nonexistent")

    def test_load_environment_config_invalid_yaml(self):
        """Test invalid YAML in environment config."""
        config_file = self.env_dir / "invalid.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ConfigError, match="Invalid YAML"):
            self.manager.load_environment_config("invalid")

    def test_load_experiment_config_success(self):
        """Test successful experiment config loading."""
        # Create test experiment config
        exp_config = {
            'experiment': {
                'name': 'test_experiment',
                'description': 'Test experiment'
            },
            'model': {
                'type': 'A3C',
                'design_head': 'universe'
            },
            'training': {
                'learning_rate': 0.0001,
                'max_global_steps': 1000000
            }
        }

        config_file = self.exp_dir / "test_experiment.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(exp_config, f)

        # Load config
        loaded_config = self.manager.load_experiment_config("test_experiment")

        assert loaded_config['experiment']['name'] == 'test_experiment'
        assert loaded_config['model']['type'] == 'A3C'
        assert loaded_config['training']['learning_rate'] == 0.0001

    def test_load_experiment_config_not_found(self):
        """Test experiment config not found error."""
        with pytest.raises(ConfigError, match="Experiment config not found"):
            self.manager.load_experiment_config("nonexistent")

    def test_load_combined_config_success(self):
        """Test successful combined config loading."""
        # Create environment config
        env_config = {
            'name': 'TestEnv',
            'type': 'test',
            'description': 'Test environment',
            'training': {
                'learning_rate': 0.001,
                'max_episodes': 100
            }
        }

        env_file = self.env_dir / "test_env.yaml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f)

        # Create experiment config
        exp_config = {
            'experiment': {
                'name': 'test_experiment',
                'description': 'Test experiment'
            },
            'model': {
                'type': 'A3C',
                'design_head': 'universe'
            },
            'training': {
                'learning_rate': 0.0001,
                'max_global_steps': 1000000
            },
            'environment': {
                'env_id': 'TestEnv-v0'
            }
        }

        exp_file = self.exp_dir / "test_experiment.yaml"
        with open(exp_file, 'w') as f:
            yaml.dump(exp_config, f)

        # Load combined config
        combined_config = self.manager.load_combined_config("test_experiment", "test_env")

        # Should have experiment config
        assert combined_config['experiment']['name'] == 'test_experiment'

        # Should have environment-specific training overrides
        assert combined_config['training']['learning_rate'] == 0.001  # From env config
        assert combined_config['training']['max_global_steps'] == 1000000  # From exp config

    def test_load_combined_config_missing_env(self):
        """Test combined config loading with missing environment."""
        # Create experiment config
        exp_config = {
            'experiment': {
                'name': 'test_experiment',
                'description': 'Test experiment'
            },
            'model': {
                'type': 'A3C',
                'design_head': 'universe'
            },
            'training': {
                'learning_rate': 0.0001,
                'max_global_steps': 1000000
            },
            'environment': {
                'env_id': 'TestEnv-v0'
            }
        }

        exp_file = self.exp_dir / "test_experiment.yaml"
        with open(exp_file, 'w') as f:
            yaml.dump(exp_config, f)

        # Load combined config (should work without env config)
        combined_config = self.manager.load_combined_config("test_experiment", "nonexistent")

        assert combined_config['experiment']['name'] == 'test_experiment'
        assert combined_config['training']['learning_rate'] == 0.0001

    def test_validate_environment_config(self):
        """Test environment config validation."""
        # Valid config
        valid_config = {
            'name': 'TestEnv',
            'type': 'test',
            'description': 'Test environment'
        }
        self.manager._validate_environment_config(valid_config)  # Should not raise

        # Invalid config (missing required field)
        invalid_config = {
            'name': 'TestEnv',
            'type': 'test'
            # Missing 'description'
        }
        with pytest.raises(ConfigError, match="Missing required field"):
            self.manager._validate_environment_config(invalid_config)

    def test_validate_experiment_config(self):
        """Test experiment config validation."""
        # Valid config
        valid_config = {
            'experiment': {'name': 'test'},
            'model': {'type': 'A3C'},
            'training': {'learning_rate': 0.001}
        }
        self.manager._validate_experiment_config(valid_config)  # Should not raise

        # Invalid config (missing required field)
        invalid_config = {
            'experiment': {'name': 'test'},
            'model': {'type': 'A3C'}
            # Missing 'training'
        }
        with pytest.raises(ConfigError, match="Missing required field"):
            self.manager._validate_experiment_config(invalid_config)

    def test_merge_configs(self):
        """Test config merging functionality."""
        exp_config = {
            'experiment': {'name': 'test'},
            'training': {'learning_rate': 0.0001, 'max_global_steps': 1000000},
            'environment': {'env_id': 'TestEnv-v0'}
        }

        env_config = {
            'name': 'TestEnv',
            'type': 'test',
            'training': {'learning_rate': 0.001, 'max_episodes': 100},
            'env_params': {'frame_skip': 4},
            'wrappers': [{'name': 'FrameSkip', 'skip': 4}]
        }

        merged = self.manager._merge_configs(exp_config, env_config)

        # Should have experiment config
        assert merged['experiment']['name'] == 'test'

        # Should have environment-specific training overrides
        assert merged['training']['learning_rate'] == 0.001  # From env
        assert merged['training']['max_global_steps'] == 1000000  # From exp

        # Should have environment-specific parameters
        assert merged['environment']['env_params']['frame_skip'] == 4
        assert merged['environment']['wrappers'][0]['name'] == 'FrameSkip'

    def test_list_available_configs(self):
        """Test listing available configuration files."""
        # Create some test configs
        env_config = {'name': 'TestEnv', 'type': 'test', 'description': 'Test'}
        exp_config = {'experiment': {'name': 'test'}, 'model': {'type': 'A3C'}, 'training': {'learning_rate': 0.001}}

        with open(self.env_dir / "test_env.yaml", 'w') as f:
            yaml.dump(env_config, f)

        with open(self.exp_dir / "test_exp.yaml", 'w') as f:
            yaml.dump(exp_config, f)

        env_configs, exp_configs = self.manager.list_available_configs()

        assert "test_env" in env_configs
        assert "test_exp" in exp_configs

    def test_save_config(self):
        """Test saving configuration to file."""
        config = {
            'experiment': {'name': 'test'},
            'training': {'learning_rate': 0.001}
        }

        save_path = Path(self.temp_dir) / "test_save.yaml"
        self.manager.save_config(config, save_path)

        assert save_path.exists()

        # Verify content
        with open(save_path, 'r') as f:
            loaded = yaml.safe_load(f)

        assert loaded['experiment']['name'] == 'test'
        assert loaded['training']['learning_rate'] == 0.001


class TestConfigFunctions:
    """Test cases for configuration utility functions."""

    def test_load_config_function(self):
        """Test load_config convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            env_dir = config_dir / "environments"
            exp_dir = config_dir / "experiments"

            env_dir.mkdir(parents=True)
            exp_dir.mkdir(parents=True)

            # Create test configs
            env_config = {
                'name': 'TestEnv',
                'type': 'test',
                'description': 'Test environment'
            }

            exp_config = {
                'experiment': {'name': 'test'},
                'model': {'type': 'A3C'},
                'training': {'learning_rate': 0.001, 'max_global_steps': 1000000},
                'environment': {'env_id': 'TestEnv-v0'}
            }

            with open(env_dir / "test_env.yaml", 'w') as f:
                yaml.dump(env_config, f)

            with open(exp_dir / "test_exp.yaml", 'w') as f:
                yaml.dump(exp_config, f)

            # Test loading
            config = load_config("test_exp", "test_env", str(config_dir))

            assert config['experiment']['name'] == 'test'
            assert config['training']['learning_rate'] == 0.001

    def test_create_config_from_args(self):
        """Test creating config from command line arguments."""
        # Mock argparse.Namespace
        class MockArgs:
            def __init__(self):
                self.experiment_name = "test"
                self.design_head = "universe"
                self.unsup = "action"
                self.num_workers = 4
                self.max_steps = 1000000
                self.learning_rate = 0.0001
                self.gamma = 0.99
                self.entropy_beta = 0.01
                self.prediction_beta = 0.2
                self.env_id = "CartPole-v1"
                self.env_wrap = True
                self.no_reward = False
                self.no_life_reward = False
                self.use_wandb = True
                self.use_tensorboard = False
                self.log_dir = "logs"
                self.save_dir = "checkpoints"
                self.save_interval = 1000
                self.resume = False
                self.seed = 42

        args = MockArgs()
        config = create_config_from_args(args)

        assert config['experiment']['name'] == "test"
        assert config['model']['design_head'] == "universe"
        assert config['training']['learning_rate'] == 0.0001
        assert config['environment']['env_id'] == "CartPole-v1"
        assert config['logging']['use_wandb'] is True
        assert config['reproducibility']['seed'] == 42


class TestConfigError:
    """Test cases for ConfigError exception."""

    def test_config_error_creation(self):
        """Test ConfigError creation and message."""
        error = ConfigError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__])
