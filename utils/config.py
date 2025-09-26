"""
Configuration management utilities for NoReward-RL.

This module provides comprehensive configuration loading, validation,
and management for experiments and environments.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigManager:
    """
    Manages loading and validation of configuration files.
    
    This class handles YAML configuration files for environments,
    experiments, and provides validation and merging capabilities.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.environments_dir = self.config_dir / "environments"
        self.experiments_dir = self.config_dir / "experiments"
        
        # Ensure directories exist
        self.environments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Config manager initialized: {self.config_dir}")
    
    def load_environment_config(self, env_name: str) -> Dict[str, Any]:
        """
        Load environment configuration.
        
        Args:
            env_name: Name of the environment (e.g., 'cartpole', 'doom')
            
        Returns:
            Environment configuration dictionary
            
        Raises:
            ConfigError: If configuration file not found or invalid
        """
        config_file = self.environments_dir / f"{env_name}.yaml"
        
        if not config_file.exists():
            raise ConfigError(f"Environment config not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            self._validate_environment_config(config)
            
            logger.info(f"Loaded environment config: {env_name}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_file}: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading config {config_file}: {e}")
    
    def load_experiment_config(self, exp_name: str) -> Dict[str, Any]:
        """
        Load experiment configuration.
        
        Args:
            exp_name: Name of the experiment (e.g., 'curiosity_baseline')
            
        Returns:
            Experiment configuration dictionary
            
        Raises:
            ConfigError: If configuration file not found or invalid
        """
        config_file = self.experiments_dir / f"{exp_name}.yaml"
        
        if not config_file.exists():
            raise ConfigError(f"Experiment config not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            self._validate_experiment_config(config)
            
            logger.info(f"Loaded experiment config: {exp_name}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_file}: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading config {config_file}: {e}")
    
    def load_combined_config(
        self, 
        exp_name: str, 
        env_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load and merge experiment and environment configurations.
        
        Args:
            exp_name: Name of the experiment
            env_name: Name of the environment (if None, uses env_id from experiment)
            
        Returns:
            Merged configuration dictionary
        """
        # Load experiment config
        exp_config = self.load_experiment_config(exp_name)
        
        # Determine environment name
        if env_name is None:
            env_name = exp_config.get('environment', {}).get('env_id', '').lower()
            # Convert env_id to config name (e.g., 'CartPole-v1' -> 'cartpole')
            env_name = env_name.split('-')[0]
        
        # Load environment config
        try:
            env_config = self.load_environment_config(env_name)
        except ConfigError:
            logger.warning(f"Environment config not found: {env_name}, using defaults")
            env_config = {}
        
        # Merge configurations
        merged_config = self._merge_configs(exp_config, env_config)
        
        # Validate merged config
        self._validate_merged_config(merged_config)
        
        logger.info(f"Loaded combined config: {exp_name} + {env_name}")
        return merged_config
    
    def _validate_environment_config(self, config: Dict[str, Any]) -> None:
        """Validate environment configuration."""
        required_fields = ['name', 'type', 'description']
        for field in required_fields:
            if field not in config:
                raise ConfigError(f"Missing required field in environment config: {field}")
    
    def _validate_experiment_config(self, config: Dict[str, Any]) -> None:
        """Validate experiment configuration."""
        required_fields = ['experiment', 'model', 'training']
        for field in required_fields:
            if field not in config:
                raise ConfigError(f"Missing required field in experiment config: {field}")
    
    def _validate_merged_config(self, config: Dict[str, Any]) -> None:
        """Validate merged configuration."""
        # Check for required top-level sections
        required_sections = ['experiment', 'model', 'training', 'environment']
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"Missing required section in merged config: {section}")
        
        # Validate training parameters
        training = config.get('training', {})
        if 'learning_rate' not in training:
            raise ConfigError("Missing learning_rate in training config")
        if 'max_global_steps' not in training:
            raise ConfigError("Missing max_global_steps in training config")
    
    def _merge_configs(
        self, 
        exp_config: Dict[str, Any], 
        env_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge experiment and environment configurations.
        
        Args:
            exp_config: Experiment configuration
            env_config: Environment configuration
            
        Returns:
            Merged configuration with environment overrides
        """
        merged = exp_config.copy()
        
        # Merge environment-specific training parameters
        if 'training' in env_config:
            env_training = env_config['training']
            if 'training' not in merged:
                merged['training'] = {}
            
            # Override with environment-specific values
            for key, value in env_training.items():
                merged['training'][key] = value
        
        # Add environment-specific parameters
        if 'env_params' in env_config:
            merged['environment']['env_params'] = env_config['env_params']
        
        # Add environment wrappers
        if 'wrappers' in env_config:
            merged['environment']['wrappers'] = env_config['wrappers']
        
        # Add performance targets
        if 'targets' in env_config:
            merged['targets'] = env_config['targets']
        
        return merged
    
    def list_available_configs(self) -> Tuple[List[str], List[str]]:
        """
        List available configuration files.
        
        Returns:
            Tuple of (environment_configs, experiment_configs)
        """
        env_configs = []
        exp_configs = []
        
        # List environment configs
        for config_file in self.environments_dir.glob("*.yaml"):
            env_configs.append(config_file.stem)
        
        # List experiment configs
        for config_file in self.experiments_dir.glob("*.yaml"):
            exp_configs.append(config_file.stem)
        
        return sorted(env_configs), sorted(exp_configs)
    
    def save_config(self, config: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            filepath: Path to save the configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved config to: {filepath}")
            
        except Exception as e:
            raise ConfigError(f"Error saving config to {filepath}: {e}")


def load_config(
    exp_name: str, 
    env_name: Optional[str] = None,
    config_dir: Union[str, Path] = "configs"
) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        exp_name: Name of the experiment
        env_name: Name of the environment (optional)
        config_dir: Directory containing configuration files
        
    Returns:
        Configuration dictionary
    """
    manager = ConfigManager(config_dir)
    return manager.load_combined_config(exp_name, env_name)


def create_config_from_args(args: Any) -> Dict[str, Any]:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {
        'experiment': {
            'name': getattr(args, 'experiment_name', 'default'),
            'description': 'Generated from command line arguments'
        },
        'model': {
            'type': 'A3C',
            'design_head': getattr(args, 'design_head', 'universe'),
            'unsup_type': getattr(args, 'unsup', 'action')
        },
        'training': {
            'num_workers': getattr(args, 'num_workers', 4),
            'max_global_steps': getattr(args, 'max_steps', 1000000),
            'learning_rate': getattr(args, 'learning_rate', 0.0001),
            'gamma': getattr(args, 'gamma', 0.99),
            'entropy_beta': getattr(args, 'entropy_beta', 0.01),
            'prediction_beta': getattr(args, 'prediction_beta', 0.2)
        },
        'environment': {
            'env_id': getattr(args, 'env_id', 'CartPole-v1'),
            'env_wrap': getattr(args, 'env_wrap', True),
            'no_reward': getattr(args, 'no_reward', False),
            'no_life_reward': getattr(args, 'no_life_reward', False)
        },
        'logging': {
            'use_wandb': getattr(args, 'use_wandb', False),
            'use_tensorboard': getattr(args, 'use_tensorboard', False),
            'log_dir': getattr(args, 'log_dir', 'logs')
        },
        'checkpointing': {
            'save_dir': getattr(args, 'save_dir', None),
            'save_interval': getattr(args, 'save_interval', 1000),
            'resume': getattr(args, 'resume', False)
        },
        'reproducibility': {
            'seed': getattr(args, 'seed', None),
            'deterministic': True
        }
    }
    
    return config


if __name__ == "__main__":
    # Test configuration loading
    print("🧪 Testing configuration manager...")
    
    manager = ConfigManager()
    
    # List available configs
    env_configs, exp_configs = manager.list_available_configs()
    print(f"Environment configs: {env_configs}")
    print(f"Experiment configs: {exp_configs}")
    
    # Test loading configs
    try:
        env_config = manager.load_environment_config("cartpole")
        print(f"✅ Loaded environment config: {env_config['name']}")
        
        exp_config = manager.load_experiment_config("curiosity_baseline")
        print(f"✅ Loaded experiment config: {exp_config['experiment']['name']}")
        
        combined_config = manager.load_combined_config("curiosity_baseline", "cartpole")
        print(f"✅ Loaded combined config successfully")
        
    except ConfigError as e:
        print(f"❌ Config error: {e}")
    
    print("✅ Configuration manager test completed!")
