#!/usr/bin/env python3
"""
Configuration validation script for NoReward-RL.

This script validates all configuration files and checks for
consistency, required fields, and potential issues.
"""
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ConfigManager, ConfigError


def validate_all_configs(config_dir: str = "configs") -> Tuple[bool, List[str]]:
    """
    Validate all configuration files.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Tuple of (all_valid, error_messages)
    """
    manager = ConfigManager(config_dir)
    errors = []
    
    print("🔍 Validating configuration files...")
    print("=" * 50)
    
    # Get available configs
    env_configs, exp_configs = manager.list_available_configs()
    
    print(f"Found {len(env_configs)} environment configs: {env_configs}")
    print(f"Found {len(exp_configs)} experiment configs: {exp_configs}")
    print()
    
    # Validate environment configs
    print("🌍 Validating environment configurations...")
    for env_name in env_configs:
        try:
            config = manager.load_environment_config(env_name)
            print(f"  ✅ {env_name}: {config.get('name', 'Unknown')}")
            
            # Additional validation
            if 'training' in config:
                training = config['training']
                if 'learning_rate' in training and training['learning_rate'] <= 0:
                    errors.append(f"Environment {env_name}: Invalid learning_rate {training['learning_rate']}")
                if 'max_episodes' in training and training['max_episodes'] <= 0:
                    errors.append(f"Environment {env_name}: Invalid max_episodes {training['max_episodes']}")
                    
        except ConfigError as e:
            error_msg = f"Environment {env_name}: {e}"
            errors.append(error_msg)
            print(f"  ❌ {env_name}: {e}")
        except Exception as e:
            error_msg = f"Environment {env_name}: Unexpected error - {e}"
            errors.append(error_msg)
            print(f"  ❌ {env_name}: Unexpected error - {e}")
    
    print()
    
    # Validate experiment configs
    print("🧪 Validating experiment configurations...")
    for exp_name in exp_configs:
        try:
            config = manager.load_experiment_config(exp_name)
            print(f"  ✅ {exp_name}: {config.get('experiment', {}).get('name', 'Unknown')}")
            
            # Additional validation
            training = config.get('training', {})
            if 'learning_rate' in training and training['learning_rate'] <= 0:
                errors.append(f"Experiment {exp_name}: Invalid learning_rate {training['learning_rate']}")
            if 'max_global_steps' in training and training['max_global_steps'] <= 0:
                errors.append(f"Experiment {exp_name}: Invalid max_global_steps {training['max_global_steps']}")
            if 'num_workers' in training and training['num_workers'] <= 0:
                errors.append(f"Experiment {exp_name}: Invalid num_workers {training['num_workers']}")
                
        except ConfigError as e:
            error_msg = f"Experiment {exp_name}: {e}"
            errors.append(error_msg)
            print(f"  ❌ {exp_name}: {e}")
        except Exception as e:
            error_msg = f"Experiment {exp_name}: Unexpected error - {e}"
            errors.append(error_msg)
            print(f"  ❌ {exp_name}: Unexpected error - {e}")
    
    print()
    
    # Test combined configs
    print("🔗 Testing combined configurations...")
    for exp_name in exp_configs:
        try:
            # Try to load with default environment
            config = manager.load_combined_config(exp_name)
            env_id = config.get('environment', {}).get('env_id', 'Unknown')
            print(f"  ✅ {exp_name} + default env: {env_id}")
            
            # Test with specific environments
            for env_name in env_configs:
                try:
                    combined_config = manager.load_combined_config(exp_name, env_name)
                    print(f"    ✅ {exp_name} + {env_name}: OK")
                except ConfigError as e:
                    error_msg = f"Combined {exp_name} + {env_name}: {e}"
                    errors.append(error_msg)
                    print(f"    ❌ {exp_name} + {env_name}: {e}")
                    
        except ConfigError as e:
            error_msg = f"Combined {exp_name}: {e}"
            errors.append(error_msg)
            print(f"  ❌ {exp_name}: {e}")
        except Exception as e:
            error_msg = f"Combined {exp_name}: Unexpected error - {e}"
            errors.append(error_msg)
            print(f"  ❌ {exp_name}: Unexpected error - {e}")
    
    print()
    
    # Summary
    all_valid = len(errors) == 0
    if all_valid:
        print("🎉 All configurations are valid!")
    else:
        print(f"❌ Found {len(errors)} validation errors:")
        for error in errors:
            print(f"  • {error}")
    
    return all_valid, errors


def validate_specific_config(config_path: str) -> bool:
    """
    Validate a specific configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"🔍 Validating {config_path}...")
        
        # Basic YAML validation
        if not isinstance(config, dict):
            print("❌ Configuration must be a dictionary")
            return False
        
        # Check for required sections based on file type
        if "environments" in str(config_path):
            required_sections = ['name', 'type', 'description']
            for section in required_sections:
                if section not in config:
                    print(f"❌ Missing required section: {section}")
                    return False
        elif "experiments" in str(config_path):
            required_sections = ['experiment', 'model', 'training']
            for section in required_sections:
                if section not in config:
                    print(f"❌ Missing required section: {section}")
                    return False
        
        print(f"✅ {config_path} is valid")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating {config_path}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate NoReward-RL configuration files")
    
    parser.add_argument("--config-dir", type=str, default="configs",
                       help="Directory containing configuration files")
    parser.add_argument("--file", type=str, help="Validate specific configuration file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🔍 NoReward-RL Configuration Validator")
    print("=" * 50)
    
    if args.file:
        # Validate specific file
        success = validate_specific_config(args.file)
        return 0 if success else 1
    else:
        # Validate all configs
        all_valid, errors = validate_all_configs(args.config_dir)
        
        if args.verbose and errors:
            print("\nDetailed error information:")
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
        
        return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
