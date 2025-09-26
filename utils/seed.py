"""
Reproducibility utilities for setting random seeds across different libraries.

This module provides centralized control over random number generation to ensure
reproducible experiments across Python's random, NumPy, TensorFlow, and Gymnasium.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import tensorflow as tf


def set_global_seed(seed: Union[int, None]) -> None:
    """
    Set random seeds for all major libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value. If None, seeds will be randomized.
        
    Note:
        This function sets seeds for:
        - Python's random module
        - NumPy
        - TensorFlow
        - Environment variable for additional libraries
    """
    if seed is not None:
        # Set Python's random seed
        random.seed(seed)
        
        # Set NumPy seed
        np.random.seed(seed)
        
        # Set TensorFlow seed
        tf.random.set_seed(seed)
        
        # Set environment variable for additional libraries
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set TensorFlow deterministic behavior
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        print(f"🌱 Global seed set to: {seed}")
    else:
        print("🌱 Using random seeds (non-deterministic)")


def set_gymnasium_seed(env: gym.Env, seed: Optional[int] = None) -> tuple[gym.Env, Optional[int]]:
    """
    Set the seed for a Gymnasium environment and return the environment with seed info.
    
    Args:
        env: The Gymnasium environment to seed
        seed: Random seed value. If None, a random seed will be generated.
        
    Returns:
        Tuple of (seeded_environment, actual_seed_used)
        
    Note:
        Some environments may not support seeding. In such cases, the original
        environment is returned with seed=None.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    try:
        # Try to seed the environment
        if hasattr(env, 'reset') and hasattr(env, 'action_space'):
            # For standard Gymnasium environments
            env.reset(seed=seed)
            actual_seed = seed
        else:
            # For wrapped environments, try to seed the unwrapped version
            if hasattr(env, 'unwrapped'):
                env.unwrapped.reset(seed=seed)
                actual_seed = seed
            else:
                print(f"⚠️  Warning: Could not seed environment {type(env).__name__}")
                actual_seed = None
                
    except Exception as e:
        print(f"⚠️  Warning: Failed to seed environment: {e}")
        actual_seed = None
    
    return env, actual_seed


def get_deterministic_config() -> dict[str, Union[int, bool, str]]:
    """
    Get configuration for deterministic behavior across all libraries.
    
    Returns:
        Dictionary containing configuration for deterministic behavior
    """
    return {
        'TF_DETERMINISTIC_OPS': '1',
        'TF_CUDNN_DETERMINISTIC': '1',
        'PYTHONHASHSEED': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_GPU_THREAD_MODE': 'gpu_private',
    }


def apply_deterministic_config() -> None:
    """
    Apply deterministic configuration to environment variables.
    
    This function sets environment variables that promote deterministic
    behavior in TensorFlow and other libraries.
    """
    config = get_deterministic_config()
    for key, value in config.items():
        os.environ[key] = str(value)
    
    print("🔧 Applied deterministic configuration")


def create_reproducible_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a reproducible NumPy random number generator.
    
    Args:
        seed: Random seed value. If None, a random seed will be generated.
        
    Returns:
        NumPy random number generator with the specified seed
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    rng = np.random.default_rng(seed)
    return rng


def test_reproducibility(seed: int = 42, num_steps: int = 10) -> None:
    """
    Test that the same seed produces identical results.
    
    Args:
        seed: Random seed to test
        num_steps: Number of random operations to test
        
    Returns:
        True if reproducible, False otherwise
    """
    # Test Python random
    random.seed(seed)
    python_values = [random.random() for _ in range(num_steps)]
    
    random.seed(seed)
    python_values_2 = [random.random() for _ in range(num_steps)]
    
    # Test NumPy random
    np.random.seed(seed)
    numpy_values = [np.random.random() for _ in range(num_steps)]
    
    np.random.seed(seed)
    numpy_values_2 = [np.random.random() for _ in range(num_steps)]
    
    # Test TensorFlow random
    tf.random.set_seed(seed)
    tf_values = [tf.random.normal([]).numpy() for _ in range(num_steps)]
    
    tf.random.set_seed(seed)
    tf_values_2 = [tf.random.normal([]).numpy() for _ in range(num_steps)]
    
    # Check reproducibility
    python_reproducible = python_values == python_values_2
    numpy_reproducible = all(np.allclose(a, b) for a, b in zip(numpy_values, numpy_values_2))
    tf_reproducible = all(np.allclose(a, b) for a, b in zip(tf_values, tf_values_2))
    
    all_reproducible = python_reproducible and numpy_reproducible and tf_reproducible
    
    print(f"🧪 Reproducibility test (seed={seed}):")
    print(f"   Python random: {'✅' if python_reproducible else '❌'}")
    print(f"   NumPy random: {'✅' if numpy_reproducible else '❌'}")
    print(f"   TensorFlow random: {'✅' if tf_reproducible else '❌'}")
    print(f"   Overall: {'✅' if all_reproducible else '❌'}")
    
    assert all_reproducible, "Reproducibility test failed"


if __name__ == "__main__":
    # Test the reproducibility utilities
    print("🧪 Testing reproducibility utilities...")
    
    # Test with a specific seed
    test_reproducibility(seed=42, num_steps=5)
    
    # Test global seed setting
    print("\n🌱 Testing global seed setting...")
    set_global_seed(123)
    
    # Test deterministic config
    print("\n🔧 Testing deterministic configuration...")
    apply_deterministic_config()
    
    print("\n✅ Reproducibility utilities test completed!")
