"""
Tests for reproducibility controls.

This module tests that the same seed produces identical results across
different random number generators and environment interactions.
"""

import pytest
import numpy as np
import tensorflow as tf
import gymnasium as gym
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add project root to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.seed import (
    set_global_seed,
    set_gymnasium_seed,
    test_reproducibility,
    create_reproducible_rng,
)


class TestReproducibility:
    """Test reproducibility utilities."""

    def test_seed_utility_functions(self):
        """Test that seed utility functions work correctly."""
        # Test global seed setting
        set_global_seed(42)

        # Test reproducibility test
        test_reproducibility(seed=42, num_steps=5)  # Should not raise exception

    def test_python_random_reproducibility(self):
        """Test that Python's random module is reproducible."""
        import random

        # Test with same seed
        random.seed(42)
        values1 = [random.random() for _ in range(10)]

        random.seed(42)
        values2 = [random.random() for _ in range(10)]

        assert values1 == values2, "Python random should be reproducible with same seed"

    def test_numpy_random_reproducibility(self):
        """Test that NumPy random is reproducible."""
        # Test with same seed
        np.random.seed(42)
        values1 = [np.random.random() for _ in range(10)]

        np.random.seed(42)
        values2 = [np.random.random() for _ in range(10)]

        assert all(np.allclose(a, b) for a, b in zip(values1, values2)), \
            "NumPy random should be reproducible with same seed"

    def test_tensorflow_random_reproducibility(self):
        """Test that TensorFlow random is reproducible."""
        # Test with same seed
        tf.random.set_seed(42)
        values1 = [tf.random.normal([]).numpy() for _ in range(10)]

        tf.random.set_seed(42)
        values2 = [tf.random.normal([]).numpy() for _ in range(10)]

        assert all(np.allclose(a, b) for a, b in zip(values1, values2)), \
            "TensorFlow random should be reproducible with same seed"

    def test_reproducible_rng(self):
        """Test the create_reproducible_rng function."""
        rng1 = create_reproducible_rng(42)
        rng2 = create_reproducible_rng(42)

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]

        assert all(np.allclose(a, b) for a, b in zip(values1, values2)), \
            "Reproducible RNG should produce same values with same seed"

    def test_gymnasium_environment_reproducibility(self):
        """Test that Gymnasium environments are reproducible with same seed."""
        # Create two identical environments
        env1 = gym.make("CartPole-v1")
        env2 = gym.make("CartPole-v1")

        # Seed both environments
        seed = 42
        env1, actual_seed1 = set_gymnasium_seed(env1, seed)
        env2, actual_seed2 = set_gymnasium_seed(env2, seed)

        # Reset both environments
        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)

        # Take several steps and compare
        for _ in range(10):
            action = env1.action_space.sample()  # Same action for both

            obs1, reward1, terminated1, truncated1, info1 = env1.step(action)
            obs2, reward2, terminated2, truncated2, info2 = env2.step(action)

            # Check that observations and rewards are identical
            assert np.allclose(obs1, obs2), "Observations should be identical at step {_}"
            assert reward1 == reward2, "Rewards should be identical at step {_}"
            assert terminated1 == terminated2, "Termination should be identical at step {_}"
            assert truncated1 == truncated2, "Truncation should be identical at step {_}"

            if terminated1 or truncated1:
                break

        env1.close()
        env2.close()

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        # Test Python random
        import random
        random.seed(42)
        values1 = [random.random() for _ in range(10)]

        random.seed(123)
        values2 = [random.random() for _ in range(10)]

        assert values1 != values2, "Different seeds should produce different results"

    def test_global_seed_setting(self):
        """Test that global seed setting affects all libraries."""
        # Set global seed
        set_global_seed(42)

        # Get values from different libraries
        import random
        python_val = random.random()
        numpy_val = np.random.random()
        tf_val = tf.random.normal([]).numpy()

        # Reset and set same seed again
        set_global_seed(42)

        # Get values again
        python_val2 = random.random()
        numpy_val2 = np.random.random()
        tf_val2 = tf.random.normal([]).numpy()

        # Check reproducibility
        assert python_val == python_val2, "Python random should be reproducible"
        assert np.allclose(numpy_val, numpy_val2), "NumPy random should be reproducible"
        assert np.allclose(tf_val, tf_val2), "TensorFlow random should be reproducible"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
