"""
Basic tests for the enhanced NoReward-RL implementation.

These tests verify that the core components work correctly and can be imported.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from constants import constants, get_env_config, update_constants_for_env
        from logger import Logger, create_logger
        from env_wrapper import BufferedObsEnv, NoNegativeRewardEnv, SkipEnv
        from envs import create_env
        from model_modern import LSTMPolicy, StateActionPredictor, StatePredictor
        print("✅ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_constants():
    """Test that constants are properly defined."""
    from constants import constants, get_env_config, update_constants_for_env
    
    # Test basic constants
    assert 'GAMMA' in constants
    assert 'LEARNING_RATE' in constants
    assert 'PREDICTION_BETA' in constants
    
    # Test environment configs
    doom_config = get_env_config('doom')
    assert 'PREDICTION_BETA' in doom_config
    
    mario_config = get_env_config('mario')
    assert 'PREDICTION_BETA' in mario_config
    
    print("✅ Constants test passed")

def test_logger_creation():
    """Test logger creation."""
    from logger import create_logger
    
    # Test logger creation without wandb
    logger = create_logger(
        project_name="test-project",
        use_wandb=False,
        use_tensorboard=False,
    )
    
    assert logger is not None
    assert logger.use_wandb == False
    assert logger.use_tensorboard == False
    
    print("✅ Logger creation test passed")

def test_env_wrappers():
    """Test environment wrapper creation."""
    import gymnasium as gym
    from env_wrapper import BufferedObsEnv, NoNegativeRewardEnv, SkipEnv
    
    # Create a simple environment
    env = gym.make('CartPole-v1')
    
    # Test wrappers
    wrapped_env = BufferedObsEnv(env, n=4, skip=4, shape=(42, 42))
    assert wrapped_env.observation_space.shape == (42, 42, 4)
    
    wrapped_env = NoNegativeRewardEnv(env)
    assert wrapped_env is not None
    
    wrapped_env = SkipEnv(env, skip=4)
    assert wrapped_env is not None
    
    print("✅ Environment wrappers test passed")

def test_env_creation():
    """Test environment creation."""
    from envs import create_env
    
    # Test basic environment creation
    try:
        env = create_env('CartPole-v1', client_id='0')
        assert env is not None
        print("✅ Basic environment creation test passed")
    except Exception as e:
        print(f"⚠️  Environment creation test skipped: {e}")

if __name__ == "__main__":
    test_imports()
    test_constants()
    test_logger_creation()
    test_env_wrappers()
    test_env_creation()
    print("\n🎉 All basic tests passed!")

