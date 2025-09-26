#!/usr/bin/env python3
"""
Simple demo script to test the refactored NoReward-RL implementation.

This script demonstrates the key features without requiring complex training setup.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from constants import constants, get_env_config
        from logger import create_logger
        from env_wrapper import BufferedObsEnv, NoNegativeRewardEnv, SkipEnv
        from envs import create_env
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_environment_creation():
    """Test environment creation."""
    print("\nTesting environment creation...")
    try:
        from envs import create_env
        
        # Test basic environment
        env = create_env('CartPole-v1', client_id='0')
        print(f"✅ Created environment: {env}")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test environment step
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Initial observation shape: {obs.shape}")
        
        # Test environment step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

def test_logger():
    """Test logger creation."""
    print("\nTesting logger creation...")
    try:
        from logger import create_logger
        
        # Test logger without wandb
        logger = create_logger(
            project_name="test-project",
            use_wandb=False,
            use_tensorboard=False,
        )
        print(f"✅ Logger created successfully")
        print(f"   Use wandb: {logger.use_wandb}")
        print(f"   Use tensorboard: {logger.use_tensorboard}")
        
        # Test logging
        logger.log_scalar("test/metric", 1.0, step=0)
        print("✅ Logging test successful")
        
        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        return False

def test_environment_wrappers():
    """Test environment wrappers."""
    print("\nTesting environment wrappers...")
    try:
        import gymnasium as gym
        from env_wrapper import BufferedObsEnv, NoNegativeRewardEnv, SkipEnv
        
        # Create base environment
        env = gym.make('CartPole-v1')
        
        # Test BufferedObsEnv
        wrapped_env = BufferedObsEnv(env, n=4, skip=4, shape=(42, 42))
        print(f"✅ BufferedObsEnv created")
        print(f"   Observation space: {wrapped_env.observation_space.shape}")
        
        # Test NoNegativeRewardEnv
        reward_env = NoNegativeRewardEnv(env)
        print(f"✅ NoNegativeRewardEnv created")
        
        # Test SkipEnv
        skip_env = SkipEnv(env, skip=4)
        print(f"✅ SkipEnv created")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Environment wrappers test failed: {e}")
        return False

def test_constants():
    """Test constants and configuration."""
    print("\nTesting constants and configuration...")
    try:
        from constants import constants, get_env_config, HARD_EXPLORATION_GAMES
        
        print(f"✅ Constants loaded: {len(constants)} parameters")
        print(f"   GAMMA: {constants['GAMMA']}")
        print(f"   LEARNING_RATE: {constants['LEARNING_RATE']}")
        print(f"   PREDICTION_BETA: {constants['PREDICTION_BETA']}")
        
        # Test environment configs
        doom_config = get_env_config('doom')
        print(f"✅ Doom config: {doom_config}")
        
        mario_config = get_env_config('mario')
        print(f"✅ Mario config: {mario_config}")
        
        print(f"✅ Hard exploration games: {len(HARD_EXPLORATION_GAMES)}")
        print(f"   Examples: {HARD_EXPLORATION_GAMES[:3]}")
        
        return True
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced NoReward-RL Implementation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_constants,
        test_logger,
        test_environment_wrappers,
        test_environment_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored implementation is working correctly.")
        print("\nNext steps:")
        print("1. Install additional dependencies for specific environments:")
        print("   - VizDoom: pip install vizdoomgym")
        print("   - Mario: pip install gym-super-mario-bros")
        print("   - Atari: pip install gymnasium[atari]")
        print("2. Try training with: python3 src/train_modern.py --env-id CartPole-v1")
        print("3. Try evaluation with: python3 scripts/eval_and_record.py --env-id CartPole-v1 --model-path dummy")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

