"""
Configuration constants for curiosity-driven RL training.

This module contains all hyperparameters and configuration options used throughout
the training pipeline.
"""
from typing import Dict, Any

# Core RL hyperparameters
constants: Dict[str, Any] = {
    # Discount factor for rewards
    'GAMMA': 0.99,
    
    # Lambda for Generalized Advantage Estimation: https://arxiv.org/abs/1506.02438
    'LAMBDA': 1.0,
    
    # Entropy regularization constant
    'ENTROPY_BETA': 0.01,
    
    # Number of 'local steps': the number of timesteps we run the policy 
    # before we update the parameters. The larger local steps is, the lower 
    # is the variance in our policy gradients estimate on the one hand; but 
    # on the other hand, we get less frequent parameter updates, which slows 
    # down learning. In this code, we found that making local steps be much
    # smaller than 20 makes the algorithm more difficult to tune and to get to work.
    'ROLLOUT_MAXLEN': 20,
    
    # Gradient norm clipping
    'GRAD_NORM_CLIP': 40.0,
    
    # Reward value clipping in [-x,x]
    'REWARD_CLIP': 1.0,
    
    # Total steps taken across all workers
    'MAX_GLOBAL_STEPS': 100000000,
    
    # Learning rate for adam optimizer
    'LEARNING_RATE': 1e-4,

    # Weight of prediction bonus (curiosity)
    'PREDICTION_BETA': 0.01,
    # Set 0.5 for unsup=state
    
    # Scale lr of predictor wrt to policy network
    'PREDICTION_LR_SCALE': 10.0,
    # Set 30-50 for unsup=state
    
    # Should be between [0,1]
    'FORWARD_LOSS_WT': 0.2,
    # predloss = ( (1-FORWARD_LOSS_WT) * inv_loss + FORWARD_LOSS_WT * forward_loss) * PREDICTION_LR_SCALE
    
    # Number of global steps after which we start backpropagating to policy
    'POLICY_NO_BACKPROP_STEPS': 0,
}

# Environment-specific configurations
ENV_CONFIGS = {
    'doom': {
        'PREDICTION_BETA': 0.01,
        'ENTROPY_BETA': 0.01,
        'FRAME_SKIP': 4,
        'OBS_SHAPE': (42, 42),
    },
    'mario': {
        'PREDICTION_BETA': 0.2,
        'ENTROPY_BETA': 0.0005,
        'FRAME_SKIP': 6,
        'OBS_SHAPE': (42, 42),
    },
    'atari': {
        'PREDICTION_BETA': 0.01,
        'ENTROPY_BETA': 0.01,
        'FRAME_SKIP': 4,
        'OBS_SHAPE': (42, 42),
    },
    'montezuma': {
        'PREDICTION_BETA': 0.1,
        'ENTROPY_BETA': 0.01,
        'FRAME_SKIP': 4,
        'OBS_SHAPE': (42, 42),
    },
}

# Logging configuration
LOGGING_CONFIG = {
    'USE_WANDB': True,
    'USE_TENSORBOARD': True,
    'LOG_INTERVAL': 100,
    'SAVE_INTERVAL': 1000,
    'EVAL_INTERVAL': 5000,
}

# Hard exploration Atari games
HARD_EXPLORATION_GAMES = [
    'MontezumaRevenge-v5',
    'Pitfall-v5',
    'PrivateEye-v5', 
    'Solaris-v5',
    'Venture-v5',
    'Frostbite-v5',
    'Freeway-v5',
    'Gravitar-v5',
    'Qbert-v5',
    'Seaquest-v5',
    'SpaceInvaders-v5',
]

def get_env_config(env_id: str) -> Dict[str, Any]:
    """Get environment-specific configuration.
    
    Args:
        env_id: Environment identifier
        
    Returns:
        Environment-specific configuration dictionary
    """
    env_type = 'atari'  # default
    
    if 'doom' in env_id.lower():
        env_type = 'doom'
    elif 'mario' in env_id.lower():
        env_type = 'mario'
    elif any(game.lower() in env_id.lower() for game in HARD_EXPLORATION_GAMES):
        env_type = 'montezuma'
    
    return ENV_CONFIGS.get(env_type, ENV_CONFIGS['atari'])

def update_constants_for_env(env_id: str) -> None:
    """Update global constants based on environment.
    
    Args:
        env_id: Environment identifier
    """
    env_config = get_env_config(env_id)
    
    # Update constants with environment-specific values
    for key, value in env_config.items():
        if key in constants:
            constants[key] = value