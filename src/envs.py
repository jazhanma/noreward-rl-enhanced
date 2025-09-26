"""
Environment creation and configuration for curiosity-driven RL.

Supports multiple environments including Doom, Mario, Atari, and hard-exploration games.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from PIL import Image

import env_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_env(
    env_id: str,
    client_id: str,
    remotes: Optional[List[str]] = None,
    **kwargs: Any
) -> gym.Env:
    """Create and configure an environment.
    
    Args:
        env_id: Environment identifier
        client_id: Client identifier for distributed training
        remotes: Remote environment addresses
        **kwargs: Additional environment configuration
        
    Returns:
        Configured gymnasium environment
    """
    if "doom" in env_id.lower() or "labyrinth" in env_id.lower():
        return create_doom(env_id, client_id, **kwargs)
    if "mario" in env_id.lower():
        return create_mario(env_id, client_id, **kwargs)
    if "montezuma" in env_id.lower() or "pitfall" in env_id.lower():
        return create_hard_exploration_atari(env_id, **kwargs)

    # Try to create as gymnasium environment
    try:
        env = gym.make(env_id)
        if hasattr(env, "tags") and env.tags.get("atari", False):
            return create_atari_env(env_id, **kwargs)
        return env
    except gym.error.Error:
        logger.warning(f"Could not create environment {env_id} with gymnasium, trying legacy gym")
        # Fallback to legacy gym if needed
        import gym as legacy_gym
        env = legacy_gym.make(env_id)
        return create_atari_env(env_id, **kwargs)


def create_doom(
    env_id: str,
    client_id: str,
    env_wrap: bool = True,
    record: bool = False,
    outdir: Optional[str] = None,
    no_life_reward: bool = False,
    ac_repeat: int = 0,
    **_kwargs: Any,
) -> gym.Env:
    """Create VizDoom environment.
    
    Args:
        env_id: Doom environment identifier
        client_id: Client identifier
        env_wrap: Whether to apply environment wrappers
        record: Whether to record episodes
        outdir: Output directory for recordings
        no_life_reward: Whether to remove negative rewards
        ac_repeat: Action repeat count
        
    Returns:
        Configured Doom environment
    """
    try:
        import vizdoomgym
    except ImportError:
        logger.error("vizdoomgym not installed. Install with: pip install vizdoomgym")
        raise

    # Map environment IDs to VizDoom scenarios
    doom_env_map = {
        "doom": "VizdoomMyWayHome-v0",
        "doomsparse": "VizdoomMyWayHomeSparse-v0", 
        "doomverysparse": "VizdoomMyWayHomeVerySparse-v0",
        "doomdense": "VizdoomMyWayHomeDense-v0",
    }
    
    actual_env_id = doom_env_map.get(env_id.lower(), "VizdoomMyWayHome-v0")
    
    # VizDoom workaround: Simultaneously launching multiple vizdoom processes
    # makes program stuck, so use delays in multi-threading/processing
    client_id_int = int(client_id)
    time.sleep(client_id_int * 10)
    
    env = gym.make(actual_env_id)
    
    if record and outdir is not None:
        env = gym.wrappers.RecordVideo(env, outdir, episode_trigger=lambda x: True)

    if env_wrap:
        fshape = (42, 42)
        frame_skip = ac_repeat if ac_repeat > 0 else 4
        env.seed(None)
        if no_life_reward:
            env = env_wrapper.NoNegativeRewardEnv(env)
        env = env_wrapper.BufferedObsEnv(env, skip=frame_skip, shape=fshape)
        env = env_wrapper.SkipEnv(env, skip=frame_skip)
    elif no_life_reward:
        env = env_wrapper.NoNegativeRewardEnv(env)

    env = DiagnosticsInfo(env)
    return env


def create_mario(
    env_id: str,
    client_id: str,
    env_wrap: bool = True,
    record: bool = False,
    outdir: Optional[str] = None,
    no_life_reward: bool = False,
    ac_repeat: int = 0,
    **_kwargs: Any,
) -> gym.Env:
    """Create Super Mario Bros environment.
    
    Args:
        env_id: Mario environment identifier
        client_id: Client identifier
        env_wrap: Whether to apply environment wrappers
        record: Whether to record episodes
        outdir: Output directory for recordings
        no_life_reward: Whether to remove negative rewards
        ac_repeat: Action repeat count
        
    Returns:
        Configured Mario environment
    """
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    except ImportError:
        logger.error("gym_super_mario_bros not installed. Install with: pip install gym_super_mario_bros")
        raise

    if "-v" in env_id.lower():
        env_id = env_id
    else:
        env_id = "SuperMarioBros-v0"  # shape: (224,256,3)=(h,w,c)

    # Mario workaround: Simultaneously launching multiple processes makes program stuck,
    # so use delays in multi-threading/multi-processing
    client_id_int = int(client_id)
    time.sleep(client_id_int * 50)
    
    env = gym.make(env_id)
    env = env_wrapper.MarioEnv(env)

    if record and outdir is not None:
        env = gym.wrappers.RecordVideo(env, outdir, episode_trigger=lambda x: True)

    if env_wrap:
        frame_skip = ac_repeat if ac_repeat > 0 else 6
        fshape = (42, 42)
        env.seed(None)
        if no_life_reward:
            env = env_wrapper.NoNegativeRewardEnv(env)
        env = env_wrapper.BufferedObsEnv(
            env, skip=frame_skip, shape=fshape, max_frames=False
        )
        if frame_skip > 1:
            env = env_wrapper.SkipEnv(env, skip=frame_skip)
    elif no_life_reward:
        env = env_wrapper.NoNegativeRewardEnv(env)

    env = DiagnosticsInfo(env)
    return env


def create_hard_exploration_atari(
    env_id: str,
    record: bool = False,
    outdir: Optional[str] = None,
    **_kwargs: Any,
) -> gym.Env:
    """Create hard exploration Atari environment.
    
    Args:
        env_id: Atari environment identifier
        record: Whether to record episodes
        outdir: Output directory for recordings
        
    Returns:
        Configured Atari environment
    """
    # Hard exploration Atari games
    hard_exploration_games = [
        "MontezumaRevenge-v5",
        "Pitfall-v5", 
        "PrivateEye-v5",
        "Solaris-v5",
        "Venture-v5",
        "Frostbite-v5",
        "Freeway-v5",
        "Gravitar-v5",
        "Qbert-v5",
        "Seaquest-v5",
        "SpaceInvaders-v5",
    ]
    
    if env_id.lower() in [game.lower() for game in hard_exploration_games]:
        actual_env_id = env_id
    else:
        # Default to Montezuma's Revenge
        actual_env_id = "MontezumaRevenge-v5"
    
    env = gym.make(actual_env_id)
    
    if record and outdir is not None:
        env = gym.wrappers.RecordVideo(env, outdir, episode_trigger=lambda x: True)
    
    # Apply Atari preprocessing
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    
    # Apply frame stacking
    env = gym.wrappers.FrameStack(env, num_stack=4)
    
    # Apply observation rescaling
    env = env_wrapper.AtariRescale42x42(env)
    
    env = DiagnosticsInfo(env)
    return env


def create_atari_env(
    env_id: str,
    record: bool = False,
    outdir: Optional[str] = None,
    **_kwargs: Any,
) -> gym.Env:
    """Create standard Atari environment.
    
    Args:
        env_id: Atari environment identifier
        record: Whether to record episodes
        outdir: Output directory for recordings
        
    Returns:
        Configured Atari environment
    """
    env = gym.make(env_id)
    
    if record and outdir is not None:
        env = gym.wrappers.RecordVideo(env, outdir, episode_trigger=lambda x: True)
    
    # Apply Atari preprocessing
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    
    # Apply frame stacking
    env = gym.wrappers.FrameStack(env, num_stack=4)
    
    # Apply observation rescaling
    env = env_wrapper.AtariRescale42x42(env)
    
    env = DiagnosticsInfo(env)
    return env


class DiagnosticsInfo(gym.Wrapper):
    """Diagnostic information wrapper for logging."""

    def __init__(self, env: gym.Env, log_interval: int = 503):
        super().__init__(env)
        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._last_episode_id = -1

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset environment and logging."""
        logger.info("Resetting environment logs")
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Step environment and collect diagnostics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            to_log["diagnostics/fps"] = fps

        if reward is not None:
            self._episode_reward += reward
            if obs is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if terminated or truncated:
            logger.info(
                "Episode finished: reward=%s length=%s",
                self._episode_reward,
                self._episode_length,
            )
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        # Add environment-specific info
        if "distance" in info:
            to_log["distance"] = info["distance"]  # mario
        if "POSITION_X" in info:  # doom
            to_log["POSITION_X"] = info["POSITION_X"]
            to_log["POSITION_Y"] = info["POSITION_Y"]

        info.update(to_log)
        return obs, reward, terminated, truncated, info