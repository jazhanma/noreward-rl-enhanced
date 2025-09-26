"""
Environment wrappers for curiosity-driven RL.

Author: Deepak Pathak (original), Enhanced for gymnasium compatibility

Acknowledgement:
    - The wrappers (BufferedObsEnv, SkipEnv) were originally written by
        Evan Shelhamer and modified by Deepak. Thanks Evan!
    - This file is derived from
        https://github.com/shelhamer/ourl/envs.py
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers_deprecated.py
"""
from __future__ import annotations

import time
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from PIL import Image


class BufferedObsEnv(gym.ObservationWrapper):
    """Buffer observations and stack e.g. for frame skipping.

    Args:
        env: The environment to wrap
        n: The length of the buffer, and number of observations stacked
        skip: The number of steps between buffered observations (min=1)
        shape: The shape to resize observations to
        channel_last: Whether to stack along the last dimension
        max_frames: Whether to use max pooling across time steps

    Note:
        - First obs is the oldest, last obs is the newest
        - The buffer is zeroed out on reset
        - *must* call reset() for init!
    """

    def __init__(
        self,
        env: gym.Env,
        n: int = 4,
        skip: int = 4,
        shape: Tuple[int, int] = (84, 84),
        channel_last: bool = True,
        max_frames: bool = True,
    ):
        super().__init__(env)
        self.obs_shape = shape
        # Most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)
        self.max_frames = max_frames
        self.n = n
        self.skip = skip
        self.buffer = deque(maxlen=self.n)
        self.counter = 0  # init and reset should agree on this
        shape = shape + (n,) if channel_last else (n,) + shape
        self.observation_space = spaces.Box(0.0, 1.0, shape, dtype=np.float32)
        self.ch_axis = -1 if channel_last else 0
        self.scale = 1.0 / 255

    def observation(self, obs: ObsType) -> np.ndarray:
        """Process observation through the buffer."""
        obs = self._convert(obs)
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        obs_new = np.stack(self.buffer, axis=self.ch_axis)
        return obs_new.astype(np.float32) * self.scale

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs_buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.zeros_like(self._convert(obs)))
        self.buffer.append(self._convert(obs))
        obs_new = np.stack(self.buffer, axis=self.ch_axis)
        return obs_new.astype(np.float32) * self.scale, info

    def _convert(self, obs: ObsType) -> np.ndarray:
        """Convert observation to the required format."""
        self.obs_buffer.append(obs)
        if self.max_frames:
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        else:
            max_frame = obs
        intensity_frame = self._rgb2y(max_frame).astype(np.uint8)
        small_frame = np.array(
            Image.fromarray(intensity_frame).resize(
                self.obs_shape, resample=Image.BILINEAR
            ),
            dtype=np.uint8,
        )
        return small_frame

    def _rgb2y(self, im: np.ndarray) -> np.ndarray:
        """Convert an RGB image to a Y image (as in YUV).

        These coefficients are taken from the torch/image library.
        Beware: these are more critical than you might think, as the
        monochromatic contrast can be surprisingly low.
        """
        if len(im.shape) < 3:
            return im
        return np.sum(im * [0.299, 0.587, 0.114], axis=2)


class NoNegativeRewardEnv(gym.RewardWrapper):
    """Clip reward in negative direction."""

    def __init__(self, env: gym.Env, neg_clip: float = 0.0):
        super().__init__(env)
        self.neg_clip = neg_clip

    def reward(self, reward: float) -> float:
        """Clip negative rewards."""
        return self.neg_clip if reward < self.neg_clip else reward


class SkipEnv(gym.Wrapper):
    """Skip timesteps: repeat action, accumulate reward, take last obs."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Execute action for skip steps."""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for i in range(self.skip):
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            info.update(step_info)
            info["steps"] = i + 1
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class MarioEnv(gym.Wrapper):
    """Mario environment wrapper for faster resets."""

    def __init__(self, env: gym.Env, tiles_env: bool = False):
        """Reset mario environment without actually restarting fceux everytime.

        This speeds up unrolling by approximately 10 times.

        Args:
            env: The mario environment to wrap
            tiles_env: Whether to use tiles environment
        """
        super().__init__(env)
        self.reset_count = -1
        # Reward is distance travelled. So normalize it with total distance
        # https://github.com/ppaquette/gym-super-mario/blob/master/ppaquette_gym_super_mario/lua/super-mario-bros.lua
        # However, we will not use this reward at all. It is only for completion.
        self.max_distance = 3000.0
        self.tiles_env = tiles_env

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the mario environment."""
        if self.reset_count < 0:
            print("\nDoing hard mario fceux reset (40 seconds wait) !")
            sys.stdout.flush()
            obs, info = self.env.reset(seed=seed, options=options)
            time.sleep(40)

        obs, _, terminated, truncated, info = self.env.step(7)  # take right once to start game
        if info.get("ignore", False):  # assuming this happens only in beginning
            self.reset_count = -1
            self.env.close()
            return self.reset(seed=seed, options=options)

        self.reset_count = info.get("iteration", -1)
        if self.tiles_env:
            return obs, info
        return obs[24:-12, 8:-8, :], info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Execute action in mario environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Use iteration count to determine if episode is done
        done = info["iteration"] > self.reset_count
        reward = float(reward) / self.max_distance  # note: we do not use this rewards at all.

        if self.tiles_env:
            return obs, reward, done, done, info
        return obs[24:-12, 8:-8, :], reward, done, done, info

    def close(self) -> None:
        """Close the environment."""
        self.reset_count = -1
        return self.env.close()


class MakeEnvDynamic(gym.ObservationWrapper):
    """Make observation dynamic by adding noise."""

    def __init__(self, env: gym.Env, percent_pad: int = 5):
        super().__init__(env)
        self.orig_shape = env.observation_space.shape
        newside = int(round(max(self.orig_shape[:-1]) * 100.0 / (100.0 - percent_pad)))
        self.new_shape = [newside, newside, 3]
        self.observation_space = spaces.Box(0.0, 255.0, self.new_shape, dtype=np.uint8)
        self.bottom_ignore = 20  # doom 20px bottom is useless
        self.ob = None

    def observation(self, obs: ObsType) -> np.ndarray:
        """Add noise to observation."""
        im_noise = np.random.randint(0, 256, self.new_shape).astype(obs.dtype)
        im_noise[
            : self.orig_shape[0] - self.bottom_ignore, : self.orig_shape[1], :
        ] = obs[:-self.bottom_ignore, :, :]
        self.ob = im_noise
        return im_noise


class AtariRescale42x42(gym.ObservationWrapper):
    """Rescale Atari observations to 42x42."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, [42, 42, 1], dtype=np.float32)

    def observation(self, obs: ObsType) -> np.ndarray:
        """Rescale observation to 42x42."""
        return self._process_frame42(obs)

    def _process_frame42(self, frame: np.ndarray) -> np.ndarray:
        """Process frame to 42x42 format."""
        frame = frame[34 : 34 + 160, :160]
        # Resize by half, then down to 42x42 (essentially mipmapping). If
        # we resize directly we lose pixels that, when mapped to 42x42,
        # aren't close enough to the pixel boundary.
        frame = np.asarray(
            Image.fromarray(frame)
            .resize((80, 80), resample=Image.BILINEAR)
            .resize((42, 42), resample=Image.BILINEAR)
        )
        frame = frame.mean(2)  # take mean along channels
        frame = frame.astype(np.float32)
        frame *= 1.0 / 255.0
        frame = np.reshape(frame, [42, 42, 1])
        return frame
