# standard library imports
from typing import Mapping, Sequence

# third-party imports
import gymnasium
import numpy as np


class AffineObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shift, scale):
        super().__init__(env)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        self.shift = shift
        self.scale = scale
        self.observation_space = gymnasium.spaces.Box(self.observation(env.observation_space.low), self.observation(env.observation_space.high), dtype=env.observation_space.dtype)

    def observation(self, obs):
        return (obs + self.shift) * self.scale


class Float64ToFloat32(gymnasium.ObservationWrapper):
    """Converts np.float64 arrays in the observations to np.float32 arrays."""

    # TODO: change observation/action spaces to correct dtype
    def observation(self, observation):
        observation = deepmap({np.ndarray: float64_to_float32,
                               float: float_to_float32,
                               np.float32: float_to_float32,
                               np.float64: float_to_float32}, observation)
        return observation

    def step(self, action):
        s, r, d, t, info = super().step(action)
        return s, r, d, t, info


# === Utilities ========================================================================================================


def deepmap(f, m):
    """Apply functions to the leaves of a dictionary or list, depending type of the leaf value."""
    for cls in f:
        if isinstance(m, cls):
            return f[cls](m)
    if isinstance(m, Sequence):
        return type(m)(deepmap(f, x) for x in m)
    elif isinstance(m, Mapping):
        return type(m)((k, deepmap(f, m[k])) for k in m)
    else:
        raise AttributeError(f"m is a {type(m)}, not a Sequence nor a Mapping: {m}")


def float64_to_float32(x):
    return np.asarray([x, ], np.float32) if x.dtype == np.float64 else x


def float_to_float32(x):
    return np.asarray([x, ], np.float32)
