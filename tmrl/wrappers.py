# standard library imports
from collections.abc import Mapping, Sequence

# third-party imports
import gymnasium
import numpy as np


class AffineObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shift, scale):
        super().__init__(env)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        self.shift = shift
        self.scale = scale
        self.observation_space = gymnasium.spaces.Box(
            self.observation(env.observation_space.low),
            self.observation(env.observation_space.high),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return (observation + self.shift) * self.scale


def _space_to_float32(space):
    """Return a copy of the space with float Box dtypes set to np.float32."""
    if isinstance(space, gymnasium.spaces.Box):
        if np.issubdtype(space.dtype, np.floating):
            return gymnasium.spaces.Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=np.float32,
            )
        return space
    if isinstance(space, gymnasium.spaces.Dict):
        return gymnasium.spaces.Dict({k: _space_to_float32(v) for k, v in space.spaces.items()})
    if isinstance(space, gymnasium.spaces.Tuple):
        return gymnasium.spaces.Tuple([_space_to_float32(s) for s in space.spaces])
    return space


class Float64ToFloat32(gymnasium.ObservationWrapper):
    """Converts np.float64 arrays in the observations to np.float32 arrays."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = _space_to_float32(env.observation_space)
        self.action_space = _space_to_float32(env.action_space)

    def observation(self, observation):
        observation = deepmap(
            {
                np.ndarray: float64_to_float32,
                float: float_to_float32,
                int: int_to_float32,
                np.float32: float_to_float32,
                np.float64: float_to_float32,
            },
            observation,
        )
        return observation

    def step(self, action):
        observation, reward, done, terminated, info = super().step(action)
        return observation, reward, done, terminated, info


# === Utilities ================================================================


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
    return (
        np.asarray(
            [
                x,
            ],
            np.float32,
        )
        if x.dtype == np.float64
        else x
    )


def float_to_float32(x):
    return np.asarray(
        [
            x,
        ],
        np.float32,
    )


def int_to_float32(x):
    return np.asarray(
        [
            float(x),
        ],
        np.float32,
    )
