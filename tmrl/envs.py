# standard library imports
from dataclasses import InitVar, dataclass

# third-party imports
import gymnasium

# local imports
from tmrl.wrappers import (AffineObservationWrapper, Float64ToFloat32)


__docformat__ = "google"


class GenericGymEnv(gymnasium.Wrapper):
    def __init__(self, id: str = "Pendulum-v0", gym_kwargs=None, obs_scale: float = 0., wrappers=None):
        """
        Use this wrapper when using the framework with arbitrary environments.

        Args:
            id (str): gymnasium id
            gym_kwargs (dict): keyword arguments of the gymnasium environment (i.e. between -1.0 and 1.0 when the actual action space is something else)
            obs_scale (float): change this if wanting to rescale actions by a scalar
            wrappers (list): list of tuples (gymnasium.Wrapper, args, kwargs)
        """
        if gym_kwargs is None:
            gym_kwargs = {}
        env = gymnasium.make(id, **gym_kwargs, disable_env_checker=True)
        if obs_scale:
            env = AffineObservationWrapper(env, 0, obs_scale)

        if wrappers is not None:
            for wrapper, args, kwargs in wrappers:
                env = wrapper(env, *args, **kwargs)

        # assert isinstance(env.action_space, gymnasium.spaces.Box), f"{env.action_space}"
        # env = NormalizeActionWrapper(env)
        super().__init__(env)


if __name__ == '__main__':
    pass
