# standard library imports
from dataclasses import InitVar, dataclass

# third-party imports
import gym

# local imports
from tmrl.wrappers import (AffineObservationWrapper, Float64ToFloat32)


class GenericGymEnv(gym.Wrapper):
    def __init__(self, id: str = "Pendulum-v0", obs_scale: float = 0., gym_kwargs={}):
        """
        Use this wrapper when using the framework with arbitrary environments
        :param id: gym id
        :param obs_scale: change this if wanting to rescale actions by a scalar
        :param gym_kwargs: keyword arguments of the gym environment
            (i.e. between -1.0 and 1.0 when the actual action space is something else)
        """
        env = gym.make(id, **gym_kwargs)
        if obs_scale:
            env = AffineObservationWrapper(env, 0, obs_scale)
        env = Float64ToFloat32(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        # env = NormalizeActionWrapper(env)
        super().__init__(env)


if __name__ == '__main__':
    pass
