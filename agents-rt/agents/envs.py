import atexit
import os
from dataclasses import dataclass, InitVar
import gym
from gym.wrappers import TimeLimit

from agents.wrappers import Float64ToFloat32, TimeLimitResetWrapper, NormalizeActionWrapper, RealTimeWrapper, TupleObservationWrapper, AffineObservationWrapper, AffineRewardWrapper, PreviousActionWrapper, FrameSkip, get_wrapper_by_class
from agents.wrappers_rd import RandomDelayWrapper
import numpy as np
import pickle
from agents.batch_env import get_env_state


def mujoco_py_issue_424_workaround():
    """Mujoco_py generates files in site-packages for some reason.
    It causes trouble with docker and during runtime.
    https://github.com/openai/mujoco-py/issues/424
    """
    import os
    from os.path import dirname, join
    from shutil import rmtree
    import pkgutil
    path = join(dirname(pkgutil.get_loader("mujoco_py").path), "generated")
    [os.remove(join(path, name)) for name in os.listdir(path) if name.endswith("lock")]


class Env(gym.Wrapper):
    """Environment class wrapping gym.Env that automatically resets and stores the last transition"""

    def __init__(self, env, store_env=False):
        super().__init__(env)
        self.transition = (self.reset(), 0., True, {})
        self.store_env = store_env

    def reset(self):
        return self.observation(self.env.reset())

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.reset() if done else self.observation(next_state)
        self.transition = next_state, reward, done, info

        if self.store_env:
            info['env_state'] = pickle.dumps(get_env_state(self))

        return self.transition

    def observation(self, observation):
        return observation


class GymEnv(Env):
    def __init__(self, seed_val=0, id: str = "Pendulum-v0", real_time: bool = False, frame_skip: int = 0, obs_scale: float = 0., store_env: bool = False):
        env = gym.make(id)

        if obs_scale:
            env = AffineObservationWrapper(env, 0, obs_scale)

        if frame_skip:
            original_frame_skip = getattr(env.unwrapped, 'frame_skip', 1)  # on many Mujoco environments this is 5
            # print("Original frame skip", original_frame_skip)

            # I think the two lines below were actually a mistake after all (at least for HalfCheetah)
            # if hasattr(env, 'dt'):
            #   env.dt = env.dt  # in case this is an attribute we fix it to its orignal value to not distort rewards (see
            #   halfcheetah.py)
            env.unwrapped.frame_skip = 1
            tl = get_wrapper_by_class(env, TimeLimit)
            tl._max_episode_steps = int(tl._max_episode_steps * original_frame_skip)
            # print("New max episode steps", env._max_episode_steps)
            env = FrameSkip(env, frame_skip, 1 / original_frame_skip)

        env = Float64ToFloat32(env)
        # env = TimeLimitResetWrapper(env)  # obsolete
        assert isinstance(env.action_space, gym.spaces.Box)
        env = NormalizeActionWrapper(env)
        if real_time:
            env = RealTimeWrapper(env)
        else:
            env = TupleObservationWrapper(env)

        super().__init__(env, store_env=store_env)

        # self.seed(seed_val)


class UntouchedGymEnv(gym.Wrapper):
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


class AvenueEnv(Env):
    def __init__(self, seed_val=0, id: str = "RaceSolo-v0", real_time: bool = False, width: int = 256, height: int = 64):
        import avenue
        env = avenue.make(id, width=width, height=height)
        assert isinstance(env.action_space, gym.spaces.Box)
        env = NormalizeActionWrapper(env)
        if real_time:
            env = RealTimeWrapper(env)
        else:
            # Avenue environments are non-markovian. We don't want to give real-time methods an advantage by having the past action as part of it's state while non-real-time methods have not. I.e. we add the past action to the state below.
            env = PreviousActionWrapper(env)
        super().__init__(env)

        # bring images into right format: batch x channels x height x width
        (img_sp, vec_sp), *more = env.observation_space
        img_sp = gym.spaces.Box(img_sp.low.transpose(2, 0, 1), img_sp.high.transpose(2, 0, 1), dtype=img_sp.dtype)
        self.observation_space = gym.spaces.Tuple((gym.spaces.Tuple((img_sp, vec_sp)), *more))

    # self.seed(seed_val)

    def observation(self, observation):
        (img, vec), *more = observation
        return ((img.transpose(2, 0, 1), vec), *more)


class RandomDelayEnv(Env):
    def __init__(self,
                 seed_val=0, id: str = "Pendulum-v0",
                 frame_skip: int = 0,
                 min_observation_delay: int = 0,
                 sup_observation_delay: int = 8,
                 min_action_delay: int = 0,  # this is equivalent to a MIN of 1 in the paper
                 sup_action_delay: int = 2):  # this is equivalent to a MAX of 2 in the paper
        env = gym.make(id)

        if frame_skip:
            original_frame_skip = getattr(env.unwrapped, 'frame_skip', 1)  # on many Mujoco environments this is 5
            # print("Original frame skip", original_frame_skip)

            # I think the two lines below were actually a mistake after all (at least for HalfCheetah)
            # if hasattr(env, 'dt'):
            #   env.dt = env.dt  # in case this is an attribute we fix it to its orignal value to not distort rewards (see
            #   halfcheetah.py)
            env.unwrapped.frame_skip = 1
            tl = get_wrapper_by_class(env, TimeLimit)
            tl._max_episode_steps = int(tl._max_episode_steps * original_frame_skip)
            # print("New max episode steps", env._max_episode_steps)
            env = FrameSkip(env, frame_skip, 1 / original_frame_skip)

        env = Float64ToFloat32(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        env = NormalizeActionWrapper(env)
        env = RandomDelayWrapper(env, range(min_observation_delay, sup_observation_delay), range(min_action_delay, sup_action_delay))
        super().__init__(env)


def test_avenue():
    env = AvenueEnv(id="CityPedestrians-v0")
    env.reset()
    [env.step(env.action_space.sample()) for _ in range(1000)]
    (img,), _, _, _ = env.step(env.action_space.sample())
    assert img == 3
    print('done')


def test_random_delay_env():
    env = RandomDelayEnv()
    obs = env.reset()
    [env.step(env.action_space.sample()) for _ in range(1000)]
    obs, _, _, _ = env.step(env.action_space.sample())
    print('done')


if __name__ == '__main__':
    test_random_delay_env()
