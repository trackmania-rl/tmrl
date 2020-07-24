import atexit
import os
from dataclasses import dataclass, InitVar
import gym
from rtrl.wrappers import Float64ToFloat32, TimeLimitResetWrapper, NormalizeActionWrapper, RealTimeWrapper, \
  TupleObservationWrapper, AffineObservationWrapper, AffineRewardWrapper, PreviousActionWrapper
import numpy as np


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
  def __init__(self, env):
    super().__init__(env)
    self.transition = (self.reset(), 0., True, {})

  def reset(self):
    return self.observation(self.env.reset())

  def step(self, action):
    next_state, reward, done, info = self.env.step(action)
    next_state = self.reset() if done else self.observation(next_state)
    self.transition = next_state, reward, done, info
    return self.transition

  def observation(self, observation):
    return observation


class GymEnv(Env):
  def __init__(self, seed_val=0, id: str = "Pendulum-v0", real_time: bool = False):
    env = gym.make(id)
    env = Float64ToFloat32(env)
    env = TimeLimitResetWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    env = NormalizeActionWrapper(env)
    if real_time:
      env = RealTimeWrapper(env)
    else:
      env = TupleObservationWrapper(env)

    super().__init__(env)
    # self.seed(seed_val)


class UntouchedGymEnv(Env):
    def __init__(self, seed_val=0, id: str = "Pendulum-v0", obs_scale: float = 0.):
        env = gym.make(id)
        if obs_scale:
            env = AffineObservationWrapper(env, 0, obs_scale)
        env = Float64ToFloat32(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        env = NormalizeActionWrapper(env)
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


def test_avenue():
  env = AvenueEnv(id="CityPedestrians-v0")
  env.reset()
  [env.step(env.action_space.sample()) for _ in range(1000)]
  (img, ), _, _, _ = env.step(env.action_space.sample())
  print('fjdk')