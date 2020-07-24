from collections import Sequence, Mapping, deque

import gym
import numpy as np


class RealTimeWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
    # self.initial_action = env.action_space.sample()
    assert isinstance(env.action_space, gym.spaces.Box)
    self.initial_action = env.action_space.high * 0

  def reset(self):
    self.previous_action = self.initial_action
    return super().reset(), self.previous_action

  def step(self, action):
    observation, reward, done, info = super().step(self.previous_action)
    self.previous_action = action
    return (observation, action), reward, done, info


class PreviousActionWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
    # self.initial_action = env.action_space.sample()
    assert isinstance(env.action_space, gym.spaces.Box)
    self.initial_action = env.action_space.high * 0

  def reset(self):
    self.previous_action = self.initial_action
    return super().reset(), self.previous_action

  def step(self, action):
    observation, reward, done, info = super().step(action)  # this line is different from RealTimeWrapper
    self.previous_action = action
    return (observation, action), reward, done, info


class StatsWrapper(gym.Wrapper):
  """Compute running statistics (return, number of episodes, etc.) over a certain time window."""

  def __init__(self, env, window=100):
    super().__init__(env)
    self.reward_hist = deque([0], maxlen=window + 1)
    self.done_hist = deque([1], maxlen=window + 1)
    self.total_steps = 0

  def reset(self, **kwargs):
    return super().reset(**kwargs)

  def step(self, action):
    m, r, d, info = super().step(action)
    self.reward_hist.append(r)
    self.done_hist.append(d)
    self.total_steps += 1
    return m, r, d, info

  def stats(self):
    returns = [0]
    steps = [0]
    for reward, done in zip(self.reward_hist, self.done_hist):
      returns[-1] += reward
      steps[-1] += 1
      if done:
        returns.append(0)
        steps.append(0)
    returns = returns[1:-1]  # first and last episodes are incomplete
    steps = steps[1:-1]

    return dict(
      episodes=len(returns),
      episode_length=np.mean(steps) if len(steps) else np.nan,
      returns=np.mean(returns) if len(returns) else np.nan,
      average_reward=np.mean(tuple(self.reward_hist)[1:]),
    )


class DictObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env, key='vector'):
    super().__init__(env)
    self.key = key
    self.observation_space = gym.spaces.Dict({self.key: env.observation_space})

  def observation(self, observation):
    return {self.key: observation}


class TupleObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.observation_space = gym.spaces.Tuple((env.observation_space,))

  def observation(self, observation):
    return observation,


class DictActionWrapper(gym.Wrapper):
  def __init__(self, env, key='value'):
    super().__init__(env)
    self.key = key
    self.action_space = gym.spaces.Dict({self.key: env.action_space})

  def step(self, action: dict):
    return self.env.step(action['value'])


class AffineObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env, shift, scale):
    super().__init__(env)
    assert isinstance(env.observation_space, gym.spaces.Box)
    self.shift = shift
    self.scale = scale
    self.observation_space = gym.spaces.Box(self.observation(env.observation_space.low), self.observation(env.observation_space.high), dtype=env.observation_space.dtype)

  def observation(self, obs):
    return (obs + self.shift) * self.scale


class AffineRewardWrapper(gym.RewardWrapper):
  def __init__(self, env, shift, scale):
    super().__init__(env)
    self.shift = shift
    self.scale = scale

  def reward(self, reward):
    return (reward + self.shift) / self.scale


class NormalizeActionWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.scale = env.action_space.high - env.action_space.low
    self.shift = env.action_space.low
    self.action_space = gym.spaces.Box(-np.ones_like(self.shift), np.ones_like(self.shift), dtype=env.action_space.dtype)

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

  def step(self, action):
    action = action / 2 + 0.5  # 0 < a < 1
    action = action * self.scale + self.shift
    return self.env.step(action)


class TimeLimitResetWrapper(gym.Wrapper):
  """Adds a `reset` key to `info` that indicates whether an episode was ended just because of a time limit.
  This can be important as running out of time, should usually not be considered a "true" terminal state."""

  def __init__(self, env, max_steps=None, key='reset'):
    super().__init__(env)
    self.reset_key = key
    from gym.wrappers import TimeLimit
    self.enforce = bool(max_steps)
    if max_steps is None:
      tl = get_wrapper_by_class(env, TimeLimit)
      max_steps = 1 << 31 if tl is None else tl._max_episode_steps
      # print("TimeLimitResetWrapper.max_steps =", max_steps)

    self.max_steps = max_steps
    self.t = 0

  def reset(self, **kwargs):
    m = self.env.reset(**kwargs)
    self.t = 0
    return m

  def step(self, action):
    m, r, d, info = self.env.step(action)

    reset = (self.t == self.max_steps - 1) or info.get(self.reset_key, False)
    if not self.enforce:
      if reset:
        assert d, f"something went wrong t={self.t}, max_steps={self.max_steps}, info={info}"
    else:
      d = d or reset
    info = {**info, self.reset_key: reset}
    self.t += 1
    return m, r, d, info


class Float64ToFloat32(gym.ObservationWrapper):
  """Converts np.float64 arrays in the observations to np.float32 arrays."""

  # TODO: change observation/action spaces to correct dtype
  def observation(self, observation):
    observation = deepmap({np.ndarray: float64_to_float32}, observation)
    return observation

  def step(self, action):
    s, r, d, info = super().step(action)
    return s, r, d, info


# === Utilities ========================================================================================================

def get_wrapper_by_class(env, cls):
  if isinstance(env, cls):
    return env
  elif isinstance(env, gym.Wrapper):
    return get_wrapper_by_class(env.env, cls)


def deepmap(f, m):
  """Apply functions to the leaves of a dictionary or list, depending type of the leaf value.
  Example: deepmap({torch.Tensor: lambda t: t.detach()}, x)."""
  for cls in f:
    if isinstance(m, cls):
      return f[cls](m)
  if isinstance(m, Sequence):
    return type(m)(deepmap(f, x) for x in m)
  elif isinstance(m, Mapping):
    return type(m)((k, deepmap(f, m[k])) for k in m)
  else:
    raise AttributeError()


def float64_to_float32(x):
    return np.asarray(x, np.float32) if x.dtype == np.float64 else x
