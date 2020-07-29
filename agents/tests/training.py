from dataclasses import dataclass, InitVar

import gym
import numpy as np
import torch

from agents.sac_models import Mlp
from agents import *


class TestMlp(Mlp):
	def act(self, obs: torch.tensor, r, done, info):
		return obs.copy(), {}


@dataclass
class TestEnv(gym.Env):
	seed_val: InitVar[int]

	high: float = 10

	def __post_init__(self, seed_val):
		self.observation_space = gym.spaces.Box(0, self.high, shape=[1], dtype=np.float32)
		self.action_space = gym.spaces.Box(0, self.high, shape=[1], dtype=np.float32)

	def reset(self):
		self.state = np.zeros([1], np.float32)
		return self.state

	def step(self, action: np.ndarray):
		reward = float(self.state[0])
		self.state = np.asarray((self.state[0] + np.float32(1),))
		return self.state, reward, self.state[0] > self.high, {"reset": False}


Test = partial(
	Training,
	epochs=3,
	rounds=5,
	steps=100,
	Agent=partial(memory_size=1000000, Model=TestMlp),
	Env=partial(TestEnv),
)

if __name__ == "__main__":
	spec_init_run(Test)
