from copy import deepcopy
import pickle
from dataclasses import dataclass

import pandas as pd
import numpy as np
from pandas import DataFrame, Timestamp

import agents.sac

from agents.testing import Test
from agents.util import pandas_dict, cached_property
from agents.wrappers import StatsWrapper
from agents.envs import GymEnv
from agents.batch_env import get_env_state

# import pybullet_envs


@dataclass(eq=0)
class Training:
    Env: type = GymEnv
    Test: type = Test
    Agent: type = agents.sac.Agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of steps per round, one step = environment step
    stats_window: int = None  # default = steps, should be at least as long as a single episode
    seed: int = 0  # seed is currently not used
    tag: str = ''  # for logging, e.g. allows to compare groups of runs

    def __post_init__(self):
        self.epoch = 0
        self.agent = self.Agent(self.Env)

    def run_epoch(self):
        stats = []
        state = None

        with StatsWrapper(self.Env(seed_val=self.seed + self.epoch), window=self.stats_window or self.steps) as env:
            for rnd in range(self.rounds):
                print(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))

                t0 = pd.Timestamp.utcnow()
                stats_training = []

                # start test and run it in parallel to the training process
                test = self.Test(
                    Env=self.Env,
                    actor=self.agent.model,
                    steps=self.stats_window or self.steps,
                    base_seed=self.seed + self.epochs
                )

                for step in range(self.steps):
                    action, state, training_stats = self.agent.act(state, *env.transition, train=True)
                    stats_training += training_stats
                    env.step(action)

                stats += pandas_dict(
                    **env.stats(),
                    round_time=Timestamp.utcnow() - t0,
                    **test.stats().add_suffix("_test"),  # this blocks until the tests have finished
                    round_time_total=Timestamp.utcnow() - t0,
                    **DataFrame(stats_training).mean(skipna=True)
                ),

                print(stats[-1].add_prefix("  ").to_string(), '\n')

        self.epoch += 1
        return stats
