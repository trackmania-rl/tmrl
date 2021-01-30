from copy import deepcopy
import pickle
from dataclasses import dataclass

import pandas as pd
import numpy as np
from pandas import DataFrame, Timestamp

import tmrl.sac

from tmrl.testing import Test
from tmrl.util import pandas_dict, cached_property
from tmrl.wrappers import StatsWrapper
from tmrl.envs import GymEnv
from tmrl.batch_env import get_env_state

# import pybullet_envs


@dataclass(eq=0)
class Training:
    Env: type = GymEnv
    Test: type = Test
    Agent: type = tmrl.sac.Agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of steps per round, one step = global step
    nb_env_it_per_step: int = 1  # number of environment steps per global step (real-world: multiple of the episode)
    nb_train_it_per_step: int = 1  # number of training steps per global step
    stats_window: int = None  # default = steps, should be at least as long as a single episode
    seed: int = 0  # seed is currently not used
    tag: str = ''  # for logging, e.g. allows to compare groups of runs
    start_training: int = 10000

    total_updates = 0
    environment_steps = 0

    def __post_init__(self):
        self.epoch = 0
        self.agent = self.Agent(self.Env)

    def run_epoch(self):
        stats = []
        state = None

        with StatsWrapper(self.Env(seed_val=self.seed + self.epoch), window=self.stats_window or self.steps) as env:
            for rnd in range(self.rounds):
                print(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
                print(f"DEBUG: SAC (Training): current memory size:{len(self.agent.memory)}")

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

                    for _ in range(self.nb_env_it_per_step):
                        action, state = self.agent.act(state, *env.transition, train=True)
                        self.environment_steps += 1
                        env.step(action)

                    if self.environment_steps >= self.start_training:
                        if self.total_updates == 0:
                            print("starting training")
                        t_stats = []
                        for _ in range(self.nb_train_it_per_step):
                            t_stats += self.agent.train(),
                            self.total_updates += 1
                        stats_training += t_stats

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
