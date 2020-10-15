from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame, Timestamp
import agents.sac
from agents.util import pandas_dict, cached_property
import gym

from agents.networking import TrainerInterface

from agents.envs import Env
import time

# import pybullet_envs


@dataclass(eq=0)
class TrainingOffline:
    Env: type = Env
    Agent: type = agents.sac.Agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of steps per round
    update_model_interval: int = 100  # number of steps between model broadcasts
    update_buffer_interval: int = 100  # number of steps between retrieving buffered experiences in the interface
    max_training_steps_per_env_step: float = 1.0  # training will pause when above this ratio
    sleep_between_buffer_retrieval_attempts: float = 0.1  # algorithm will sleep for this amount of time when waiting for needed incoming samples
    stats_window: int = None  # default = steps, should be at least as long as a single episode
    seed: int = 0  # seed is currently not used
    tag: str = ''  # for logging, e.g. allows to compare groups of runs

    total_updates = 0

    def __post_init__(self):
        self.epoch = 0
        self.agent = self.Agent(Env=self.Env)
        self.total_samples = len(self.agent.memory)
        print(f"DEBUG: initial total_samples:{self.total_samples}")

    def update_buffer(self, interface):
        buffer = interface.retrieve_buffer()
        self.agent.memory.append(buffer)
        self.total_samples += len(buffer)

    def check_ratio(self, interface):
        ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 else -1.0
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            print("INFO: Waiting for new samples")
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                # wait for new samples
                self.update_buffer(interface)
                ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 else -1.0
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)

    def run_epoch(self, interface: TrainerInterface):
        stats = []
        state = None

        for rnd in range(self.rounds):
            print(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
            print(f"DEBUG: SAC (Training): current memory size:{len(self.agent.memory)}")

            t0 = pd.Timestamp.utcnow()
            stats_training = []

            self.check_ratio(interface)
            for step in range(self.steps):
                if self.total_updates == 0:
                    print("starting training")
                stats_training_dict = self.agent.train()
                stats_training_dict["return_test"] = self.agent.memory.stat_test_return
                stats_training_dict["return_train"] = self.agent.memory.stat_train_return
                stats_training += stats_training_dict,
                self.total_updates += 1
                if self.total_updates % self.update_model_interval == 0:
                    # broadcast model weights
                    interface.broadcast_model(self.agent.model_nograd.actor)
                if self.total_updates % self.update_buffer_interval == 0:
                    # retrieve local buffer in replay memory
                    self.update_buffer(interface)
                self.check_ratio(interface)

            stats += pandas_dict(
                round_time=Timestamp.utcnow() - t0,
                round_time_total=Timestamp.utcnow() - t0,
                **DataFrame(stats_training).mean(skipna=True)
            ),

            print(stats[-1].add_prefix("  ").to_string(), '\n')

        self.epoch += 1
        return stats
