from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame, Timestamp
import agents.sac
from agents.util import pandas_dict, cached_property
import gym

from agents.tm import TrainerInterface

# import pybullet_envs


@dataclass(eq=0)
class TrainingOffline:
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    Agent: type = agents.sac.Agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of steps per round
    update_model_interval: int = 100  # number of steps between model broadcasts
    update_buffer_interval: int = 100  # number of steps between retrieving buffered experiences in the interface
    stats_window: int = None  # default = steps, should be at least as long as a single episode
    seed: int = 0  # seed is currently not used
    tag: str = ''  # for logging, e.g. allows to compare groups of runs

    total_updates = 0

    def __post_init__(self):
        self.epoch = 0
        self.agent = self.Agent(Env=None, action_space=self.action_space, observation_space=self.observation_space)

    def run_epoch(self, interface: TrainerInterface):
        stats = []
        state = None

        for rnd in range(self.rounds):
            print(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
            print(f"DEBUG: SAC (Training): current memory size:{len(self.agent.memory)}")

            t0 = pd.Timestamp.utcnow()
            stats_training = []

            for step in range(self.steps):
                if self.total_updates == 0:
                    print("starting training")
                stats_training += self.agent.train(),
                self.total_updates += 1
                if self.total_updates % self.update_model_interval == 0:
                    # broadcast model weights
                    interface.broadcast_model(self.agent.model_nograd.actor)
                if self.total_updates % self.update_buffer_interval == 0:
                    # retrieve local buffer
                    pass

            stats += pandas_dict(
                round_time=Timestamp.utcnow() - t0,
                round_time_total=Timestamp.utcnow() - t0,
                **DataFrame(stats_training).mean(skipna=True)
            ),

            print(stats[-1].add_prefix("  ").to_string(), '\n')

        self.epoch += 1
        return stats
