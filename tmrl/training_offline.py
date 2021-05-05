from dataclasses import dataclass
from pandas import DataFrame
import time
import tmrl.sac
from tmrl.util import pandas_dict, cached_property
from tmrl.networking import TrainerInterface

from tmrl.envs import Env

from tmrl.memory_dataloading import MemoryDataloading

import torch

import tmrl.custom.config_constants as cfg

# import pybullet_envs


@dataclass(eq=0)
class TrainingOffline:
    Env: type = Env
    Agent: type = tmrl.sac.SacAgent
    Memory: type = MemoryDataloading
    use_dataloader: bool = False  # Whether to use pytorch dataloader for multiprocess dataloading
    nb_workers: int = 0  # Number of parallel workers in pytorch dataloader
    batchsize: int = 256  # training batch size
    memory_size: int = 1000000  # replay memory size
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
    profiling: bool = False  # if True, run_epoch will be profiled and the profiling will be printed at the enc of each epoch
    agent_scheduler: callable = None  # if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
    start_training: int = 0  # minimum number of samples in the buffer before starting training

    device: str = None
    total_updates = 0

    def __post_init__(self):
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0
        # print(self.SacAgent)
        # print(self.Env)
        self.memory = self.Memory(memory_size=self.memory_size, batchsize=self.batchsize, nb_steps=self.steps, use_dataloader=False, device=device)
        self.agent = self.Agent(Env=self.Env, device=device)
        self.total_samples = len(self.memory)
        print(f"INFO: Initial total_samples:{self.total_samples}")

    def update_buffer(self, interface):
        buffer = interface.retrieve_buffer()
        self.memory.append(buffer)
        self.total_samples += len(buffer)

    def check_ratio(self, interface):
        ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            print("INFO: Waiting for new samples")
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                # wait for new samples
                self.update_buffer(interface)
                ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)
            print("INFO: Resuming training")

    def run_epoch(self, interface: TrainerInterface):
        stats = []
        state = None

        if self.agent_scheduler is not None:
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            print(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
            print(f"DEBUG: SAC (Training): current memory size:{len(self.memory)}")

            stats_training = []

            t0 = time.time()
            self.check_ratio(interface)
            t1 = time.time()

            if self.profiling:
                from pyinstrument import Profiler
                pro = Profiler()
                pro.start()

            t2 = time.time()

            t_sample_prev = t2

            for batch in self.memory:  # this samples a fixed number of batches

                if cfg.SYNCHRONIZE_CUDA:
                    torch.cuda.synchronize()

                t_sample = time.time()

                if self.total_updates % self.update_buffer_interval == 0:
                    # retrieve local buffer in replay memory
                    self.update_buffer(interface)

                t_update_buffer = time.time()

                if self.total_updates == 0:
                    print("starting training")
                stats_training_dict = self.agent.train(batch)

                if cfg.SYNCHRONIZE_CUDA:
                    torch.cuda.synchronize()

                t_train = time.time()

                stats_training_dict["return_test"] = self.memory.stat_test_return
                stats_training_dict["return_train"] = self.memory.stat_train_return
                stats_training_dict["episode_length_test"] = self.memory.stat_test_steps
                stats_training_dict["episode_length_train"] = self.memory.stat_train_steps
                stats_training_dict["sampling_duration"] = t_sample - t_sample_prev
                stats_training_dict["training_step_duration"] = t_train - t_update_buffer
                stats_training += stats_training_dict,
                self.total_updates += 1
                if self.total_updates % self.update_model_interval == 0:
                    # broadcast model weights
                    interface.broadcast_model(self.agent.model_nograd.actor)
                self.check_ratio(interface)

                if cfg.SYNCHRONIZE_CUDA:
                    torch.cuda.synchronize()

                t_sample_prev = time.time()

            t3 = time.time()

            round_time = t3 - t0
            idle_time = t1 - t0
            update_buf_time = t2 - t1
            train_time = t3 - t2
            print(f"DEBUG: round_time:{round_time}, idle_time:{idle_time}, update_buf_time:{update_buf_time}, train_time:{train_time}")
            stats += pandas_dict(memory_len=len(self.memory), round_time=round_time, idle_time=idle_time, **DataFrame(stats_training).mean(skipna=True)),

            print(stats[-1].add_prefix("  ").to_string(), '\n')

            if self.profiling:
                pro.stop()
                print(pro.output_text(unicode=True, color=False))

        self.epoch += 1
        return stats
