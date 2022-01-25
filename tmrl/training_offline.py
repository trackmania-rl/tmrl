# standard library imports
import time
from dataclasses import dataclass

# third-party imports
import torch
from pandas import DataFrame

# local imports
import tmrl.config.config_constants as cfg
import tmrl.sac
# from tmrl.envs import GenericGymEnv
# from tmrl.memory_dataloading import MemoryDataloading
# from tmrl.networking import TrainerInterface
# from tmrl.training import TrainingAgent
from tmrl.util import pandas_dict

import logging
# import pybullet_envs


@dataclass(eq=0)
class TrainingOffline:
    env_cls: type = None  # = GenericGymEnv  # dummy environment, used only to retrieve observation and action spaces if needed
    memory_cls: type = None  # = MemoryDataloading  # replay memory
    training_agent_cls: type = None  # = TrainingAgent  # training agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of training steps per round
    update_model_interval: int = 100  # number of training steps between model broadcasts
    update_buffer_interval: int = 100  # number of training steps between retrieving buffered samples
    max_training_steps_per_env_step: float = 1.0  # training will pause when above this ratio
    sleep_between_buffer_retrieval_attempts: float = 0.1  # algorithm will sleep for this amount of time when waiting for needed incoming samples
    profiling: bool = False  # if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
    agent_scheduler: callable = None  # if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
    start_training: int = 0  # minimum number of samples in the replay buffer before starting training
    device: str = None  # device on which the model of the TrainingAgent will live (None for automatic)

    total_updates = 0

    def __post_init__(self):
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0
        self.memory = self.memory_cls(nb_steps=self.steps, device=device)
        if type(self.env_cls) == tuple:
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
        self.agent = self.training_agent_cls(observation_space=observation_space,
                                             action_space=action_space,
                                             device=device)
        self.total_samples = len(self.memory)
        logging.info(f" Initial total_samples:{self.total_samples}")

    def update_buffer(self, interface):
        buffer = interface.retrieve_buffer()
        self.memory.append(buffer)
        self.total_samples += len(buffer)

    def check_ratio(self, interface):
        ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            logging.info(f" Waiting for new samples")
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                # wait for new samples
                self.update_buffer(interface)
                ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)
            logging.info(f" Resuming training")

    def run_epoch(self, interface):
        stats = []
        state = None

        if self.agent_scheduler is not None:
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logging.info(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
            logging.debug(f" SAC (Training): current memory size:{len(self.memory)}")

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
                    logging.info(f"starting training")
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
                    interface.broadcast_model(self.agent.get_actor())
                self.check_ratio(interface)

                if cfg.SYNCHRONIZE_CUDA:
                    torch.cuda.synchronize()

                t_sample_prev = time.time()

            t3 = time.time()

            round_time = t3 - t0
            idle_time = t1 - t0
            update_buf_time = t2 - t1
            train_time = t3 - t2
            logging.debug(f" round_time:{round_time}, idle_time:{idle_time}, update_buf_time:{update_buf_time}, train_time:{train_time}")
            stats += pandas_dict(memory_len=len(self.memory), round_time=round_time, idle_time=idle_time, **DataFrame(stats_training).mean(skipna=True)),

            logging.info(stats[-1].add_prefix("  ").to_string() + '\n')

            if self.profiling:
                pro.stop()
                logging.info(pro.output_text(unicode=True, color=False, show_all=True))

        self.epoch += 1
        return stats
