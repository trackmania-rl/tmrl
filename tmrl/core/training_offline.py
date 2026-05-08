# standard library imports
import time
from dataclasses import dataclass

# third-party imports
from pandas import DataFrame

# local imports
from tmrl.core.util import pandas_dict

import logging


__docformat__ = "google"


@dataclass(eq=0)
class TrainingOffline:
    """
    Training wrapper for off-policy algorithms.

    Args:
        env_cls (type): class of a dummy environment, used only to retrieve observation and action spaces if needed. Alternatively, this can be a tuple of the form (observation_space, action_space).
        memory_cls (type): class of the replay memory
        training_agent_cls (type): class of the training agent
        epochs (int): total number of epochs, we save the agent every epoch
        rounds (int): number of rounds per epoch, we generate statistics every round
        steps (int): number of training steps per round
        update_model_interval (int): number of training steps between model broadcasts
        update_buffer_interval (int): number of training steps between retrieving buffered samples
        max_training_steps_per_env_step (float): training will pause when above this ratio
        sleep_between_buffer_retrieval_attempts (float): algorithm will sleep for this amount of time when waiting for needed incoming samples
        profiling (bool): if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
        agent_scheduler (callable): if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
        start_training (int): minimum number of samples in the replay buffer before starting training
        device (str): device on which the memory will collate training samples
    """
    env_cls: type = None  # = GenericGymEnv  # dummy environment, used only to retrieve observation and action spaces if needed
    memory_cls: type = None  # = TorchMemory  # replay memory
    training_agent_cls: type = None  # = TrainingAgent  # training agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of training steps per round
    update_model_interval: int = 100  # number of training steps between model broadcasts
    update_buffer_interval: int = 100  # number of training steps between retrieving buffered samples
    max_training_steps_per_env_step: float = 1.0  # training will pause when above this ratio
    sleep_between_buffer_retrieval_attempts: float = 1.0  # algorithm will sleep for this amount of time when waiting for needed incoming samples
    profiling: bool = False  # if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
    agent_scheduler: callable = None  # if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
    start_training: int = 0  # minimum number of samples in the replay buffer before starting training
    device: str = None  # device on which the model of the TrainingAgent will live

    total_updates = 0

    def __post_init__(self):
        device = self.device
        self.epoch = 0
        self.memory = self.memory_cls(device=device)
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

        benchmarks_names = self.memory.get_benchmarks_names()

        if self.agent_scheduler is not None:
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logging.info(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
            logging.debug(f"(Training): current memory size:{len(self.memory)}")

            # round benchmarks
            update_buffer_duration = 0.0
            sampling_duration = 0.0
            training_step_duration = 0.0
            model_broadcast_duration = 0.0
            idle_duration = 0.0

            stats_training = []

            t0 = time.perf_counter()
            self.check_ratio(interface)
            t1 = time.perf_counter()

            idle_duration += t1 - t0

            if self.profiling:
                from pyinstrument import Profiler
                pro = Profiler()
                pro.start()

            t2 = time.perf_counter()

            for _ in range(self.steps):

                t_round_start = time.perf_counter()

                if self.total_updates % self.update_buffer_interval == 0:
                    # retrieve local buffer in replay memory
                    self.update_buffer(interface)

                t_update_buffer = time.perf_counter()

                batch = self.memory.sample()

                t_sample = time.perf_counter()

                if self.total_updates == 0:
                    logging.info(f"starting training")

                stats_training_dict = self.agent.train(batch)

                t_train = time.perf_counter()

                self.total_updates += 1
                if self.total_updates % self.update_model_interval == 0:
                    # broadcast model weights
                    interface.broadcast_model(self.agent.get_actor())

                t_broadcast = time.perf_counter()

                self.check_ratio(interface)

                t_round_end = time.perf_counter()

                # RolloutWorker performance:
                stats_training_dict["return_test"] = self.memory.stat_test_return
                stats_training_dict["return_train"] = self.memory.stat_train_return
                stats_training_dict["episode_length_test"] = self.memory.stat_test_steps
                stats_training_dict["episode_length_train"] = self.memory.stat_train_steps

                # Memory time benchmarks:
                memory_benchmarks = self.memory.get_benchmarks()
                if memory_benchmarks is not None:
                    if benchmarks_names is not None:
                        for i, benchmark in enumerate(memory_benchmarks):
                            stats_training_dict[benchmarks_names[i]] = benchmark
                    else:
                        for i, benchmark in enumerate(memory_benchmarks):
                            stats_training_dict[f"memory_benchmark_{i}"] = benchmark
                
                # Round time benchmarks:
                update_buffer_duration += t_update_buffer - t_round_start
                sampling_duration += t_sample - t_update_buffer
                training_step_duration += t_train - t_sample
                model_broadcast_duration += t_broadcast- t_train
                idle_duration += t_round_end - t_broadcast

                stats_training += stats_training_dict

            t3 = time.perf_counter()

            round_duration = t3 - t0
            stats += pandas_dict(memory_len=len(self.memory),
                                 round_duration=round_duration,
                                 idle_duration=idle_duration,
                                 sampling_duration=sampling_duration,
                                 update_buffer_duration=update_buffer_duration,
                                 training_step_duration=training_step_duration,
                                 model_broadcast_duration=model_broadcast_duration,
                                 **DataFrame(stats_training).mean(skipna=True)),

            logging.info("Round statistics:\n" + stats[-1].add_prefix("  ").to_string() + '\n')

            if self.profiling:
                pro.stop()
                logging.info(pro.output_text(unicode=True, color=False, show_all=True))

        self.epoch += 1
        return stats
