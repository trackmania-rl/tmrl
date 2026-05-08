import time
import logging

from pandas import DataFrame
from flax import nnx

from tmrl.core.training_offline import TrainingOffline
from tmrl.core.util import pandas_dict


@nnx.jit
def _train_jit(batch, agent):
    return agent.train(batch)


@nnx.jit
def _sample_and_train_jit(memory, agent):
    batch = memory.sample()
    return agent.train(batch)


class NNXTrainingOffline(TrainingOffline):
    """
    TrainingOffline for trainers based on Flax NNX, with jit-able train() method.
    Optionally, NNXTrainingOffline can be used with memories that have a jit-able sample() method.
    """
    def __init__(self,
                 env_cls: type = None,
                 memory_cls: type = None,
                 training_agent_cls: type = None,
                 epochs: int = 10,
                 rounds: int = 50,
                 steps: int = 2000,
                 update_model_interval: int = 100,
                 update_buffer_interval: int = 100,
                 max_training_steps_per_env_step: float = 1.0,
                 sleep_between_buffer_retrieval_attempts: float = 1.0,
                 profiling: bool = False,
                 agent_scheduler: callable = None,
                 start_training: int = 0,
                 device: str = None,
                 jit_sampling: bool = False):
        """
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
            device (str): device to use (None for automatic)
            jit_sampling (bool): whether to jit the sampling method from memory_cls with nnx.jit
        """
        super().__init__(env_cls,
                         memory_cls,
                         training_agent_cls,
                         epochs,
                         rounds,
                         steps,
                         update_model_interval,
                         update_buffer_interval,
                         max_training_steps_per_env_step,
                         sleep_between_buffer_retrieval_attempts,
                         profiling,
                         agent_scheduler,
                         start_training,
                         device)
        self.jit_sampling = jit_sampling
        
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
            sampling_and_training_duration = 0.0
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

                if self.total_updates == 0:
                    logging.info(f"starting training")
                
                if not self.jit_sampling:  # sampling method not jit-able
                    batch = self.memory.sample()
                    t_sample = time.perf_counter()
                    stats_training_dict = _train_jit(batch, self.agent)
                    t_train = time.perf_counter()
                    sampling_duration += t_sample - t_update_buffer
                    training_step_duration += t_train - t_sample
                else:  # both method are jit-able
                    stats_training_dict = _sample_and_train_jit(self.memory, self.agent)
                    t_train = time.perf_counter()
                    sampling_and_training_duration += t_train - t_update_buffer
                stats_training_dict = {k: float(v) for k, v in stats_training_dict.items()}

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
                model_broadcast_duration += t_broadcast- t_train
                idle_duration += t_round_end - t_broadcast

                stats_training += stats_training_dict,

            t3 = time.perf_counter()

            round_duration = t3 - t0
            stats += pandas_dict(memory_len=len(self.memory),
                                 round_duration=round_duration,
                                 idle_duration=idle_duration,
                                 update_buffer_duration=update_buffer_duration,
                                 **(
                                     dict(sampling_duration=sampling_duration, training_step_duration=training_step_duration)
                                     if not self.jit_sampling else
                                     dict(sampling_and_training_duration=sampling_and_training_duration)
                                    ),
                                 model_broadcast_duration=model_broadcast_duration,
                                 **DataFrame(stats_training).mean(skipna=True)),

            logging.info("Round statistics:\n" + stats[-1].add_prefix("  ").to_string() + '\n')

            if self.profiling:
                pro.stop()
                logging.info(pro.output_text(unicode=True, color=False, show_all=True))

        self.epoch += 1
        return stats
