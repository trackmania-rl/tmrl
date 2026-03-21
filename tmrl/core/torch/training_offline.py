import torch

from tmrl.core.training_offline import TrainingOffline


class TorchTrainingOffline(TrainingOffline):
    """
    TrainingOffline for trainers based on PyTorch.

    This class implements automatic device selection with PyTorch.
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
                 device: str = None):
        """
        Same arguments as `TrainingOffline`, but when `device` is `None` it is selected automatically for torch.

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
            device (str): device on which the memory will collate training samples (None for automatic)
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
