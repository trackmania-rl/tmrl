from abc import ABC, abstractmethod


class TrainingAgent(ABC):
    """
    Training algorithm.

    CAUTION: When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 device):
        """
        Args:
            observation_space (Gym.spaces.Space): observation space (here for your convenience)
            action_space (Gym.spaces.Space): action space (here for your convenience)
            device (str): torch device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def train(self, batch):
        """
        Executes a training step.

        Args:
            batch: tuple or batched torch.tensors (previous observation, action, reward, new observation, terminated, truncated)

        Returns:
            ret_dict: dictionary: a dictionary containing one entry per metric you wish to log (e.g. for wandb)
        """
        raise NotImplementedError

    @abstractmethod
    def get_actor(self):
        """
        Returns the current ActorModule to be broadcast to the RolloutWorkers.

        Returns:
             actor: ActorModule: current actor to be broadcast
        """
        raise NotImplementedError


