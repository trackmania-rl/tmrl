from abc import ABC, abstractmethod


class TrainingAgent(ABC):
    """
    Training algorithm.

    CAUTION: When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """

    def __init__(self, observation_space, action_space, device):
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space
            action_space (gymnasium.spaces.Space): action space
            device (str): device for training
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def train(
        self,
        batch,
        epoch: int | None = None,
        batch_index: int | None = None,
        iters: int | None = None,
    ):
        """
        Executes a training step.

        Args:
            epoch:
            iters:
            batch_index:
            batch: (prev_obs, action, reward, new_obs, terminated, truncated)

        Returns:
            dict: a dictionary containing one entry per metric you wish to log (e.g. for wandb)
        """
        raise NotImplementedError

    @abstractmethod
    def get_actor(self):
        """
        Returns the current ActorModule to be broadcast to the RolloutWorkers.

        Returns:
             ActorModule: current actor to be broadcast
        """
        raise NotImplementedError
