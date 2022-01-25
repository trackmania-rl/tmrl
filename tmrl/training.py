from abc import ABC, abstractmethod


class TrainingAgent(ABC):
    def __init__(self,
                 observation_space,
                 action_space,
                 device):
        """
        observation_space, action_space and device are here for your convenience.

        You are free to use them or not, but your subclass must have them as args or kwargs of __init__() .
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def train(self, batch):
        """
        Executes a training step.

        Args:
            batch: tuple or batched torch.tensors (previous observation, action, reward, new observation, done)

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


