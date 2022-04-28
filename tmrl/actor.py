from abc import ABC, abstractmethod
import torch


__docformat__ = "google"


class ActorModule(torch.nn.Module, ABC):
    """
    Implement this interface for the RolloutWorker(s) to interact with your policy.

    This is a sublass of torch.nn.Module and must implement forward().
    Typically, your implementation of act() may call forward() with gradients turned off.

    .. note::
       If overidden, the __init()__ definition must at least take the two following arguments (args or kwargs):
       `observation_space` and `action_space`.
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """
    def __init__(self, observation_space, action_space):
        """
        Args:
            observation_space (Gym.spaces.Space): observation space (here for your convenience)
            action_space (Gym.spaces.Space): action space (here for your convenience)
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = None

    @abstractmethod
    def act(self, obs, test=False):
        """
        Must compute an action from an observation.
        
        Args:
            obs (object): the observation
            test (bool): True at test time, False otherwise

        Returns:
            act (numpy.array): the computed action
        """
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def to(self, device):
        """
        Keeps track which device this module has been moved to.

        Args:
            device (str): the device on which the torch module lives (e.g., `"cpu"` or `"cuda:0"`)
        """
        self.device = device
        return super().to(device=device)
