from abc import ABC, abstractmethod
import torch


class ActorModule(torch.nn.Module, ABC):
    """
    Implement this interface for the RolloutWorker(s) to interact with your policy.

    This is a sublass of torch.nn.Module and must implement forward().
    Typically, your implementation of act() may call forward() with gradients turned off.

    If overidden, the __init()__ definition must at least take the two following arguments (args or kwargs):
        `observation_space`
        `action_space`
    The __init()__ method must initialize the superclass via super().__init__(observation_space, action_space)
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = None

    @abstractmethod
    def act(self, obs, test=False):
        """
        Computes an action from an observation.
        
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
        keeps track which device this module has been moved to
        """
        self.device = device
        return super().to(device=device)
