from abc import ABC, abstractmethod
import torch


class ActorModule(torch.nn.Module, ABC):
    """
    Interface for the RolloutWorker(s) to interact with the policy.

    This is a torch neural network and must implement forward().
    Typically, act() calls forward() with gradients turned off.

    The __init()__ definition must at least take the two following arguments (args or kwargs):
        observation_space
        action_space
    The __init()__ method must also call the superclass __init__() via super().__init__(observation_space, action_space)
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = None

    @abstractmethod
    def act(self, obs, test=False):
        """
        Returns an action from an observation.
        
        Args:
            obs: the observation
            test: bool: True at test time, False otherwise
        Returns:
            act: numpy array: the computed action
        """
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def to(self, device):
        """
        keeps track which device this module has been moved to
        """
        self.device = device
        return super().to(device=device)
