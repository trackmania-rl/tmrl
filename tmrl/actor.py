from abc import ABC, abstractmethod
import torch


class ActorModule(ABC, torch.nn.Module):
    """
    Interface for the RolloutWorker(s) to interact with the policy.

    This is a torch neural network and must implement forward().
    Typically, act() calls forward() with gradients turned off.
    """

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
