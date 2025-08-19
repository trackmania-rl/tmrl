from abc import ABC

import torch

from tmrl.actor import ActorModule
from tmrl.torch.util import collate_torch


class TorchActorModule(ActorModule, torch.nn.Module, ABC):
    """
    Partial implementation of `ActorModule` as a `torch.nn.Module`.

    You can implement this instead of `ActorModule` when using PyTorch.
    `TorchActorModule` is a subclass of `torch.nn.Module` and may implement `forward()`.
    Typically, your implementation of `act()` can call `forward()` with gradients turned off.

    When using `TorchActorModule`, the `act` method receives observations collated on `device`,
    with an additional dimension corresponding to the batch size.

    .. note::
       If overidden, the __init()__ definition must at least take the two following arguments (args or kwargs):
       `observation_space` and `action_space`.
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """
    def __init__(self, observation_space, action_space, device="cpu"):
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
            action_space (gymnasium.spaces.Space): action space (here for your convenience)
            device: device where your model should live and where observations for `act` will be collated
        """
        super().__init__(observation_space, action_space)  # ActorModule
        # super().__init__()  # torch.nn.Module
        self.device = device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return self

    def act_(self, obs, test=False):
        obs = collate_torch([obs], device=self.device)
        with torch.no_grad():
            action = self.act(obs, test=test)
        return action

    # noinspection PyMethodOverriding
    def to(self, device):
        self.device = device
        return super().to(device=device)

    def to_device(self, device):
        return self.to(device)
