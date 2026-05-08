from abc import ABC
import pickle as pkl

import jax
from flax import nnx
import gymnasium

from tmrl.core.actor import ActorModule
from tmrl.core.jax.util import collate_jax


__docformat__ = "google"


class NNXActorModule(ActorModule, nnx.Module, ABC):
    """
    Partial implementation of `ActorModule` as a `nnx.Module`.

    You can implement this instead of `ActorModule` when using Jax.
    `NNXActorModule` is a subclass of `nnx.Module`.

    When using `NNXActorModule`, the device is selected by Jax.

    .. note::
       If overidden, the __init()__ definition must at least take the two following arguments (args or kwargs):
       `observation_space` and `action_space`.
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """
    def __init__(self,
                 observation_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space,
                 device=None):
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
            action_space (gymnasium.spaces.Space): action space (here for your convenience)
        """
        ActorModule.__init__(self, observation_space, action_space)  # ActorModule
        self.device = device  # or jax.devices()[0]  # FIXME: not picklable

    def save(self, path):
        _, state = nnx.split(self)
        with open(path, 'wb') as f:
            pkl.dump(state, f)

    def load(self, path, device):
        _, abs_state = nnx.split(self)
        with open(path, 'rb') as f:
            state = pkl.load(f)
        nnx.update(self, state)
        if device is not None:
            self.to_device(device)
        return self

    def act_(self, obs, test=False):
        """
        Transforms obs into a tree of jax arrays
        """
        device = self.device
        if device is not None:
            device = jax.devices(device)[0]
        obs = collate_jax([obs], device=device)
        action = self.act(obs, test=test)
        return action
    
    def to_device(self, device:str):
        self.device = device  # store as string
        _, state = nnx.split(self)
        if device is not None:
            device = jax.devices(device)[0]
            state = jax.device_put(state, device)
        nnx.update(self, state)
        return self
