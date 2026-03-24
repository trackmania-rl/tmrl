from abc import ABC

import jax
import jax.numpy as jnp
import orbax
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
        self.device = device or jax.devices()[0]

    def save(self, path):
        _, state = nnx.split(self)
        with orbax.checkpoint.StandardCheckpointer() as checkpointer:
            checkpointer.save(path, state)

    def load(self, path, device):
        if device is not None:
            self.device = device
        _, abs_state = nnx.split(self)
        with orbax.checkpoint.StandardCheckpointer() as checkpointer:
            state = checkpointer.restore(path, abs_state)
        nnx.update(self, state)
        return self

    def act_(self, obs, test=False):
        """
        Transforms obs into a tree of jax arrays
        """
        obs = collate_jax([obs], device=self.device)
        action = self.act(obs, test=test)
        return action
    
    def to_device(self, device):
        self.device = device
        _, state = nnx.split(self)
        state = jax.device_put(state, device)
        nnx.update(self, state)
        return self
