"""
The `tmrl` library is a complete framework designed to help you implement reinforcement learning pipelines
in real-world applications such as robots or videogames.

As a very fun example, we readily provide a training pipeline for the
[TrackMania 2020](https://www.trackmania.com) videogame.

We strongly encourage new readers to visit our [GitHub](https://github.com/trackmania-rl/tmrl)
as it contains a lot of information and tutorials to help you get on track :)

The documentation describes the `tmrl` python API and is intended for developers who want to implement their own
training pipelines.
We also provide an [advanced tutorial](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/tuto.py)
for this purpose.

The three most important classes are `Server`, `RolloutWorker` and `Trainer`, all defined in the
[tmrl.networking](tmrl/networking.html) submodule.
"""

# logger (basicConfig must be called before importing anything)
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# standard library imports
from dataclasses import dataclass
# from tmrl.networking import Server, RolloutWorker, Trainer
# from tmrl.custom.custom_gym_interfaces import TM2020InterfaceLidar
from tmrl.envs import GenericGymEnv
from tmrl.config.config_objects import CONFIG_DICT


def get_environment():
    """
    Default TMRL Gym environment for TrackMania 2020.

    Returns:
        An instance of the default TMRL Gym environment
    """
    return GenericGymEnv(id="real-time-gym-v0", gym_kwargs={"config": CONFIG_DICT})
