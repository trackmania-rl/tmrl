# logger (basicConfig must be called before importing anything)
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# standard library imports
from dataclasses import dataclass
from tmrl.networking import Server, RolloutWorker
from tmrl.training import Trainer
from tmrl.custom.custom_gym_interfaces import TM2020InterfaceLidar
from tmrl.envs import GenericGymEnv
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.tools.record import record_reward_dist
from tmrl.tools.check_environment import check_env_tm20lidar


def get_environment():
    """
    Gets TMRL Gym environment
    """
    return GenericGymEnv(id="rtgym:real-time-gym-v0", gym_kwargs={"config": CONFIG_DICT})
