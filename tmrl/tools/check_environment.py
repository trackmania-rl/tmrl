# standard library imports
import random
import time

# third-party imports
import gym
import mss
import numpy as np
from gym import spaces
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

# local imports
from tmrl.custom.custom_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TMInterface, TMInterfaceLidar
from tmrl.custom.utils.tools import Lidar
import logging


def check_env_tm20lidar():

    sct = mss.mss()
    monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
    lidar = Lidar(monitor=monitor, road_point=(440, 479))
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceLidar
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False, "min_nb_steps_before_early_done": int(20 * 60), "road_point": (440, 479), "record": False}
    # env_config["time_step_duration"] = 0.5  # nominal duration of your time-step
    # env_config["start_obs_capture"] = 0.4
    env = gym.make("real-time-gym-v0", config=env_config)
    o = env.reset()
    while True:
        o, r, d, i = env.step(None)
        logging.info(f"r:{r}, d:{d}")
        if d:
            o = env.reset()
        img = np.asarray(sct.grab(monitor))[:, :, :3]
        lidar.lidar_20(img, True)


if __name__ == "__main__":
    check_env_tm20lidar()
