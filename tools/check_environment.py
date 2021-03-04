import gym
from gym import spaces
import random
import time
import numpy as np
import mss
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT
from tmrl.custom.custom_gym_interfaces import TMInterface, TM2020Interface, TMInterfaceLidar, TM2020InterfaceLidar
from tmrl.custom.utils.tools import Lidar


def check_environment():

    sct = mss.mss()
    monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
    lidar = Lidar(monitor=monitor, road_point=(440, 479))
    env_config = DEFAULT_CONFIG_DICT
    env_config["interface"] = TM2020InterfaceLidar
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False, "min_nb_steps_before_early_done": int(20 * 60), "road_point": (440, 479), "record": False}
    # env_config["time_step_duration"] = 0.5  # nominal duration of your time-step
    # env_config["start_obs_capture"] = 0.4
    env = gym.make("rtgym:real-time-gym-v0", config=env_config)
    o = env.reset()
    while True:
        o, r, d, i = env.step(None)
        print(f"r:{r}, d:{d}")
        if d:
            o = env.reset()
        img = np.asarray(sct.grab(monitor))[:, :, :3]
        lidar.lidar_20(img, True)


if __name__ == "__main__":
    check_environment()
