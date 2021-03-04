import gym
from gym import spaces
import random
import time
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT
from tmrl.custom.custom_gym_interfaces import TMInterface, TM2020Interface, TMInterfaceLidar, TM2020InterfaceLidar


def run_car():
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(3, ))
    env_config = DEFAULT_CONFIG_DICT
    env_config["interface"] = TM2020InterfaceLidar
    env_config["worker"] = True
    env_config["running_average_factor"] = 0.05
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1,
                                      "gamepad": False,
                                      "min_nb_steps_before_early_done": int(20 * 60),
                                      "road_point": (440, 479),
                                      "record": False,
                                      "save_replay": True}
    env = gym.make("rtgym:real-time-gym-v0", config=env_config)
    obs = env.reset()
    while True:
        o, r, d, i = env.step(None)


if __name__ == "__main__":
    run_car()
