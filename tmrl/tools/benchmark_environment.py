# standard library imports
import random
import time

# third-party imports
import gymnasium
from gymnasium import spaces
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

# local imports
from tmrl.custom.custom_gym_interfaces import (TM2020Interface, TM2020InterfaceLidar)
import logging

NB_STEPS = 1000
ACT_COMPUTE_MIN = 0.0
ACT_COMPUTE_MAX = 0.05


def benchmark():
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceLidar
    env_config["benchmark"] = True
    env_config["running_average_factor"] = 0.05
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False, "min_nb_steps_before_failure": int(20 * 60)}
    env = gymnasium.make("real-time-gym-v1", config=env_config)

    t_d = time.time()
    o, i = env.reset()
    for idx in range(NB_STEPS - 1):
        act = action_space.sample()
        time.sleep(random.uniform(ACT_COMPUTE_MIN, ACT_COMPUTE_MAX))
        # o, r, d, t, i = env.step(act)
        o, r, d, t, i = env.step(None)
        if d or t:
            env.reset()
        logging.info(f"rew:{r}")
    t_f = time.time()

    elapsed_time = t_f - t_d
    logging.info(f"benchmark results: {env.benchmarks()}")
    logging.info(f"elapsed time: {elapsed_time}")
    logging.info(f"time-step duration: {elapsed_time / NB_STEPS}")


if __name__ == "__main__":
    benchmark()
