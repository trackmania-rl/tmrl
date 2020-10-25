import gym
from gym import spaces
import random
import time
from gym_real_time.envs.real_time_env import DEFAULT_CONFIG_DICT
from agents.custom.custom_gym_interfaces import TMInterface, TM2020Interface, TMInterfaceLidar, TM2020InterfaceLidar


env_config = DEFAULT_CONFIG_DICT
env_config["interface"] = TM2020InterfaceLidar
# env_config["time_step_duration"] = 0.5  # nominal duration of your time-step
# env_config["start_obs_capture"] = 0.4
env = gym.make("gym_real_time:gym-rt-v0", config=env_config)
o = env.reset()
while True:
    o, r, d, i = env.step(None)
    print(r)

    if d:
        o = env.reset()

