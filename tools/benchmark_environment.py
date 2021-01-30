import gym
from gym import spaces
import random
import time
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT
from tmrl.custom.custom_gym_interfaces import TMInterface, TM2020Interface, TMInterfaceLidar, TM2020InterfaceLidar

NB_STEPS = 1000
ACT_COMPUTE_MIN = 0.0
ACT_COMPUTE_MAX = 0.05

action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

env_config = DEFAULT_CONFIG_DICT
env_config["interface"] = TM2020InterfaceLidar
env_config["benchmark"] = True
env_config["running_average_factor"] = 0.05
env = gym.make("rtgym:real-time-gym-v0", config=env_config)

t_d = time.time()
obs = env.reset()
for idx in range(NB_STEPS-1):
    act = action_space.sample()
    time.sleep(random.uniform(ACT_COMPUTE_MIN, ACT_COMPUTE_MAX))
    # o, r, d, i = env.step(act)
    o, r, d, i = env.step(None)
    print(f"rew:{r}")
t_f = time.time()

elapsed_time = t_f - t_d
print(f"benchmark results: {env.benchmarks()}")
print(f"elapsed time: {elapsed_time}")
print(f"time-step duration: {elapsed_time / NB_STEPS}")
