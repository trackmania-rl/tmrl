import gym
from gym import spaces
import random
import time
from gym_tmrl.envs.tmrl_env import DEFAULT_CONFIG_DICT, TMInterface, TM2020Interface, TMInterfaceLidar

NB_STEPS = 1000
ACT_COMPUTE_MIN = 0.03
ACT_COMPUTE_MAX = 0.02

action_space = spaces.Box(low=0.0, high=1.0, shape=(4,))

env_config = DEFAULT_CONFIG_DICT
env_config["interface"] = TMInterfaceLidar
env = gym.make("gym_tmrl:gym-tmrl-v0", config=env_config)

t_d = time.time()
obs = env.reset()
for idx in range(NB_STEPS-1):
    act = action_space.sample()
    time.sleep(random.uniform(ACT_COMPUTE_MIN, ACT_COMPUTE_MAX))
    o, r, d, i = env.step(act)
t_f = time.time()

elapsed_time = t_f - t_d
print(f"benchmark results: obs capture: {env.benchmarks()}")
print(f"elapsed time: {elapsed_time}")
print(f"time-step duration: {elapsed_time / NB_STEPS}")

