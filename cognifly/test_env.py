import gym
from agents.custom.custom_gym_interfaces import CogniflyInterfaceTask1
from rtgym import DEFAULT_CONFIG_DICT
import numpy as np

my_config = DEFAULT_CONFIG_DICT
my_config['interface'] = CogniflyInterfaceTask1

env = gym.make("rtgym:real-time-gym-v0", config=my_config)

obs = env.reset()
print(f"obs:{obs}")
target = obs[0][3]
pos = obs[0][0]
i = 0
done = False
while i < 1000 and not done:
    act = np.array([target - pos], dtype=np.float32)
    obs, rew, done, info = env.step(act)
    target = obs[0][3]
    pos = obs[0][0]
    delay = obs[0][4]
    print(f"pos:{pos:.2f}, tar:{target:.2f}, rew:{rew:.2f}, del:{delay:.3f}, act:{act}")

