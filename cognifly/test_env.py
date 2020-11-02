import gym
from agents.custom.custom_gym_interfaces import Cognifly1Interface
from rtgym import DEFAULT_CONFIG_DICT

my_config = DEFAULT_CONFIG_DICT
my_config['interface'] = Cognifly1Interface

env = gym.make("rtgym:real-time-gym-v0", config=my_config)

