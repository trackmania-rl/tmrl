import unittest
from gym_real_time import GymRealTimeInterface, DEFAULT_CONFIG_DICT
import time
import gym
import numpy as np


class DummyInterface(GymRealTimeInterface):
    def __init__(self):
        self.control_time = None
        self.control = None

    def send_control(self, control):
        self.control_time = time.time()
        self.control = control

    def reset(self):
        return [time.time(), self.control, self.control_time]

    def get_obs_rew_done(self):
        return [time.time(), self.control, self.control_time], 0.0, False

    def get_observation_space(self):
        ob = gym.spaces.Box(low=np.array([0.0]), high=np.array([np.inf]), dtype=np.float32)
        co = gym.spaces.Box(low=np.array([-1.0]), high=np.array([np.inf]), dtype=np.float32)
        ct = gym.spaces.Box(low=np.array([0.0]), high=np.array([np.inf]), dtype=np.float32)
        return gym.spaces.Tuple((ob, co, ct))

    def get_action_space(self):
        return gym.spaces.Box(low=np.array([-1.0]), high=np.array([np.inf]), dtype=np.float32)

    def get_default_action(self):
        return -1.0


config = DEFAULT_CONFIG_DICT
config["interface"] = DummyInterface
config["time_step_duration"] = 0.1
config["start_obs_capture"] = 0.08
config["act_buf_len"] = 1


class TestEnv(unittest.TestCase):
    def test_timing(self):
        epsilon = 0.01
        env = gym.make("gym_real_time:gym-rt-v0", config=config)
        obs1 = env.reset()
        elapsed_since_obs1_capture = time.time() - obs1[0]
        self.assertGreater(epsilon, elapsed_since_obs1_capture)
        self.assertGreater(elapsed_since_obs1_capture, - epsilon)
        self.assertEqual(obs1[3], -1)
        self.assertIs(obs1[1], None)
        self.assertIs(obs1[2], None)
        act = 0.0
        obs2, _, _, _ = env.step(act)
        self.assertEqual(obs2[3], act)
        self.assertEqual(obs2[1], -1.0)
        self.assertGreater(obs2[2] - obs1[0], - epsilon)
        self.assertGreater(epsilon, obs2[2] - obs1[0])
        self.assertGreater(obs2[0] - obs1[0], 0.08 - epsilon)
        self.assertGreater(0.08 + epsilon, obs2[0] - obs1[0])
        for i in range(3):
            obs1 = obs2
            act = float(i + 1)
            obs2, _, _, _ = env.step(act)
            self.assertEqual(obs2[3], act)
            self.assertEqual(obs2[1], act - 1.0)
            self.assertGreater(time.time() - obs2[2], 0.1 - epsilon)
            self.assertGreater(0.1 + epsilon, time.time() - obs2[2])
            self.assertGreater(obs2[0] - obs1[0], 0.1 - epsilon)
            self.assertGreater(0.1 + epsilon, obs2[0] - obs1[0])


if __name__ == '__main__':
    unittest.main()
