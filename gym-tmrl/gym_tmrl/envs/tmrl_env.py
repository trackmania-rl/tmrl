# This is an environment for trackmania
# http://www.flint.jp/misc/?q=dik&lang=en  key indicator

from gym import Env
import gym.spaces as spaces
import numpy as np
import time
from threading import Thread

import cv2
import mss
import sys

from gym_tmrl.envs.tools import load_digits, get_speed
from gym_tmrl.envs.key_event import apply_control, keyres

from collections import deque

# from pynput.keyboard import Key, Controller
import ctypes


class TMInterface():
    """
    This is the API needed for the algorithm to control Trackmania
    """
    def __init__(self, img_hist_len=4):
        """
        Args:
        """
        self.monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
        self.sct = mss.mss()
        self.last_time = time.time()
        self.digits = load_digits()
        self.img_hist_len = img_hist_len
        self.img_hist = deque(maxlen=self.img_hist_len)

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        Args:
            control: np.array: [forward,backward,right,left]
        """
        actions = []
        if control[0] > 0.5:
            actions.append("f")
        elif control[1] > 0.5:
            actions.append("b")
        if control[2] > 0.5:
            actions.append("r")
        elif control[3] > 0.5:
            actions.append("l")
        apply_control(actions)  # TODO: format this
    
    def grab_img_and_speed(self):
        img = np.asarray(self.sct.grab(self.monitor))[:,:,:3]
        speed = np.array([get_speed(img, self.digits), ], dtype='float32')
        img = cv2.resize(img, (32,32))
        img = np.moveaxis(img, -1, 0)
        img = img.astype('float32') / 255.0 - 0.5  # normalized and centered
        print(f"DEBUG: Env: captured speed:{speed}")
        speed = speed / 1000.0  # normalized, but not centered
        self.img = img  # for render()
        return img, speed
    
    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.send_control([0, 0, 0, 0])
        keyres()
        time.sleep(0.05)  # must be long enough for image to be refreshed
        img, speed = self.grab_img_and_speed()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array([i for i in self.img_hist])
        obs = [speed, imgs]
        return obs

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        apply_control([0, 0, 0, 0])

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed = self.grab_img_and_speed()
        rew = speed[0]
        self.img_hist.append(img)
        imgs = np.array([i for i in self.img_hist])
        obs = [speed, imgs]
        done = False  # TODO: True if race complete
        
        return obs, rew, done

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1.0, shape=(1,))
        img = spaces.Box(low=0.0, high=1.0, shape=(self.img_hist_len, 3, 32, 32))
        return spaces.Tuple((speed, img))

    def get_action_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(4,))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0, 0.0])


DEFAULT_CONFIG_DICT = {
    "forced_sleep_time": 0.1,
    "ep_max_length": 100,
    "real_time": False,
    "act_threading": False,
    "act_in_obs": False,
}


class TMRLEnv(Env):
    def __init__(self, config=DEFAULT_CONFIG_DICT):
        """
        :param ep_max_length: (int) the max length of each episodes in timesteps
        :param real_time: bool: whether to use the RTRL setting
        :param act_threading: bool (optional, default: True): whether actions are executed asynchronously in the RTRL setting.
            Typically this is useful for the real world and for external simulators
            When this is True, __send_act_and_wait_n() should be a cpu-light I/O operation or python multithreading will slow down the calling program
            For cpu-intensive tasks (e.g. embedded simulators), this should be True only if you ensure that the CPU-intensive part is executed in another process while __send_act_and_wait_n() is only used for interprocess communications
        :param act_in_obs: bool (optional, default True): whether to augment the observation with the action buffer (DCRL)
        :param default_action: float (optional, default None): default action to append at reset when the previous is True
        :param act_prepro_func: function (optional, default None): function that maps the action input to the actual applied action
        :param obs_prepro_func: function (optional, default None): function that maps the observation output to the actual returned observation
        :param forced_sleep_time: float (optional, default None): seconds slept after apply_action() (~ time-step duration)
        :param reset_act_buf: bool (optional, defaut True): whether action buffer should be re-initialized at reset
        """
        # config variables:
        self.act_prepro_func = config["act_prepro_func"] if "act_prepro_func" in config else None
        self.obs_prepro_func = config["obs_prepro_func"] if "obs_prepro_func" in config else None
        self.forced_sleep_time = config["forced_sleep_time"] if "forced_sleep_time" in config else None
        self.ep_max_length = config["ep_max_length"]
        self.real_time = config["real_time"]
        self.act_threading = config["act_threading"] if "act_threading" in config else True
        if not self.real_time:
            self.act_threading = False
        if self.act_threading:
            self._at_thread = Thread(target=None, args=(), kwargs={}, daemon=True)
            self._at_thread.start()  # dummy start for later call to join()
        self.act_in_obs = config["act_in_obs"] if "act_in_obs" in config else True
        self.reset_act_buf = config["reset_act_buf"] if "reset_act_buf" in config else True
        self.interface = TMInterface()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.current_step = 0
        self.initialized = False
        # state variables:
        self.default_action = self.interface.get_default_action()
        self.last_action = self.default_action

    def _join_act_thread(self):
        """
        This is called at the beginning of every user-side API functions (step(), reset()...) for thread safety
        This ensures that the previous time-step is completed when starting a new one
        """
        if self.act_threading:
            self._at_thread.join()

    def _run_time_step(self, *args, **kwargs):
        """
        This is what must be called in step() to apply an action
        Call this with the args and kwargs expected by self.__send_act_and_wait()
        This in turn calls self.__send_act_and_wait()
        In action-threading, self.__send_act_and_wait() is called in a new Thread
        """
        if not self.act_threading:
            self.__send_act_and_wait(*args, **kwargs)
        else:
            self._at_thread = Thread(target=self.__send_act_and_wait, args=args, kwargs=kwargs)
            self._at_thread.start()

    def _initialize(self):
        """
        This is called at first reset() for rllib compatibility
        All costly initializations should be performed here
        This allows creating a dummy environment for retrieving action space and observation space without performing these initializations
        """
        self.initialized = True

    def _get_action_space(self):
        return self.interface.get_action_space()

    def _get_observation_space(self):
        # TODO: add action
        return self.interface.get_observation_space()

    def __send_act_and_wait(self, action):
        """
        This function applies the control
        If forced_sleep_time > 0, this will sleep for this duration after applying the action
        """
        if self.act_prepro_func:
            action = self.act_prepro_func(action)
        self.interface.send_control(action)
        if self.forced_sleep_time:
            time.sleep(self.forced_sleep_time)

    def _get_obs_rew_done(self, act):
        """
        Args:
            act: taken action (in the observation if act_in_obs is True)
        Returns:
            observation of this step()
        """
        o, r, d = self.interface.get_obs_rew_done()
        if not d:
            d = (self.current_step >= self.ep_max_length)
        elt = o
        if self.act_in_obs:
            elt = elt + [np.array(act), ]
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        elt = tuple(elt)
        return elt, r, d

    def reset(self):
        """
        Use reset() to reset the environment
        Returns:
            obs
        """
        self._join_act_thread()
        if not self.initialized:
            self._initialize()
        self.current_step = 0
        elt = self.interface.reset()
        if self.act_in_obs:
            elt = elt + [np.array(self.default_action), ]
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        elt = tuple(elt)
        return elt

    def step(self, action):
        """
        Call this function to perform a step
        Args:
            action: numpy.array: control value
        Returns:
            obs, rew, done, info

        CAUTION: the drone is only 'paused' at the end of the episode (the entire episode must be rolled out before optimizing if the optimization is synchronous)
        """
        self._join_act_thread()
        # t_s = time.time()
        self.current_step += 1
        if not self.real_time:
            self._run_time_step(action)
        obs, rew, done = self._get_obs_rew_done(action)  # TODO : threading must be modified so observation capture is handled correctly in the RTRL setting
        info = {}
        if self.real_time:
            self._run_time_step(action)
        if done:
            self.interface.wait()
        return obs, rew, done, info

    def stop(self):
        self._join_act_thread()

    def wait(self):
        self._join_act_thread()
        self.interface.wait()

    def render(self, mode='human'):
        """
        Visually renders the current state of the environment
        """
        self._join_act_thread()
        print("render")
        cv2.imshow("press q to exit", self.interface.img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()
