# This is an environment for trackmania

from gym import Env
import gym.spaces as spaces
import numpy as np
import time
from threading import Thread
from copy import deepcopy


class TMInterface():
    """
    This is the API needed for the algorithm to control Trackmania
    """
    def __init__(self):
        """
        Args:
        """
        pass

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        Args:
        """
        pass

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        pass

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        """
        return None, 0.0, False  # TODO

    def get_observation_space(self):
        return None  # TODO

    def get_action_space(self):
        return None  # TODO

    def get_default_action(self):
        """
        initial action at episode start
        """
        return None  # TODO


def numpyze_dict(d):
    for k, i in d.items():
        d[k] = np.array(i)


class State(object):
    def __init__(self, state=None, rew=None):
        """
        state: object: instantaneous state of the universe
        rew: float: instantaneous reward
        """
        self.state = deepcopy(state)
        self.rew = rew


class TMRLEnv(Env):
    def __init__(self, config):
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
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.current_step = 0
        self.initialized = False
        self.interface = TMInterface()
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
        o, r = self.interface.get_obs_rew_done()
        d = (self.current_step >= self.ep_max_length)
        elt = {'obs': o}
        if self.act_in_obs:
            self.act_buffer.append(act)
            elt['act'] = np.array(self.act_buffer)
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        numpyze_dict(elt)
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
        self._reset_obj_alt()
        self._reset_states()
        self._reset_action_buffer()
        return self._get_obs(act=self.default_action)

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
        obs, rew, done = self.get_obs_rew_done(action)
        info = {}
        if self.real_time:
            self._run_time_step(action)
        if self.current_step >= self.ep_max_length:
            self.interface.wait()
        return obs, rew, done, info

    def stop(self):
        self._join_act_thread()

    def render(self, mode='pyplot', datadict=None, xlabel="", ylabel="", title="", figsize=(7, 5), save=False, fname=None, dpi=None):
        """
        Visually renders the current state of the environment using pyplot
        Datadic is to provide a dictionary containing data to print as a graph alongside the environment
        """
        self._join_act_thread()
        print("render")
