# This is an environment for trackmania
# http://www.flint.jp/misc/?q=dik&lang=en  key indicator

from gym import Env
import gym.spaces as spaces
import numpy as np
import time
from threading import Thread, Lock
import cv2
import mss
import sys
from gym_tmrl.envs.tools import load_digits, get_speed
from gym_tmrl.envs.key_event import apply_control, keyres
from collections import deque
import socket
import struct
from threading import Thread, Lock
from gym_tmrl.envs.compute_reward import RewardFunction

# from pynput.keyboard import Key, Controller
import ctypes

# Globals ==============================================================================================================

REWARD_PATH = r"D:\data2020reward\reward_mod.pkl"
NB_OBS_FORWARD = 500  # if reward is collected at 100Hz, this allows (and rewards) 5s cuts

# Interface for Trackmania 2020 ========================================================================================

class TM2020OpenPlanetClient:
    def __init__(self,
                 host='127.0.0.1',
                 port=9000):
        self._host = host
        self._port = port

        # Threading attributes:
        self.__lock = Lock()
        self.__data = None
        self.__t_client = Thread(target=self.__client_thread, args=(), kwargs={}, daemon=True)
        self.__t_client.start()

    def __client_thread(self):
        """
        Thread of the client.
        This listens for incoming data until the object is destroyed
        TODO: handle disconnection
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            data_raw = b''
            while True:  # main loop
                while len(data_raw) < 32:
                    data_raw += s.recv(1024)
                div = len(data_raw) // 32
                data_used = data_raw[(div - 1) * 32:div * 32]
                data_raw = data_raw[div * 32:]
                self.__lock.acquire()
                self.__data = data_used
                self.__lock.release()

    def retrieve_data(self, sleep_if_empty=0.1):
        """
        Retrieves the most recently received data
        Use this function to retrieve the most recently received data
        If block if nothing has been received so far
        """
        c = True
        while c:
            self.__lock.acquire()
            if self.__data is not None:
                data = struct.unpack('<ffffffff', self.__data)
                c = False
            self.__lock.release()
            if c:
                time.sleep(sleep_if_empty)
        return data


class TM2020Interface:
    """
    This is the API needed for the algorithm to control Trackmania2020
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
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=REWARD_PATH, nb_obs_forward=NB_OBS_FORWARD)
        self.client = TM2020OpenPlanetClient()

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        Args:
            control: np.array: [forward,backward,right,left]
        TODO update
        """
        if control is not None:
            actions = []
            if control[0] > 0.5:
                actions.append("f")
            elif control[1] > 0.5:
                actions.append("b")
            if control[2] > 0.5:
                actions.append("r")
            elif control[2] < -0.5:
                actions.append("l")
            apply_control(actions)  # TODO: format this

    def grab_data_and_img(self):
        img = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        data = self.client.retrieve_data()
        img = cv2.resize(img, (191, 98))
        self.img = img  # for render()
        return data, img

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.send_control([0, 0, 0])
        keyres()
        time.sleep(0.05)  # must be long enough for image to be refreshed
        data, img = self.grab_data_and_img()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array([i for i in self.img_hist])
        obs = [data, imgs]
        self.reward_function.reset()
        return obs

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        apply_control([0, 0, 0])

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        rew = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array([i for i in self.img_hist])
        obs = [data, imgs]
        done = False  # TODO: True if race complete
        return obs, rew, done

    def get_observation_space(self):
        """
        must be a Tuple
        TODO: update
        """
        speed = spaces.Box(low=0.0, high=1.0, shape=(1,))
        img = spaces.Box(low=0.0, high=1.0, shape=(self.img_hist_len, 3, 48, 191))
        return spaces.Tuple((speed, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=0.0, high=1.0, shape=(3,))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0])


# Interface for Trackmania Nations Forever: ============================================================================

class TMInterface:
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
        img = img[100:-150, :]
        img = cv2.resize(img, (190, 50))
        img = np.moveaxis(img, -1, 0)
        img = img.astype('float32') / 255.0 - 0.5  # normalized and centered
        #print(f"DEBUG: Env: captured speed:{speed}")
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
        img = spaces.Box(low=0.0, high=1.0, shape=(self.img_hist_len, 3, 50, 190))
        return spaces.Tuple((speed, img))

    def get_action_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(4,))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0, 0.0])


# General purpose environment: =========================================================================================

DEFAULT_CONFIG_DICT = {
    # "interface": TMInterface,
    "interface": TM2020Interface,
    "time_step_duration": 0.05,
    "start_obs_capture": 0.04,
    "time_step_timeout_factor": 1.0,
    "ep_max_length": np.inf,
    "real_time": True,
    "async_threading": True,
    "act_in_obs": True,
    "benchmark": True
}


class TMRLEnv(Env):
    def __init__(self, config=DEFAULT_CONFIG_DICT):
        """
        :param interface: (callable) external interface class (required)
        :param ep_max_length: (int) the max length of each episodes in timesteps
        :param real_time: bool: whether to use the RTRL setting
        :param async_threading: bool (optional, default: True): whether actions are executed asynchronously in the RTRL setting.
            Typically this is useful for the real world and for external simulators
        :param time_step_duration: float (optional, default 0.0): seconds slept after apply_action() (~ time-step duration)
        :param act_in_obs: bool (optional, default True): whether to augment the observation with the action buffer (DCRL)
        :param default_action: float (optional, default None): default action to append at reset when the previous is True
        :param act_prepro_func: function (optional, default None): function that maps the action input to the actual applied action
        :param obs_prepro_func: function (optional, default None): function that maps the observation output to the actual returned observation
        :param reset_act_buf: bool (optional, defaut True): whether action buffer should be re-initialized at reset
        """
        # interface:
        interface_cls = config["interface"]
        self.interface = interface_cls()

        # config variables:
        self.act_prepro_func: callable = config["act_prepro_func"] if "act_prepro_func" in config else None
        self.obs_prepro_func = config["obs_prepro_func"] if "obs_prepro_func" in config else None
        self.ep_max_length = config["ep_max_length"]

        self.time_step_duration = config["time_step_duration"] if "time_step_duration" in config else 0.0
        self.time_step_timeout_factor = config["time_step_timeout_factor"] if "time_step_timeout_factor" in config else 1.0
        self.start_obs_capture = config["start_obs_capture"] if "start_obs_capture" in config else 1.0
        self.time_step_timeout = self.time_step_duration * self.time_step_timeout_factor  # time after which elastic time-stepping is dropped
        self.real_time = config["real_time"]
        self.async_threading = config["async_threading"] if "async_threading" in config else True
        self.__t_start = time.time()  # beginning of the time-step
        self.__t_co = time.time()  # time at which observation starts being captured during the time step
        self.__t_end = time.time()  # end of the time-step
        if not self.real_time:
            self.async_threading = False
        if self.async_threading:
            self._at_thread = Thread(target=None, args=(), kwargs={}, daemon=True)
            self._at_thread.start()  # dummy start for later call to join()

        # observation capture:
        self.__o_lock = Lock()  # lock to retrieve observations asynchronously, acquire to access the following:
        self.__obs = None
        self.__rew = None
        self.__done = None
        self.__o_set_flag = False

        # environment benchmark:
        self.benchmark = config["benchmark"] if "benchmark" in config else False
        self.__b_lock = Lock()
        self.__b_obs_capture_duration = 0.0
        self.running_average_factor = config["running_average_factor"] if "running_average_factor" in config else 0.1

        self.act_in_obs = config["act_in_obs"] if "act_in_obs" in config else True
        self.reset_act_buf = config["reset_act_buf"] if "reset_act_buf" in config else True
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.current_step = 0
        self.initialized = False
        # state variables:
        self.default_action = self.interface.get_default_action()
        self.last_action = self.default_action

    def _update_timestamps(self):
        """
        This is called at the beginning of each time-step
        If the previous time-step has timed out, the beginning of the time-step is set to now
        Otherwise, the beginning of the time-step is the beginning of the previous time-step + the time-step duration
        The observation starts being captured start_obs_capture_factor time-step after the beginning of the time-step
            observation capture can exceed the time-step, it is fine, but be cautious with timeouts
        It is recommended to draw a time diagram of your system
            action computation and observation capture can be performed in parallel
        """
        now = time.time()
        if now < self.__t_end + self.time_step_timeout:
            self.__t_start = self.__t_end
            self.__t_co = self.__t_start + self.start_obs_capture
            self.__t_end = self.__t_start + self.time_step_duration
        else:
            print(f"INFO: time-step timed out. Elapsed since last time-step: {now - self.__t_end}")
            self.__t_start = now
            self.__t_co = self.__t_start + self.start_obs_capture
            self.__t_end = self.__t_start + self.time_step_duration

    def _join_thread(self):
        """
        This is called at the beginning of every user-side API functions (step(), reset()...) for thread safety
        This ensures that the previous time-step is completed when starting a new one
        """
        if self.async_threading:
            self._at_thread.join()

    def _run_time_step(self, *args, **kwargs):
        """
        This is what must be called in step() to apply an action
        Call this with the args and kwargs expected by self.__send_act_and_wait()
        This in turn calls self.__send_act_and_wait()
        In action-threading, self.__send_act_and_wait() is called in a new Thread
        """
        if not self.async_threading:
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
        t = self.interface.get_observation_space()
        if self.act_in_obs:
            t = spaces.Tuple((*t.spaces, self._get_action_space()))
        print(t)
        return t

    def __send_act_and_wait(self, action):
        """
        This function applies the control and launches observation capture at the right timestamp
        !: only one such function must run in parallel (always join thread)
        """
        act = self.act_prepro_func(action) if self.act_prepro_func else action
        self.interface.send_control(act)
        self._update_timestamps()
        now = time.time()
        while now < self.__t_co:  # wait until it is time to capture observation
            now = time.time()
        self.__update_obs_rew_done(action)  # capture observation
        while now < self.__t_end:  # wait until the end of the time-step
            now = time.time()

    def __update_obs_rew_done(self, act):
        """
        Captures o, r, d asynchronously
        Args:
            act: taken action (in the observation if act_in_obs is True)
        Returns:
            observation of this step()
        """
        if self.benchmark:
            t1 = time.time()
        self.__o_lock.acquire()
        o, r, d = self.interface.get_obs_rew_done()
        if not d:
            d = (self.current_step >= self.ep_max_length)
        elt = o
        if self.act_in_obs:
            elt = elt + [np.array(act), ]
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        elt = tuple(elt)
        self.__obs, self.__rew, self.__done = elt, r, d
        self.__o_set_flag = True
        self.__o_lock.release()
        if self.benchmark:
            t = time.time() - t1
            self.__b_lock.acquire()
            self.__b_obs_capture_duration = (1.0 - self.running_average_factor) * self.__b_obs_capture_duration + self.running_average_factor * t
            self.__b_lock.release()

    def _retrieve_obs_rew_done(self):
        """
        Waits for new available o r d and retrieves them
        """
        c = True
        while c:
            self.__o_lock.acquire()
            if self.__o_set_flag:
                elt, r, d = self.__obs, self.__rew, self.__done
                self.__o_set_flag = False
                c = False
            self.__o_lock.release()
        return elt, r, d

    def reset(self):
        """
        Use reset() to reset the environment
        Returns:
            obs
        """
        self._join_thread()
        if not self.initialized:
            self._initialize()
        self.current_step = 0
        elt = self.interface.reset()
        if self.act_in_obs:
            elt = elt + [np.array(self.default_action), ]
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        elt = tuple(elt)
        if self.real_time:
            self._run_time_step(self.default_action)
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
        self._join_thread()
        self.current_step += 1
        if not self.real_time:
            self._run_time_step(action)
        obs, rew, done = self._retrieve_obs_rew_done()
        info = {}
        if self.real_time:
            self._run_time_step(action)
        if done:
            self.interface.wait()
        return obs, rew, done, info

    def stop(self):
        self._join_thread()

    def wait(self):
        self._join_thread()
        self.interface.wait()

    def benchmarks(self):
        """
        Returns the following running averages when the benchmark option is set:
            - duration of __update_obs_rew_done()
        """
        assert self.benchmark, "The benchmark option is not set. Set benchmark=True the configuration dictionary of the environment"
        self.__b_lock.acquire()
        res_obs_capture_duration = self.__b_obs_capture_duration
        self.__b_lock.release()
        return res_obs_capture_duration

    def render(self, mode='human'):
        """
        Visually renders the current state of the environment
        """
        self._join_thread()
        print("render")
        cv2.imshow("press q to exit", self.interface.img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()
