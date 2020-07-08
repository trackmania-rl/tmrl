# This is an environment for trackmania

from gym import Env
import gym.spaces as spaces
import numpy as np
import time
from threading import Thread

import cv2
import mss
import sys

# from pynput.keyboard import Key, Controller
import ctypes

SendInput = ctypes.windll.user32.SendInput

# constants:

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

# C struct redefinitions

PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Key Functions


def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def apply_control(action):
    if "forward" in action:
        PressKey(W)
    else:
        ReleaseKey(W)
    if "backward" in action:
        PressKey(S)
    else:
        ReleaseKey(S)
    if "left" in action:
        PressKey(A)
    else:
        ReleaseKey(A)
    if "right" in action:
        PressKey(D)
    else:
        ReleaseKey(D)


def load_digits():
    zero = cv2.imread('digits/0.png', 0)
    One = cv2.imread('digits/1.png', 0)
    Two = cv2.imread('digits/2.png', 0)
    Three = cv2.imread('digits/3.png', 0)
    four = cv2.imread('digits/4.png', 0)
    five = cv2.imread('digits/5.png', 0)
    six = cv2.imread('digits/6.png', 0)
    seven = cv2.imread('digits/7.png', 0)
    eight = cv2.imread('digits/8.png', 0)
    nine = cv2.imread('digits/9.png', 0)
    digits = np.array([zero, One, Two, Three, four, five, six, seven, eight, nine])
    return digits


def get_speed(img, digits):
    img1 = np.array(img[464:, 887:908])
    img2 = np.array(img[464:, 909:930])
    img3 = np.array(img[464:, 930:951])

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1 > 250] = 255
    img1[img1 <= 250] = 0
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2[img2 > 250] = 255
    img2[img2 <= 250] = 0
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img3[img3 > 250] = 255
    img3[img3 <= 250] = 0
    # compare digit with the others mean iou
    best1 = 100000000
    best2 = 100000000
    best3 = 100000000
    for idx, num in enumerate(digits):
        if np.sum(np.bitwise_xor(img1, num)) < best1:
            best1 = np.sum(np.bitwise_xor(img1, num))
            num1 = idx
        if np.sum(np.bitwise_xor(img2, num)) < best2:
            best2 = np.sum(np.bitwise_xor(img2, num))
            num2 = idx
        if np.sum(np.bitwise_xor(img3, num)) < best3:
            best3 = np.sum(np.bitwise_xor(img3, num))
            num3 = idx
        if np.max(img1) == 0:
            best1 = 0
            num1 = 0
        if np.max(img2) == 0:
            best2 = 0
            num2 = 0
        if np.max(img3) == 0:
            best3 = 0
            num3 = 0
    speed = 100 * num1 + 10 * num2 + num3
    return speed


class TMInterface():
    """
    This is the API needed for the algorithm to control Trackmania
    """
    def __init__(self):
        """
        Args:
        """
        self.monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
        self.sct = mss.mss()
        self.last_time = time.time()
        self.digits = load_digits()
        self.img = None  # for render()

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        Args:
            control: np.array: [forward,backward,right,left]
        """
        actions = []
        if control[0] > 0.5:
            actions.append("forward")
        elif control[1] > 0.5:
            actions.append("backward")
        if control[2] > 0.5:
            actions.append("right")
        elif control[1] > 0.5:
            actions.append("left")
        apply_control(actions)  # TODO: format this

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
        img = np.asarray(self.sct.grab(self.monitor))
        speed, img = get_speed(img, self.digits)
        self.img = img  # for render()
        rew = speed
        obs = {'speed': speed,
               'img': img}
        done = False  # TODO: True if race complete
        return obs, rew, done

    def get_observation_space(self):
        elt = {}
        elt['speed'] = spaces.Box(low=0, high=1000, shape=(1,))
        elt['img'] = spaces.Box(low=0, high=255, shape=(self.monitor["width"], self.monitor["height"],))
        return spaces.Dict(elt)

    def get_action_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(4,))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0, 0.0])


def numpyze_dict(d):
    for k, i in d.items():
        d[k] = np.array(i)


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
        obs, rew, done = self.get_obs_rew_done(action)  # TODO : threading must be modified so observation capture is handled correctly in the RTRL delay
        info = {}
        if self.real_time:
            self._run_time_step(action)
        if self.current_step >= self.ep_max_length:
            self.interface.wait()
        return obs, rew, done, info

    def stop(self):
        self._join_act_thread()

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
