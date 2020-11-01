# This is an environment for Trackmania
# http://www.flint.jp/misc/?q=dik&lang=en  key indicator

import gym.spaces as spaces
import numpy as np
import time
import cv2
import mss
from collections import deque
import socket
import struct
from threading import Thread, Lock
# import pyvjoy  # CAUTION: not compatible with Linux

from rtgym import RealTimeGymInterface

from agents.custom.utils.key_event import apply_control, keyres
from agents.custom.utils.tools import load_digits, get_speed, Lidar
from agents.custom.utils.mouse_event import mouse_close_finish_pop_up_tm20
from agents.custom.utils.compute_reward import RewardFunction

import agents.custom.config as cfg

# from agents.custom.utils.gamepad_event import control_all


# from pynput.keyboard import Key, Controller

# Globals ==============================================================================================================

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
                while len(data_raw) < 36:
                    data_raw += s.recv(1024)
                div = len(data_raw) // 36
                data_used = data_raw[(div - 1) * 36:div * 36]
                data_raw = data_raw[div * 36:]
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
                data = struct.unpack('<fffffffff', self.__data)
                c = False
            self.__lock.release()
            if c:
                time.sleep(sleep_if_empty)
        return data


class TM2020Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control Trackmania2020
    """

    def __init__(self, img_hist_len=4, gamepad=False):
        """
        Args:
        """
        self.monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
        self.sct = None
        self.last_time = None
        self.digits = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        if self.gamepad:
            pass
        #     self.j = pyvjoy.VJoyDevice(1)
        #     print("DEBUG: virtual joystick in use")
        #     import signal
        #     import sys
        #
        #     def signal_handler(sig, frame):
        #
        #         self.j.reset()
        #         self.j.reset_buttons()
        #         self.j.reset_povs()
        #         control_all([0.0, 0.0, 0.0], self.j)
        #         print('You pressed Ctrl+C!')
        #         sys.exit(0)
        #
        #     signal.signal(signal.SIGINT, signal_handler)
        self.initialized = False

    def initialize(self):
        self.sct = mss.mss()
        self.last_time = time.time()
        self.digits = load_digits()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH, nb_obs_forward=NB_OBS_FORWARD)
        self.client = TM2020OpenPlanetClient()
        self.initialized = True

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [forward,backward,right,left]
        """
        if self.gamepad:
            pass
        #     if control is not None:
        #         control_all(control, self.j)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < - 0.5:
                    actions.append('l')
                apply_control(actions)

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
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        keyres()
        # time.sleep(0.05)  # must be long enough for image to be refreshed
        data, img = self.grab_data_and_img()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [data, imgs]
        self.reward_function.reset()
        return obs

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        keyres()
        time.sleep(0.5)
        mouse_close_finish_pop_up_tm20()

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        rew = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        rew = np.float32(rew)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [data, imgs]
        done = bool(data[8])  # FIXME: check that this works
        return obs, rew, done

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, 3, 48, 191))  # because the dataloader crops imgs
        return spaces.Tuple((speed, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3,))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, road_point=(440, 479), record=False):
        super().__init__(img_hist_len, gamepad)
        self.lidar = Lidar(monitor=self.monitor, road_point=road_point)
        self.record = record

    def grab_lidar_speed_and_data(self):
        img = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([data[0], ], dtype='float32')
        lidar = self.lidar.lidar_20(im=img, show=False)
        return lidar, speed, data

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        keyres()
        # time.sleep(0.05)  # must be long enough for image to be refreshed
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs  # if not self.record else data

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        rew = np.float32(rew)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        done = bool(data[8])
        if done:
            pass  # TODO: find a way to get rid of the annoying pop up
        # print(f"DEBUG: len(obs):{len(obs)}, obs[0]:{obs[0]}, obs[1].shape:{obs[1].shape}")
        return obs, rew, done  # if not self.record else data, rew, done

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19,))  # lidars
        return spaces.Tuple((speed, imgs))


# Interface for Trackmania Nations Forever: ============================================================================

class TMInterface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control Trackmania Nations Forever
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
        If control is None, does nothing
        Args:
            control: np.array: [forward,backward,right,left]
        """
        if control is not None:
            actions = []
            if control[0] > 0:
                actions.append('f')
            if control[1] > 0:
                actions.append('b')
            if control[2] > 0.5:
                actions.append('r')
            elif control[2] < - 0.5:
                actions.append('l')
            apply_control(actions)

    def grab_img_and_speed(self):
        img = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        speed = np.array([get_speed(img, self.digits), ], dtype='float32')
        img = img[100:-150, :]
        img = cv2.resize(img, (190, 50))
        # img = np.moveaxis(img, -1, 0)
        return img, speed

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.send_control(self.get_default_action())
        keyres()
        # time.sleep(0.05)  # must be long enough for image to be refreshed
        img, speed = self.grab_img_and_speed()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        return obs

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed = self.grab_img_and_speed()
        rew = speed[0]
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        done = False  # TODO: True if race complete
        # print(f"DEBUG: len(obs):{len(obs)}, obs[0]:{obs[0]}, obs[1].shape:{obs[1].shape}")
        return obs, rew, done

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, 50, 190, 3))
        return spaces.Tuple((speed, imgs))

    def get_action_space(self):
        """
        must be a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3,))  # 1=f; 1=b; -1=l,+1=r

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TMInterfaceLidar(TMInterface):
    def __init__(self, img_hist_len=4, road_point=(440, 479)):
        super().__init__(img_hist_len)
        self.lidar = Lidar(monitor=self.monitor, road_point=road_point)

    def grab_lidar_and_speed(self):
        img = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        speed = np.array([get_speed(img, self.digits), ], dtype='float32')
        lidar = self.lidar.lidar_20(im=img, show=False)
        return lidar, speed

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        self.send_control(self.get_default_action())
        keyres()
        # time.sleep(0.05)  # must be long enough for image to be refreshed
        img, speed = self.grab_lidar_and_speed()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        return obs

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed = self.grab_lidar_and_speed()
        rew = speed[0]
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        done = False  # TODO: True if race complete
        # print(f"DEBUG: len(obs):{len(obs)}, obs[0]:{obs[0]}, obs[1].shape:{obs[1].shape}")
        return obs, rew, done

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19,))  # lidars
        return spaces.Tuple((speed, imgs))
