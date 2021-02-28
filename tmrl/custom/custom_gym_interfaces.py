# rtgym interfaces for Trackmania

import gym.spaces as spaces
import numpy as np
import time
import cv2
import mss
from collections import deque

import platform
if platform.system() == "Windows":
    import vgamepad as vg

from rtgym import RealTimeGymInterface

from tmrl.custom.utils.key_event import apply_control, keyres
from tmrl.custom.utils.control_gamepad import control_gamepad
from tmrl.custom.utils.tools import load_digits, get_speed, Lidar, TM2020OpenPlanetClient
from tmrl.custom.utils.mouse_event import mouse_close_finish_pop_up_tm20
from tmrl.custom.utils.compute_reward import RewardFunction
from tmrl.custom.utils.drone_interface import DroneUDPInterface1

import tmrl.custom.config_constants as cfg

# Globals ==============================================================================================================

NB_OBS_FORWARD = 500  # if reward is collected at 100Hz, this allows (and rewards) 5s cuts

# Interface for Trackmania 2020 ========================================================================================


class TM2020Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control Trackmania2020
    """
    def __init__(self, img_hist_len: int=4, gamepad: bool=False, min_nb_steps_before_early_done: int=int(3.5 * 20)):
        """
        Args:
        """
        self.monitor = {"top": 32, "left": 1, "width": 256, "height": 127}
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
        self.min_nb_steps_before_early_done = min_nb_steps_before_early_done

        self.initialized = False

    def initialize(self):
        if self.gamepad:
            assert platform.system() == "Windows", "Sorry, Only Windows is supported for gamepad control"
            self.j = vg.VX360Gamepad()
            print("DEBUG: virtual joystick in use")
        self.sct = mss.mss()
        self.last_time = time.time()
        self.digits = load_digits()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,
                                              nb_obs_forward=NB_OBS_FORWARD,
                                              nb_obs_backward=10,
                                              nb_zero_rew_before_early_done=10,
                                              min_nb_steps_before_early_done=self.min_nb_steps_before_early_done)
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
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions)

    def grab_data_and_img(self):
        img = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        img = np.moveaxis(img, -1, 0)
        data = self.client.retrieve_data()
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
        time.sleep(1.5)  # must be long enough for image to be refreshed
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
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
        mouse_close_finish_pop_up_tm20(small_window=True)

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        rew, done = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        rew = np.float32(rew)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        done = done or bool(data[8])
        return obs, rew, done

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, 3, 127, 256))
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, min_nb_steps_before_early_done=int(20 * 3.5), road_point=(440, 479), record=False):
        super().__init__(img_hist_len, gamepad, min_nb_steps_before_early_done)
        self.monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
        self.lidar = Lidar(monitor=self.monitor, road_point=road_point)
        self.record = record

    def grab_lidar_speed_and_data(self):
        img = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([
            data[0],
        ], dtype='float32')
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
        time.sleep(1.5)  # must be long enough for image to be refreshed
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs  # if not self.record else data

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        keyres()
        time.sleep(0.5)
        mouse_close_finish_pop_up_tm20(small_window=False)

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, done = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        rew = np.float32(rew)
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        done = done or bool(data[8])
        return obs, rew, done  # if not self.record else data, rew, done

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
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
            elif control[2] < -0.5:
                actions.append('l')
            apply_control(actions)

    def grab_img_and_speed(self):
        img = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        speed = np.array([
            get_speed(img, self.digits),
        ], dtype='float32')
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
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, 50, 190, 3))
        return spaces.Tuple((speed, imgs))

    def get_action_space(self):
        """
        must be a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))  # 1=f; 1=b; -1=l,+1=r

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
        speed = np.array([
            get_speed(img, self.digits),
        ], dtype='float32')
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
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, imgs))


# Interface for the Cognifly robot: ====================================================================================


class CogniflyInterfaceTask1(RealTimeGymInterface):

    # Cognifly UDP controller: cognifly_vel_controller.py
    # messages sent to Cognifly: [vel, arm, time_step_id]
    # messages sent by Cognifly: [alt, vel, acc, ubatt, time_step_id]
    # Cognifly config: balloon3_debug_agl

    def __init__(self, img_hist_len=4):
        """
        Args:
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.udpi = None
        self.initialized = False
        self.drone_int = None
        self.min_target_alt = 40
        self.max_target_alt = 70
        self.target = self.min_target_alt

    def initialize(self):
        self.drone_int = DroneUDPInterface1(
            udp_send_ip="192.168.0.201",  # Cognifly IP
            udp_recv_ip="192.168.0.200",  # Local IP
            udp_send_port=55558,  # Cognifly port
            udp_recv_port=55557,  # Local port
            min_altitude=0.0,
            max_altitude=100.0,
            low_batt=7.5)
        self.drone_int.arm_disarm(arm=True, wait_time=2.0)
        self.drone_int.take_off(takeoff_vel=100.0, target_alt=40.0, sleep_time=0.1)
        self.initialized = True

    def send_control(self, control):
        self.drone_int.send_control(control[0] * 1000.0, time.time())  # this env uses a normalized action-space

    def reset(self):
        """
        obs must be a list of numpy arrays
        """
        if not self.initialized:
            self.initialize()
            self.initialized = True
        self.target = np.random.uniform(self.min_target_alt, self.max_target_alt)
        self.drone_int.update()
        obs = self.drone_int.read_obs()
        # print(f"reset drone obs: {obs}")
        return [
            np.array([
                obs[0],
            ], dtype=np.float32),
            np.array([
                obs[1],
            ], dtype=np.float32),
            np.array([
                obs[2],
            ], dtype=np.float32),
            np.array([
                self.target,
            ], dtype=np.float32),
            np.array([
                0.0,
            ], dtype=np.float32),
            np.array([
                0.0,
            ], dtype=np.float32),
        ]

    def wait(self):
        self.send_control(self.get_default_action())

    def get_obs_rew_done(self):
        """
        returns the observation, the reward, and a done signal for end of episode
        obs must be a list of numpy arrays
        """
        self.drone_int.update()
        obs = self.drone_int.read_obs()
        # print(f"obs:{obs}")
        t_now = time.time()
        o = [
            np.array([
                obs[0],
            ], dtype=np.float32),
            np.array([
                obs[1],
            ], dtype=np.float32),
            np.array([
                obs[2],
            ], dtype=np.float32),
            np.array([
                self.target,
            ], dtype=np.float32),
            np.array([
                t_now - obs[4],
            ], dtype=np.float32),
            np.array([
                t_now - obs[5],
            ], dtype=np.float32),
        ]
        r = -np.float32(abs(self.target - obs[0]))
        d = (r >= -0.5)
        return o, r, d

    def get_observation_space(self):
        """
        must be a Tuple
        """
        alt = spaces.Box(low=0.0, high=100.0, shape=(1, ))
        vel = spaces.Box(low=-1000.0, high=1000.0, shape=(1, ))
        acc = spaces.Box(low=-1000.0, high=1000.0, shape=(1, ))
        tar = spaces.Box(low=40.0, high=70.0, shape=(1, ))
        total_delay = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        total_delay_kappa = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        return spaces.Tuple((alt, vel, acc, tar, total_delay, total_delay_kappa))

    def get_action_space(self):
        """
        must return a Box
        """
        vel = spaces.Box(low=-1.0, high=1.0, shape=(1, ))
        return vel

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([
            0.0,
        ], dtype='float32')


if __name__ == "__main__":
    pass
