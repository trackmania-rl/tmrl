"""
This module provides the base rtgym interfaces for TrackMania 2020.
It includes classes for handling game input, extracting game state (images, lidar, data),
and computing rewards and termination conditions.
"""

import platform
import time
from collections import deque

import cv2
import numpy as np
from gymnasium import spaces
from loguru import logger
from rtgym import RealTimeGymInterface

import tmrl.config as cfg
from tmrl.custom.tm.utils.auto_drift import compute_drift_steer, is_auto_drift_action
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import (
    control_gamepad,
    gamepad_close_finish_pop_up_tm20,
    gamepad_reset,
)
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.control_mouse import (
    mouse_close_finish_pop_up_tm20,
    mouse_save_replay_tm20,
)
from tmrl.custom.tm.utils.discrete_control import (
    BRAKE_TAP_DURATION_S,
    build_yosh_action_table,
    discrete_index_to_control,
    is_brake_tap,
)
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost
from tmrl.custom.tm.utils.window import WindowInterface

# Globals
CHECK_FORWARD = 500  # this allows (and rewards) 50m cuts


class TM2020Interface(RealTimeGymInterface):
    """
    Base API for controlling TrackMania 2020 via rtgym.

    This class handles image history, gamepad/keyboard control, and communication
    with the TrackMania game via OpenPlanet.
    """

    def __init__(
        self,
        img_hist_len: int = 4,
        gamepad: bool = True,
        save_replays: bool = False,
        grayscale: bool = True,
        resize_to=(64, 64),
        finish_reward=None,
        constant_penalty=None,
        crash_penalty=None,
        min_nb_steps_before_failure=None,
        record_human: bool = False,
        **kwargs,
    ):
        """
        Initializes the TM2020Interface.

        Args:
            img_hist_len (int): Number of history images to include in observations. Defaults to 4.
            gamepad (bool): Whether to use a virtual gamepad for control. Defaults to True.
            save_replays (bool): Whether to save TrackMania replays on successful episodes.
                Defaults to False.
            grayscale (bool): Whether to output grayscale images. Defaults to True.
            resize_to (tuple): (width, height) to resize output images. Defaults to (64, 64).
            finish_reward (float, optional): Reward for finishing the track.
                Defaults to cfg.END_OF_TRACK_REWARD.
            constant_penalty (float, optional): Penalty per step. Defaults to 0.0.
            crash_penalty (float, optional): Penalty for crashing. Defaults to 10.0.
            min_nb_steps_before_failure (int, optional): Minimum steps before failure conditions
                apply. Defaults to 70.
            record_human (bool): Whether to allow human control for recording. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        self.is_crashed = None
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.record_human = record_human
        self.small_window = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = finish_reward if finish_reward is not None else cfg.END_OF_TRACK_REWARD
        self.constant_penalty = (
            constant_penalty
            if constant_penalty is not None
            else cfg.REWARD_CONFIG.get("CONSTANT_PENALTY", 0.0)
        )
        self.initialized = False
        self.crash_penalty = (
            crash_penalty
            if crash_penalty is not None
            else cfg.REWARD_CONFIG.get("CRASH_PENALTY", 10.0)
        )
        # Config overrides
        _default_min_steps = (
            min_nb_steps_before_failure if min_nb_steps_before_failure is not None else 70
        )
        self.min_nb_steps_before_failure = cfg.REWARD_CONFIG.get("MIN_STEPS", _default_min_steps)
        self.crash_cooldown = 0
        self.crash_curr = 0
        _alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        self.discrete_action_table: list[np.ndarray] | None = None
        if _alg_cfg.get("ALGORITHM") in ("IQN", "SDSAC"):
            _n_steer = int(_alg_cfg.get("IQN_N_STEER_BINS", 13))
            _, self.discrete_action_table = build_yosh_action_table(n_steer=_n_steer)
        self._send_control_logged = False
        self._img_buf: np.ndarray | None = None
        self._img_hist_count = 0
        self._img_hist_cursor = 0
        self._speed_arr = np.zeros((1,), dtype=np.float32)
        self._gear_arr = np.zeros((1,), dtype=np.float32)
        self._rpm_arr = np.zeros((1,), dtype=np.float32)
        self._last_speed_kmh: float = 0.0

    def _push_img(self, img: np.ndarray) -> None:
        """
        Pushes a new image into the history buffer.

        Args:
            img (np.ndarray): The new image to add.
        """
        if self._img_buf is None or self._img_buf.shape[1:] != img.shape:
            self._img_buf = np.zeros((self.img_hist_len, *img.shape), dtype=img.dtype)
            self._img_hist_count = 0
            self._img_hist_cursor = 0
        assert self._img_buf is not None
        self._img_buf[self._img_hist_cursor] = img
        self._img_hist_cursor = (self._img_hist_cursor + 1) % self.img_hist_len
        if self._img_hist_count < self.img_hist_len:
            self._img_hist_count += 1

    def _get_img_hist_array(self) -> np.ndarray:
        """
        Retrieves the current image history as a single NumPy array.

        Returns:
            np.ndarray: Array of history images.
        """
        if self._img_buf is None or self._img_hist_count == 0:
            return np.zeros((self.img_hist_len, 1, 1), dtype=np.uint8)
        if self._img_hist_count < self.img_hist_len:
            # Repeat first frame until buffer is full
            res = np.repeat(self._img_buf[:1], self.img_hist_len, axis=0)
            res[-self._img_hist_count :] = self._img_buf[: self._img_hist_count]
            return res
        if self._img_hist_cursor == 0:
            return self._img_buf.copy()
        idx = (
            np.arange(self.img_hist_len, dtype=np.int64) + self._img_hist_cursor
        ) % self.img_hist_len
        return self._img_buf[idx]

    def initialize_common(self):
        """Initializes the window interface, reward function, and game client."""
        if self.gamepad:
            try:
                import vgamepad as vg

                self.j = vg.VX360Gamepad()
                self.j.register_notification(callback_function=self.crash_callback)
                logger.info("Virtual gamepad (Xbox 360) initialized for control.")
            except OSError as e:
                if "libevdev" in str(e) or "libevdev" in str(e.__cause__ or ""):
                    raise RuntimeError(
                        "Virtual gamepad (vgamepad) requires libevdev on Linux. "
                        "Worker likely in WSL while TrackMania runs on Windows."
                    ) from e
                raise
            except Exception as e:
                err_msg = str(e).lower()
                if platform.system() == "Windows" and (
                    "vigem" in err_msg or "driver" in err_msg or "device" in err_msg
                ):
                    raise RuntimeError(
                        "Virtual gamepad failed on Windows. Install ViGEmBus driver."
                    ) from e
                raise
        else:
            logger.info("Using keyboard for control (VIRTUAL_GAMEPAD=false).")
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self._img_buf = None
        self._img_hist_count = 0
        self._img_hist_cursor = 0
        self.reward_function = RewardFunction(
            reward_data_path=cfg.REWARD_PATH,
            nb_obs_forward=cfg.REWARD_CONFIG.get("CHECK_FORWARD", 500),
            nb_obs_backward=cfg.REWARD_CONFIG.get("CHECK_BACKWARD", 10),
            min_nb_steps_before_failure=self.min_nb_steps_before_failure,
            max_dist_from_traj=cfg.REWARD_CONFIG.get("MAX_STRAY", 50.0),
            crash_penalty=self.crash_penalty,
            constant_penalty=self.constant_penalty,
        )
        if self.client is None:
            self.client = TM2020OpenPlanetClient()
        self.is_crashed = False
        self.crash_cooldown = 0
        self.crash_curr = self.crash_cooldown

    def crash_callback(self, client, target, large_motor, small_motor, led_number, user_data):
        """Callback for detecting crashes via gamepad vibration."""
        self.is_crashed = large_motor > 100 and self.crash_cooldown <= 0
        if self.is_crashed:
            logger.debug("crashed: True (episode will terminate)")
            self.crash_cooldown = 10

    def initialize(self):
        """Calls initialize_common and sets the interface as initialized."""
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        """
        Applies the action given by the RL policy.

        Handles three brake modes when using discrete actions:
          - 0.0: no brake
          - 1.0: full brake for the whole tick
          - BRAKE_TAP_SENTINEL: 0.01 s pulse then release

        Args:
            control (np.ndarray): [gas, brake, steer] or discrete index.
        """
        if self.record_human:
            return
        if control is not None and self.discrete_action_table is not None:
            idx = int(np.asarray(control).flat[0])
            control = discrete_index_to_control(idx, self.discrete_action_table)
        if control is not None and is_auto_drift_action(control):
            drift_steer = compute_drift_steer(self._last_speed_kmh)
            control = control.copy()
            control[2] = drift_steer
        if self.gamepad:
            if control is not None:
                if self.j is None:
                    logger.error("Virtual gamepad is None; cannot send control.")
                    return
                c = np.asarray(control, dtype=np.float32).ravel()
                control = c
                if not self._send_control_logged:
                    self._send_control_logged = True
                    gas = float(control[0]) if len(control) > 0 else 0
                    brake = float(control[1]) if len(control) > 1 else 0
                    steer = float(control[2]) if len(control) > 2 else 0
                    logger.info(
                        f"First send_control: gas={gas:.2f} brake={brake:.2f} "
                        f"steer={steer:.2f} (virtual gamepad)"
                    )
                if is_brake_tap(control):
                    tap_ctrl = control.copy()
                    tap_ctrl[1] = 1.0
                    control_gamepad(self.j, tap_ctrl)
                    time.sleep(BRAKE_TAP_DURATION_S)
                    tap_ctrl[1] = 0.0
                    control_gamepad(self.j, tap_ctrl)
                else:
                    control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append("f")
                if control[1] > 0:
                    actions.append("b")
                if control[2] > 0.5:
                    actions.append("r")
                elif control[2] < -0.5:
                    actions.append("l")
                apply_control(actions)

    def grab_data_and_img(self):
        """
        Retrieves a screenshot and game state data.

        Returns:
            tuple: (data, img)
        """
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to)
        img = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self.grayscale else img[:, :, ::-1]
        )  # RGB convention
        data = self.client.retrieve_data()
        self.img = img
        return data, img

    def reset_race(self):
        """Resets the race via gamepad or keyboard."""
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        """Common reset logic including control reset and race restart."""
        if not self.initialized:
            self.initialize()
        if self.record_human:
            self.record_human = False
            self.send_control(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            self.record_human = True
        else:
            self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = (
            max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        )
        time.sleep(time_sleep)

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation.

        Returns:
            tuple: (observation, info)
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        self._speed_arr[0] = data[0]
        self._gear_arr[0] = data[9]
        self._rpm_arr[0] = data[10]
        self._last_speed_kmh = float(data[0])
        for _ in range(self.img_hist_len):
            self._push_img(img)
        imgs = self._get_img_hist_array()
        obs = [self._speed_arr.copy(), self._gear_arr.copy(), self._rpm_arr.copy(), imgs]
        self.reward_function.reset()
        return obs, {}

    def close_finish_pop_up_tm20(self):
        """Closes the finish pop-up window in the game."""
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        """Pauses the interface and waits for the next episode."""
        self.send_control(self.get_default_action())
        if self.save_replays:
            save_ghost()
            time.sleep(1.0)
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        """
        Retrieves the current observation, reward, and termination status.

        Returns:
            tuple: (observation, reward, terminated, info)
        """
        data, img = self.grab_data_and_img()
        self._speed_arr[0] = data[0]
        self._gear_arr[0] = data[9]
        self._rpm_arr[0] = data[10]
        self._last_speed_kmh = float(data[0])
        reward, terminated, _failure_counter, _ = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]])
        )
        self._push_img(img)
        imgs = self._get_img_hist_array()
        observation = [self._speed_arr.copy(), self._gear_arr.copy(), self._rpm_arr.copy(), imgs]
        end_of_track = bool(data[8])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            terminated = True
            reward += self.finish_reward
            if self.save_replays:
                mouse_save_replay_tm20(True)
        reward = np.float32(reward)
        return observation, reward, terminated, info

    def get_observation_space(self) -> spaces.Tuple:
        """Returns the Gymnasium observation space."""
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        gear = spaces.Box(low=0.0, high=6, shape=(1,))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """Returns the Gymnasium action space."""
        if self.discrete_action_table is not None:
            return spaces.Discrete(len(self.discrete_action_table))
        return spaces.Box(low=-1.0, high=1.0, shape=(3,))

    def get_default_action(self):
        """Returns the default action at episode start."""
        if self.discrete_action_table is not None:
            return np.array(0, dtype=np.int64)
        return np.array([0.0, 0.0, 0.0], dtype="float32")


class TM2020InterfaceLidar(TM2020Interface):
    """
    Interface for TrackMania 2020 using LIDAR-like observations.
    """

    def __init__(self, img_hist_len=1, gamepad=False, save_replays: bool = False):
        super().__init__(img_hist_len, gamepad, save_replays)
        self.window_interface = None
        self.lidar = None

    def grab_lidar_speed_and_data(self):
        """Retrieves LIDAR, speed, and raw data."""
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([data[0]], dtype="float32")
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def reset(self, seed=None, options=None):
        self.reset_common()
        img, speed, _data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype="float32")
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]])
        )[:2]
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype="float32")
        obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19))
        return spaces.Tuple((speed, imgs))


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):
    """
    Interface for TrackMania 2020 using LIDAR and race progress observations.
    """

    def reset(self, seed=None, options=None):
        self.reset_common()
        img, speed, _data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype="float32")
        progress = np.array([0], dtype="float32")
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]])
        )[:2]
        progress = np.array(
            [self.reward_function.cur_idx / self.reward_function.datalen], dtype="float32"
        )
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype="float32")
        obs = [speed, progress, imgs]
        end_of_track = bool(data[8])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19))
        return spaces.Tuple((speed, progress, imgs))
