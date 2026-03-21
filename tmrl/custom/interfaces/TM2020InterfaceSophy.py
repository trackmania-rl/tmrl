"""
This module provides the TM2020InterfaceIMPALASophy class, which is a specialized interface
for the TrackMania 2020 game environment. It handles the extraction and parsing of
game data, the construction of observations for reinforcement learning, and the
computation of rewards and termination conditions, specifically tailored for the
Sophy-like model architecture.
"""

import numpy as np
from loguru import logger

import tmrl.config as cfg
from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.utils.control_mouse import mouse_save_replay_tm20

# Openplanet GrabData plugin indices (positional data from self.client.retrieve_data())
_IDX_CHECKPOINT = 0
_IDX_LAP = 1
_IDX_SPEED = 2
_IDX_POS_X = 3
_IDX_POS_Y = 4
_IDX_POS_Z = 5
_IDX_INPUT_STEER = 6
_IDX_INPUT_GAS = 7
_IDX_INPUT_BRAKE = 8
_IDX_END_OF_TRACK = 9
_IDX_ACCELERATION = 10
_IDX_JERK = 11
_IDX_AIM_YAW = 12
_IDX_AIM_PITCH = 13
_IDX_STEER_ANGLE_FL = 14
_IDX_STEER_ANGLE_FR = 15
_IDX_SLIP_COEF_FL = 16
_IDX_SLIP_COEF_FR = 17
_IDX_CRASHED = 18
_IDX_GEAR = 19


class TM2020InterfaceIMPALASophy(TM2020Interface):
    """
    Interface for TrackMania 2020 using a Sophy-like model architecture with IMPALA.

    This class extends TM2020Interface to provide a specific observation space
    and reward structure suitable for the Sophy-inspired reinforcement learning model.
    """

    def __init__(
        self,
        img_hist_len=1,
        gamepad=False,
        min_nb_steps_before_failure=160,
        record=False,
        save_replay: bool = False,
        save_replays: bool | None = None,
        grayscale: bool = False,
        resize_to: tuple = (128, 64),
        finish_reward=cfg.END_OF_TRACK_REWARD,
        constant_penalty: float = 0.05,
        crash_penalty=cfg.CRASH_PENALTY,
        checkpoint_reward=cfg.CHECKPOINT_REWARD,
        lap_reward=cfg.LAP_REWARD,
        record_human: bool = False,
        **kwargs,
    ):
        """
        Initializes the TM2020InterfaceIMPALASophy.

        Args:
            img_hist_len (int): Length of the image history. Defaults to 1.
            gamepad (bool): Whether to use a gamepad for input. Defaults to False.
            min_nb_steps_before_failure (int): Minimum steps before failure is considered.
                Defaults to 160.
            record (bool): Whether to record the session. Defaults to False.
            save_replay (bool): Whether to save a replay. Defaults to False.
            save_replays (bool, optional): Alias for save_replay. Defaults to None.
            grayscale (bool): Whether to use grayscale images. Defaults to False.
            resize_to (tuple): Dimensions to resize images to. Defaults to (128, 64).
            finish_reward (float): Reward for finishing the track.
                Defaults to cfg.END_OF_TRACK_REWARD.
            constant_penalty (float): Constant penalty per step. Defaults to 0.05.
            crash_penalty (float): Penalty for crashing. Defaults to cfg.CRASH_PENALTY.
            checkpoint_reward (float): Reward for passing a checkpoint.
                Defaults to cfg.CHECKPOINT_REWARD.
            lap_reward (float): Reward for completing a lap. Defaults to cfg.LAP_REWARD.
            record_human (bool): Whether to record human inputs. Defaults to False.
            **kwargs: Additional keyword arguments for the parent class.
        """
        if save_replays is not None:
            save_replay = save_replays
        super().__init__(
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replay,
            grayscale=grayscale,
            finish_reward=finish_reward,
            resize_to=resize_to,
            constant_penalty=constant_penalty,
            crash_penalty=crash_penalty,
            record_human=record_human,
            **kwargs,
        )
        self.record = record
        self.window_interface = None
        self.cur_lap = 0
        self.cur_checkpoint = 0
        self.lap_reward = lap_reward
        self.checkpoint_reward = checkpoint_reward
        self.points_number = cfg.POINTS_NUMBER
        # Spacing-based lookahead uses RewardFunction._points_number (trajectory length at load).
        # cfg.POINTS_NUMBER comes from the same file at import time but can theoretically differ
        # (e.g. reload / edge cases); mismatch breaks Tuple Box shapes vs get_track_info output and
        # shows up as physics_proj Linear in/out dim off by 1 when TRACK_CURVATURE_OBS is on.
        _rf = self.reward_function
        if (
            _rf is not None
            and float(getattr(_rf, "_point_spacing_m", 0) or 0) > 0
            and getattr(_rf, "_points_number", None) is not None
        ):
            n_rf_any = getattr(_rf, "_points_number", None)
            n_rf = self.points_number if n_rf_any is None else int(n_rf_any)
            if n_rf != self.points_number:
                logger.warning(
                    "POINTS_NUMBER from constants ({}) != reward spacing lookahead ({}); "
                    "using reward value for observation space.",
                    self.points_number,
                    n_rf,
                )
            self.points_number = n_rf

    def get_observation_space(self):
        """
        Returns the observation space for the environment.

        The observation space is a tuple of various game state metrics including
        track information, speed, acceleration, jerk, race progress, inputs,
        gear, orientation, steering angle, slip coefficients, and a failure counter.

        Returns:
            gymnasium.spaces.Tuple: The observation space.
        """
        from tmrl.custom.tm.tqc_observation_space import build_tqc_sophy_tuple_observation_space

        return build_tqc_sophy_tuple_observation_space(self.points_number)

    def _parse_data(self, data):
        """
        Parses raw data from the game client into a dictionary of named fields.

        Args:
            data (list): Raw data list from the game client.

        Returns:
            dict: A dictionary containing parsed game state information.
        """
        speed = np.array([data[_IDX_SPEED] * 3.6], dtype="float32")
        pos = np.array([data[_IDX_POS_X], data[_IDX_POS_Y], data[_IDX_POS_Z]], dtype="float32")
        return {
            "speed": speed,
            "pos": pos,
            "input_steer": np.array([data[_IDX_INPUT_STEER]], dtype="float32"),
            "input_gas_pedal": np.array([data[_IDX_INPUT_GAS]], dtype="float32"),
            "input_brake": np.array([data[_IDX_INPUT_BRAKE]], dtype="float32"),
            "acceleration": np.array([data[_IDX_ACCELERATION]], dtype="float32"),
            "jerk": np.array([data[_IDX_JERK]], dtype="float32"),
            "aim_yaw": np.array([data[_IDX_AIM_YAW]], dtype="float32"),
            "aim_pitch": np.array([data[_IDX_AIM_PITCH]], dtype="float32"),
            "steer_angle": np.array(
                [data[_IDX_STEER_ANGLE_FL], data[_IDX_STEER_ANGLE_FR]], dtype="float32"
            ),
            "slip_coef": np.array(
                [data[_IDX_SLIP_COEF_FL], data[_IDX_SLIP_COEF_FR]], dtype="float32"
            ),
            "gear": np.array([data[_IDX_GEAR]], dtype="float32"),
        }

    def _build_observation(self, d, race_progress, failure_counter, pos):
        """
        Assembles the observation list from parsed data fields and environment state.

        Args:
            d (dict): Parsed data dictionary.
            race_progress (float): Current race progress.
            failure_counter (float): Current failure counter value.
            pos (np.ndarray): Current position of the vehicle.

        Returns:
            list: A list of NumPy arrays forming the observation.
        """
        track_result = self.reward_function.get_track_info(pos, self.points_number)
        left_track = track_result[0]
        center_track = track_result[1]
        right_track = track_result[2]
        curvature_list = track_result[3] if len(track_result) == 4 else None

        if bool(cfg.REWARD_CONFIG.get("TRACK_LOCAL_FRAME", False)):
            yaw = float(d["aim_yaw"][0])
            c, s = np.cos(yaw), np.sin(yaw)
            rot_matrix = np.array(((c, -s), (s, c)))
            track_points = np.array(
                left_track + center_track + right_track, dtype="float32"
            ).reshape(-1, 2)
            rotated_points = track_points @ rot_matrix.T
            track = rotated_points.flatten().astype("float32")
        else:
            track = np.array(left_track + center_track + right_track, dtype="float32")

        obs = [
            track,
            d["speed"],
            d["acceleration"],
            d["jerk"],
            np.array([race_progress], dtype="float32"),
            d["input_steer"],
            d["input_gas_pedal"],
            d["input_brake"],
            d["gear"],
            d["aim_yaw"],
            d["aim_pitch"],
            d["steer_angle"],
            d["slip_coef"],
            np.array([float(failure_counter)], dtype="float32"),
        ]
        if curvature_list is not None:
            curv = np.array(curvature_list, dtype="float32")
            curv = np.clip(curv * 10.0, -1.0, 1.0)
            obs.append(curv)
        return obs

    def grab_data(self):
        """
        Retrieves raw data from the game client.

        Returns:
            list: Raw data from the game client.
        """
        data = self.client.retrieve_data()
        return data

    def get_obs_rew_terminated_info(self):
        """
        Retrieves the current observation, reward, termination status, and info dictionary.

        Returns:
            tuple: A tuple containing (observation, reward, terminated, info).
        """
        data = self.grab_data()
        cur_cp = int(data[_IDX_CHECKPOINT])
        cur_lap = int(data[_IDX_LAP])
        end_of_track = bool(data[_IDX_END_OF_TRACK])

        self.is_crashed = bool(data[_IDX_CRASHED])
        self._last_speed_kmh = float(data[_IDX_SPEED] * 3.6)

        d = self._parse_data(data)

        rew, terminated, failure_counter, reward_sum = self.reward_function.compute_reward(
            pos=d["pos"],
            crashed=bool(self.is_crashed),
            speed=d["speed"][0],
            next_cp=self.cur_checkpoint < cur_cp,
            next_lap=self.cur_lap < cur_lap,
            end_of_track=end_of_track,
            input_brake=float(data[_IDX_INPUT_BRAKE]),
            aim_yaw=float(data[_IDX_AIM_YAW]),
            input_steer=float(data[_IDX_INPUT_STEER]),
            gear=float(data[_IDX_GEAR]),
        )

        self.cur_checkpoint = cur_cp
        self.cur_lap = cur_lap

        race_progress = self.reward_function.compute_race_progress()

        if end_of_track:
            terminated = True
            failure_counter = 0.0
            if self.save_replays:
                mouse_save_replay_tm20(True)

        self.reward_function.log_model_run(terminated=terminated, end_of_track=end_of_track)

        if not self.is_crashed:
            self.crash_cooldown -= 1

        total_obs = self._build_observation(d, race_progress, failure_counter, d["pos"])
        info = {"reward_sum": reward_sum, "end_of_track": bool(end_of_track)}
        if getattr(self.client, "_last_retrieve_invalid", False):
            terminated = True
            info["telemetry_invalid"] = True
        if getattr(self.client, "_last_retrieve_position_patched", False):
            info["position_patched"] = True
        reward = np.float32(rew)
        return total_obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observation.

        Args:
            seed (int, optional): Seed for the environment. Defaults to None.
            options (dict, optional): Additional options for reset. Defaults to None.

        Returns:
            tuple: A tuple containing (observation, info).
        """
        self.reset_common()
        data = self.grab_data()

        self.cur_lap = 0
        self.cur_checkpoint = 0

        d = self._parse_data(data)

        self.reward_function.reset()

        total_obs = self._build_observation(d, race_progress=0.0, failure_counter=0.0, pos=d["pos"])
        info = {"reward_sum": 0.0, "end_of_track": False}
        return total_obs, info
