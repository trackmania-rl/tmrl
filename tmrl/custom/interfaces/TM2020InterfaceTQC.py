import cv2
import numpy as np
from gymnasium import spaces

import tmrl.config as cfg
from tmrl.custom.interfaces.TM2020InterfaceSophy import TM2020InterfaceIMPALASophy
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

TQC_GRAB_NB_FLOATS = 20
_DEFAULT_MIN_STEPS_END_OF_TRACK = 50


class TM2020InterfaceTQC(TM2020InterfaceIMPALASophy):
    """
    Interface for TrackMania 2020 using the TQC_GrabData plugin.

    This class manages the connection to the game via a specialized client
    and processes the 20-float data packets into observations for the RL agent.
    """

    def initialize_common(self):
        """
        Initializes the common components of the interface, including the game client.
        """
        self.client = TM2020OpenPlanetClient(port=9000, nb_floats=TQC_GRAB_NB_FLOATS)
        super().initialize_common()

    def get_obs_rew_terminated_info(self):
        """
        Retrieves the current observation, reward, termination status, and info dictionary.

        Returns:
            tuple: A tuple containing (observation, reward, terminated, info).
        """
        data = self.grab_data()
        self.is_crashed = bool(data[18])
        cur_cp = int(data[0])
        cur_lap = int(data[1])

        speed_kmh = data[2] * 3.6
        speed = np.array([speed_kmh / cfg.OBS_SPEED_SCALE], dtype="float32")
        pos = np.array([data[3], data[4], data[5]], dtype="float32")
        input_steer = np.array([data[6]], dtype="float32")
        input_gas_pedal = np.array([data[7]], dtype="float32")
        input_brake = np.array([data[8]], dtype="float32")
        end_of_track = bool(data[9])
        acceleration = np.array([data[10]], dtype="float32")
        jerk = np.array([data[11]], dtype="float32")
        aim_yaw = np.array([data[12]], dtype="float32")
        aim_pitch = np.array([data[13]], dtype="float32")
        steer_angle = np.array(data[14:16], dtype="float32")
        slip_coef = np.array(data[16:18], dtype="float32")
        gear = np.array([data[19] / 5.0], dtype="float32")

        rew, terminated, failure_counter, reward_sum = self.reward_function.compute_reward(
            pos=pos,
            crashed=self.is_crashed,
            speed=speed_kmh,
            next_cp=self.cur_checkpoint < cur_cp,
            next_lap=self.cur_lap < cur_lap,
            end_of_track=end_of_track,
            input_brake=float(data[8]),
            aim_yaw=float(data[12]),
            input_steer=float(data[6]),
            gear=float(data[19]),
            slip_angle_deg=None,
        )
        self._dbg_last_step = {
            "terminated": bool(terminated),
            "end_of_track": bool(end_of_track),
            "speed_kmh": float(speed_kmh),
            "reward_sum": float(reward_sum),
            "step_counter": int(getattr(self.reward_function, "step_counter", -1)),
        }

        race_progress = self.reward_function.compute_race_progress()

        self._steps_since_reset = getattr(self, "_steps_since_reset", 0) + 1
        min_steps_before_finish = max(
            _DEFAULT_MIN_STEPS_END_OF_TRACK,
            cfg.REWARD_CONFIG.get("MIN_STEPS", _DEFAULT_MIN_STEPS_END_OF_TRACK),
        )
        if end_of_track and self._steps_since_reset >= min_steps_before_finish:
            terminated = True
            failure_counter = 0.0
            if self.save_replays:
                from tmrl.custom.utils.control_mouse import mouse_save_replay_tm20

                mouse_save_replay_tm20(True)

        self.reward_function.log_model_run(terminated=terminated, end_of_track=end_of_track)

        track_result = self.reward_function.get_track_info(pos, self.points_number)
        left_track = track_result[0]
        center_track = track_result[1]
        right_track = track_result[2]
        curvature_list = track_result[3] if len(track_result) == 4 else None

        if bool(cfg.REWARD_CONFIG.get("TRACK_LOCAL_FRAME", False)):
            yaw = float(data[12])
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)

            def _rotate(pairs):
                rotated = []
                for j in range(0, len(pairs), 2):
                    dx, dz = pairs[j], pairs[j + 1]
                    rotated.append(dx * cos_y - dz * sin_y)
                    rotated.append(dx * sin_y + dz * cos_y)
                return rotated

            left_track = _rotate(left_track)
            center_track = _rotate(center_track)
            right_track = _rotate(right_track)

        track_list = left_track + center_track + right_track
        if cfg.OBS_TRACK_SCALE != 1.0:
            track_list = [x / cfg.OBS_TRACK_SCALE for x in track_list]
        track_info = [np.array(track_list, dtype="float32")]

        if not self.is_crashed:
            self.crash_cooldown -= 1

        race_progress = np.array([race_progress], dtype="float32")
        max_count = max(
            1.0,
            getattr(self.reward_function, "_max_no_progress_steps", 200.0),
        )
        failure_counter = np.array(
            [float(failure_counter) / max_count],
            dtype="float32",
        )
        info = {"reward_sum": reward_sum, "end_of_track": bool(end_of_track)}
        if getattr(self.client, "_last_retrieve_invalid", False):
            terminated = True
            info["telemetry_invalid"] = True
        if getattr(self.client, "_last_retrieve_position_patched", False):
            info["position_patched"] = True

        observation = [
            speed,
            acceleration,
            jerk,
            race_progress,
            input_steer,
            input_gas_pedal,
            input_brake,
            gear,
            aim_yaw,
            aim_pitch,
            steer_angle,
            slip_coef,
            failure_counter,
        ]
        if curvature_list is not None:
            curv = np.clip(np.array(curvature_list, dtype="float32") * 10.0, -1.0, 1.0)
            observation.append(curv)
        total_obs = track_info + observation

        self.cur_checkpoint = cur_cp
        self.cur_lap = cur_lap

        min_guaranteed = int(cfg.REWARD_CONFIG.get("MIN_EPISODE_LENGTH_GUARANTEED", 100))
        min_length = max(
            min_guaranteed,
            2 * cfg.REWARD_CONFIG.get("MIN_STEPS", _DEFAULT_MIN_STEPS_END_OF_TRACK),
        )
        if self._steps_since_reset < min_length:
            terminated = False

        reward = np.float32(rew)
        return total_obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation.

        Args:
            seed (int, optional): Seed for the environment. Defaults to None.
            options (dict, optional): Additional options for reset. Defaults to None.

        Returns:
            tuple: (observation, info)
        """
        if (
            getattr(self, "reward_function", None) is not None
            and getattr(self.reward_function, "step_counter", 0) > 0
            and not getattr(self.reward_function, "_logged_run_this_episode", False)
        ):
            self.reward_function.log_model_run(terminated=True, end_of_track=False)
        self.reset_common()
        self._steps_since_reset = 0
        data = self.grab_data()

        self.cur_lap = 0
        self.cur_checkpoint = 0

        speed_kmh = data[2] * 3.6
        speed = np.array([speed_kmh / cfg.OBS_SPEED_SCALE], dtype="float32")
        pos = np.array([data[3], data[4], data[5]], dtype="float32")
        input_steer = np.array([data[6]], dtype="float32")
        input_gas_pedal = np.array([data[7]], dtype="float32")
        input_brake = np.array([data[8]], dtype="float32")
        acceleration = np.array([data[10]], dtype="float32")
        jerk = np.array([data[11]], dtype="float32")
        aim_yaw = np.array([data[12]], dtype="float32")
        aim_pitch = np.array([data[13]], dtype="float32")
        steer_angle = np.array(data[14:16], dtype="float32")
        slip_coef = np.array(data[16:18], dtype="float32")
        gear = np.array([data[19] / 5.0], dtype="float32")

        failure_counter = np.array([0.0], dtype="float32")
        race_progress = np.array([0.0], dtype="float32")

        track_result = self.reward_function.get_track_info(pos, self.points_number)
        left_track = track_result[0]
        center_track = track_result[1]
        right_track = track_result[2]
        curvature_list = track_result[3] if len(track_result) == 4 else None

        if bool(cfg.REWARD_CONFIG.get("TRACK_LOCAL_FRAME", False)):
            yaw = float(data[12])
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)

            def _rotate(pairs):
                rotated = []
                for j in range(0, len(pairs), 2):
                    dx, dz = pairs[j], pairs[j + 1]
                    rotated.append(dx * cos_y - dz * sin_y)
                    rotated.append(dx * sin_y + dz * cos_y)
                return rotated

            left_track = _rotate(left_track)
            center_track = _rotate(center_track)
            right_track = _rotate(right_track)

        track_list = left_track + center_track + right_track
        if cfg.OBS_TRACK_SCALE != 1.0:
            track_list = [x / cfg.OBS_TRACK_SCALE for x in track_list]
        track_info = [np.array(track_list, dtype="float32")]

        observation = [
            speed,
            acceleration,
            jerk,
            race_progress,
            input_steer,
            input_gas_pedal,
            input_brake,
            gear,
            aim_yaw,
            aim_pitch,
            steer_angle,
            slip_coef,
            failure_counter,
        ]
        if curvature_list is not None:
            curv = np.clip(np.array(curvature_list, dtype="float32") * 10.0, -1.0, 1.0)
            observation.append(curv)
        total_obs = track_info + observation

        self.reward_function.reset()
        info = {"reward_sum": 0.0}
        return total_obs, info


class TM2020InterfaceTQCWithImages(TM2020InterfaceTQC):
    def get_observation_space(self):
        base_spaces = super().get_observation_space()
        spaces_list = list(base_spaces.spaces)
        h, w = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        if cfg.GRAYSCALE:
            img_space = spaces.Box(
                low=0.0, high=255.0, shape=(cfg.IMG_HIST_LEN, h, w), dtype=np.float32
            )
        else:
            img_space = spaces.Box(
                low=0.0, high=255.0, shape=(cfg.IMG_HIST_LEN, h, w, 3), dtype=np.float32
            )
        spaces_list.append(img_space)
        return spaces.Tuple(tuple(spaces_list))

    def _capture_and_process_image(self):
        img = self.window_interface.screenshot()[:, :, :3]
        img = cv2.resize(img, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if cfg.GRAYSCALE else img[:, :, ::-1]
        return img.astype(np.float32)

    def get_obs_rew_terminated_info(self):
        total_obs, reward, terminated, info = super().get_obs_rew_terminated_info()
        img = self._capture_and_process_image()
        self._push_img(img)
        imgs = self._get_img_hist_array()
        total_obs.append(imgs)
        return total_obs, reward, terminated, info

    def reset(self, seed=None, options=None):
        total_obs, info = super().reset(seed=seed, options=options)
        img = self._capture_and_process_image()
        for _ in range(self.img_hist_len):
            self._push_img(img)
        imgs = self._get_img_hist_array()
        total_obs.append(imgs)
        return total_obs, info
