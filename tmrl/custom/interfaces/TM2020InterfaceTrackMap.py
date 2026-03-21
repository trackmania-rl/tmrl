"""
This module provides the TM2020InterfaceTrackMap class, an interface for TrackMania 2020
that uses track map information (left and right track boundaries) to construct
the observation space for the agent.
"""

import numpy as np
from gymnasium import spaces
from scipy import spatial

from tmrl.config.paths import TRACKMAP_CSV_LEFT, TRACKMAP_CSV_RIGHT
from tmrl.custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidar
from tmrl.custom.utils.control_mouse import mouse_save_replay_tm20


class TM2020InterfaceTrackMap(TM2020InterfaceLidar):
    """
    Interface for TrackMania 2020 using track map information.

    This interface provides observations that include the vehicle's state (speed, gear, etc.)
    and a localized view of the track boundaries in front of the car.
    """

    def __init__(
        self,
        img_hist_len=1,
        gamepad=False,
        min_nb_steps_before_failure=int(20 * 3.5),
        record=False,
        save_replay: bool = False,
    ):
        """
        Initializes the TM2020InterfaceTrackMap.

        Args:
            img_hist_len (int): Length of the image history. Defaults to 1.
            gamepad (bool): Whether to use a gamepad for input. Defaults to False.
            min_nb_steps_before_failure (int): Minimum steps before failure is considered.
                Defaults to 70.
            record (bool): Whether to record the session. Defaults to False.
            save_replay (bool): Whether to save a replay. Defaults to False.
        """
        super().__init__(img_hist_len=img_hist_len, gamepad=gamepad, save_replays=save_replay)
        self.record = record
        self.window_interface = None
        self.lidar = None
        self.last_pos = [0, 0]
        self.index = 0
        self.map_left = np.loadtxt(TRACKMAP_CSV_LEFT, delimiter=",")
        self.map_right = np.loadtxt(TRACKMAP_CSV_RIGHT, delimiter=",")
        self.all_observed_track_parts: list[list[list[float]]] = [[], [], [], [], []]

    def get_observation_space(self):
        """
        Returns the observation space for the environment.

        The observation space includes speed, gear, RPM, track information (localized boundaries),
        acceleration, steering angle, tire slip, crash status, and a failure counter.

        Returns:
            gymnasium.spaces.Tuple: The observation space.
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        gear = spaces.Box(low=0.0, high=6, shape=(1,))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        track_information = spaces.Box(low=-300, high=300, shape=(60,))
        acceleration = spaces.Box(low=-100, high=100.0, shape=(1,))
        steering_angle = spaces.Box(low=-1, high=1.0, shape=(1,))
        slipping_tires = spaces.Box(low=0.0, high=1, shape=(4,))
        crash = spaces.Box(low=0.0, high=1, shape=(1,))
        failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))
        return spaces.Tuple(
            (
                speed,
                gear,
                rpm,
                track_information,
                acceleration,
                steering_angle,
                slipping_tires,
                crash,
                failure_counter,
            )
        )

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
        car_position = [data[2], data[4]]
        yaw = data[11]  # angle the car is facing
        self.last_pos = car_position

        # retrieving map information
        # Cut out a portion directly in front of the car, as input for the agent
        look_ahead_distance = 15
        nearby_correction = 60
        l_x, l_z, r_x, r_z = self.get_track_in_front(
            car_position, look_ahead_distance, nearby_correction
        )

        # normalize the track in front
        l_x, l_z, r_x, r_z = self.normalize_track(l_x, l_z, r_x, r_z, car_position, yaw)

        # save the track in front for later playback
        self.all_observed_track_parts[0].append(l_x.tolist())
        self.all_observed_track_parts[1].append(l_z.tolist())
        self.all_observed_track_parts[2].append(r_x.tolist())
        self.all_observed_track_parts[3].append(r_z.tolist())
        self.all_observed_track_parts[4].append(car_position)

        track_information = np.array(
            np.append(np.append(l_x, r_x), np.append(l_z, r_z)), dtype="float32"
        )
        speed = np.array([data[0]], dtype="float32")
        gear = np.array([data[9]], dtype="float32")
        rpm = np.array([data[10]], dtype="float32")
        acceleration = np.array([data[18]], dtype="float32")
        steering_angle = np.array([data[19]], dtype="float32")
        slipping_tires = np.array(data[20:24], dtype="float32")
        crash = np.array([data[24]], dtype="float32")

        end_of_track = bool(data[8])
        info = {"end_of_track": end_of_track}
        crash_penalty = -10
        reward = 0
        if crash == 1:
            reward += crash_penalty

        if end_of_track:
            reward += self.finish_reward
            terminated = True
            failure_counter = 0
            if self.save_replays:
                mouse_save_replay_tm20()
        else:
            rew, terminated, failure_counter = self.reward_function.compute_reward(
                pos=np.array([data[2], data[3], data[4]])
            )[:3]
            reward += rew

        failure_counter = float(failure_counter)
        reward = np.float32(reward)
        obs = [
            speed,
            gear,
            rpm,
            track_information,
            acceleration,
            steering_angle,
            slipping_tires,
            crash,
            failure_counter,
        ]
        return obs, reward, terminated, info

    def normalize_track(self, l_x, l_z, r_x, r_z, car_position, yaw):
        """
        Normalizes track coordinates relative to the vehicle's position and orientation.

        Args:
            l_x (np.ndarray): Left track X coordinates.
            l_z (np.ndarray): Left track Z coordinates.
            r_x (np.ndarray): Right track X coordinates.
            r_z (np.ndarray): Right track Z coordinates.
            car_position (list[float]): Current position of the car.
            yaw (float): Current yaw of the car.

        Returns:
            tuple: (normalized_l_x, normalized_l_z, normalized_r_x, normalized_r_z)
        """
        angle = yaw
        left = (np.array([l_x, l_z]).T - car_position).T
        right = (np.array([r_x, r_z]).T - car_position).T

        left_normal_x = left[0] * np.cos(angle) - left[1] * np.sin(angle)
        left_normal_y = left[0] * np.sin(angle) + left[1] * np.cos(angle)

        right_normal_x = right[0] * np.cos(angle) - right[1] * np.sin(angle)
        right_normal_y = right[0] * np.sin(angle) + right[1] * np.cos(angle)

        return left_normal_x, left_normal_y, right_normal_x, right_normal_y

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation.

        Args:
            seed (int, optional): Seed for the environment. Defaults to None.
            options (dict, optional): Additional options for reset. Defaults to None.

        Returns:
            tuple: (observation, info)
        """
        self.reset_common()
        data = self.grab_data()
        track_information = np.full((60,), 0, dtype="float32")
        speed = np.array([data[0]], dtype="float32")
        gear = np.array([data[9]], dtype="float32")
        rpm = np.array([data[10]], dtype="float32")
        acceleration = np.array([data[18]], dtype="float32")
        steering_angle = np.array([data[19]], dtype="float32")
        slipping_tires = np.array(data[20:24], dtype="float32")
        crash = np.array([data[24]], dtype="float32")
        failure_counter = 0.0
        obs = [
            speed,
            gear,
            rpm,
            track_information,
            acceleration,
            steering_angle,
            slipping_tires,
            crash,
            failure_counter,
        ]
        self.reward_function.reset()
        return obs, {}

    def get_track_in_front(self, car_position, look_ahead_distance, nearby_correction):
        """
        Identifies the segments of the track map directly in front of the vehicle.

        Args:
            car_position (list[float]): Current position of the car.
            look_ahead_distance (int): Number of points to look ahead.
            nearby_correction (int): Window size for finding corresponding points on the
                opposite side.

        Returns:
            tuple: (l_x, l_z, r_x, r_z) coordinates of the track in front.
        """
        # Find point that is closest to the car
        entire_map = self.map_left.T.tolist() + self.map_right.T.tolist()
        tree = spatial.KDTree(entire_map)
        (_, i) = tree.query(car_position)
        if i < len(self.map_left.T):  # if the closest point is on the left side
            i_l_min = i
            # find the nearest point on the right side of the track
            j_min = max(i_l_min - nearby_correction, 0)
            j_max = min(i_l_min + nearby_correction, len(self.map_left.T) - 1)
            tree_r = spatial.KDTree(self.map_right.T[j_min:j_max])
            (_, i_r_min) = tree_r.query(self.map_left.T[i_l_min])
            i_r_min = i_r_min + j_min
        else:
            i_r_min = i - len(self.map_left.T)
            # find the nearest point on the left side of the track
            j_min = max(i_r_min - nearby_correction, 0)
            j_max = min(i_r_min + nearby_correction, len(self.map_right.T) - 1)
            tree_l = spatial.KDTree(self.map_left.T[j_min:j_max])
            (_, i_l_min) = tree_l.query(self.map_right.T[i_r_min])
            i_l_min = i_l_min + j_min

        i_l_max = i_l_min + look_ahead_distance
        i_r_max = i_r_min + look_ahead_distance

        extra = np.full((look_ahead_distance, 2), self.map_left.T[-1])
        map_left_extended = np.append(self.map_left.T, extra, axis=0).T

        extra = np.full((look_ahead_distance, 2), self.map_right.T[-1])
        map_right_extended = np.append(self.map_right.T, extra, axis=0).T

        l_x = map_left_extended[0][i_l_min:i_l_max]
        l_z = map_left_extended[1][i_l_min:i_l_max]
        r_x = map_right_extended[0][i_r_min:i_r_max]
        r_z = map_right_extended[1][i_r_min:i_r_max]
        return l_x, l_z, r_x, r_z
