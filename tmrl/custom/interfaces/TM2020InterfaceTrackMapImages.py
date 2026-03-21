"""Interface: camera images + pre-recorded track left/right (points in space ahead)."""

import os
import pickle

import cv2
import numpy as np
from gymnasium import spaces
from scipy import spatial

import tmrl.config as cfg
from tmrl.custom.interfaces.TM2020InterfaceLidarProgress import TM2020InterfaceLidarProgress


def _load_track_from_config():
    """Load track left/right from config pkl paths. Returns (map_left, map_right) as (2,N) (x,z)."""

    # pkl from record_track.py: (N, 3) with columns x, y, z. TrackMap uses (x, z) = columns 0 and 2.
    def load(path):
        if not os.path.exists(path):
            return np.array([[0.0, 1.0], [0.0, 1.0]])  # dummy (2, 2)
        with open(path, "rb") as f:
            pts = pickle.load(f)
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = np.expand_dims(pts, 0)
        if pts.shape[1] >= 3:
            return pts[:, [0, 2]].T  # (2, N) x, z
        return pts.T  # (2, N)

    return load(cfg.TRACK_PATH_LEFT), load(cfg.TRACK_PATH_RIGHT)


class TM2020InterfaceTrackMapImages(TM2020InterfaceLidarProgress):
    """
    Images + pre-recorded track left/right (world-space points ahead).
    Observation: (speed, progress, track_information, image_history).
    Uses TmrlData/track/track_<MAP_NAME>_left.pkl and _right.pkl.
    """

    def __init__(
        self,
        img_hist_len=4,
        gamepad=False,
        grayscale=True,
        resize_to=None,
        min_nb_steps_before_failure=int(20 * 3.5),
        save_replays=False,
        **kwargs,
    ):
        super().__init__(
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replays,
            **kwargs,
        )
        self.grayscale = grayscale
        self.resize_to = resize_to or (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
        self.image_hist = []
        self.map_left, self.map_right = _load_track_from_config()
        self.look_ahead_distance = 15
        self.nearby_correction = 60

    def normalize_track(self, l_x, l_z, r_x, r_z, car_position, yaw):
        angle = yaw
        left = (np.array([l_x, l_z]).T - car_position).T
        right = (np.array([r_x, r_z]).T - car_position).T
        left_normal_x = left[0] * np.cos(angle) - left[1] * np.sin(angle)
        left_normal_y = left[0] * np.sin(angle) + left[1] * np.cos(angle)
        right_normal_x = right[0] * np.cos(angle) - right[1] * np.sin(angle)
        right_normal_y = right[0] * np.sin(angle) + right[1] * np.cos(angle)
        return left_normal_x, left_normal_y, right_normal_x, right_normal_y

    def get_track_in_front(self, car_position, look_ahead_distance, nearby_correction):
        entire_map = self.map_left.T.tolist() + self.map_right.T.tolist()
        tree = spatial.KDTree(entire_map)
        (_, i) = tree.query(car_position)
        if i < len(self.map_left.T):
            i_l_min = i
            j_min = max(i_l_min - nearby_correction, 0)
            j_max = min(i_l_min + nearby_correction, len(self.map_left.T) - 1)
            tree_r = spatial.KDTree(self.map_right.T[j_min:j_max])
            (_, i_r_min) = tree_r.query(self.map_left.T[i_l_min])
            i_r_min = i_r_min + j_min
        else:
            i_r = i - len(self.map_left.T)
            j_min = max(i_r - nearby_correction, 0)
            j_max = min(i_r + nearby_correction, len(self.map_right.T) - 1)
            tree_l = spatial.KDTree(self.map_left.T[j_min:j_max])
            (_, i_l_min) = tree_l.query(self.map_right.T[i_r])
            i_l_min = i_l_min + j_min
            i_r_min = i_r
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

    def grab_speed_data_track_and_image(self):
        raw_img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([data[0]], dtype="float32")
        car_position = [data[2], data[4]]
        yaw = data[11]
        l_x, l_z, r_x, r_z = self.get_track_in_front(
            car_position, self.look_ahead_distance, self.nearby_correction
        )
        l_x, l_z, r_x, r_z = self.normalize_track(l_x, l_z, r_x, r_z, car_position, yaw)
        track_information = np.array(
            np.append(np.append(l_x, r_x), np.append(l_z, r_z)), dtype="float32"
        )
        w, h = self.resize_to
        img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) if img.ndim == 2 else np.transpose(img, (2, 0, 1))
        return speed, data, track_information, img

    def reset(self, seed=None, options=None):
        self.reset_common()
        speed, _data, track_information, img = self.grab_speed_data_track_and_image()
        self.image_hist = [img for _ in range(self.img_hist_len)]
        progress = np.array([0], dtype="float32")
        images = np.array(list(self.image_hist), dtype="float32")
        obs = [speed, progress, track_information, images]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        speed, data, track_information, img = self.grab_speed_data_track_and_image()
        rew, terminated, _failure_counter = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]])
        )[:3]
        progress = np.array(
            [self.reward_function.cur_idx / max(1, self.reward_function.datalen)],
            dtype="float32",
        )
        self.image_hist.append(img)
        self.image_hist = self.image_hist[-self.img_hist_len :]
        images = np.array(list(self.image_hist), dtype="float32")
        obs = [speed, progress, track_information, images]
        end_of_track = bool(data[8])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        c = 1 if self.grayscale else 3
        h, w = self.resize_to[1], self.resize_to[0]
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        track_information = spaces.Box(low=-300.0, high=300.0, shape=(60,))
        images = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.img_hist_len, c, h, w),
        )
        return spaces.Tuple((speed, progress, track_information, images))
