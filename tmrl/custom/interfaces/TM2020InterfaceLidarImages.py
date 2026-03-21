"""Interface that provides both camera images and LIDAR (speed + progress + LIDAR + images)."""

import cv2
import numpy as np
from gymnasium import spaces

import tmrl.config as cfg
from tmrl.custom.interfaces.TM2020InterfaceLidarProgress import TM2020InterfaceLidarProgress


class TM2020InterfaceLidarProgressImages(TM2020InterfaceLidarProgress):
    """
    LIDAR + progress + camera images from the same screenshot.
    Observation: (speed, progress, lidar_history, image_history).
    Use with a fusion model (e.g. frozen EffNet for images + residual MLP for vector).
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

    def grab_lidar_speed_data_and_image(self):
        """Screenshot once; compute LIDAR and return resized image for the model."""
        raw_img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([data[0]], dtype="float32")
        lidar = self.lidar.lidar_20(img=raw_img, show=False)

        # Resize and optionally grayscale for the image branch
        w, h = self.resize_to
        img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) if img.ndim == 2 else np.transpose(img, (2, 0, 1))
        return lidar, speed, data, img

    def reset(self, seed=None, options=None):
        self.reset_common()
        lidar, speed, _data, img = self.grab_lidar_speed_data_and_image()

        self.img_hist = [lidar for _ in range(self.img_hist_len)]
        self.image_hist = [img for _ in range(self.img_hist_len)]

        progress = np.array([0], dtype="float32")
        lidars = np.array(list(self.img_hist), dtype="float32")
        images = np.array(list(self.image_hist), dtype="float32")
        obs = [speed, progress, lidars, images]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        lidar, speed, data, img = self.grab_lidar_speed_data_and_image()
        rew, terminated, _failure_counter = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]])
        )[:3]
        progress = np.array(
            [self.reward_function.cur_idx / max(1, self.reward_function.datalen)],
            dtype="float32",
        )

        self.img_hist.append(lidar)
        self.img_hist = self.img_hist[-self.img_hist_len :]
        self.image_hist.append(img)
        self.image_hist = self.image_hist[-self.img_hist_len :]

        lidars = np.array(list(self.img_hist), dtype="float32")
        images = np.array(list(self.image_hist), dtype="float32")
        obs = [speed, progress, lidars, images]
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
        lidars = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.img_hist_len, 19),
        )
        images = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.img_hist_len, c, h, w),
        )
        return spaces.Tuple((speed, progress, lidars, images))
