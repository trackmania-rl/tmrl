import numpy as np
from gymnasium import spaces

from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.tm.utils.tools import Lidar


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(
        self,
        img_hist_len=1,
        gamepad=False,
        min_nb_steps_before_failure=int(20 * 3.5),
        save_replays: bool = False,
        **kwargs,
    ):
        super().__init__(
            img_hist_len=img_hist_len, gamepad=gamepad, save_replays=save_replays, **kwargs
        )
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.window_interface = None
        self.lidar = None

    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array(
            [
                data[0],
            ],
            dtype="float32",
        )
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, _data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype="float32")
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated, _failure_counter = self.reward_function.compute_reward(
            pos=np.array([data[2], data[3], data[4]])
        )[:3]
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
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        imgs = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(
                self.img_hist_len,
                19,
            ),
        )  # lidars
        return spaces.Tuple((speed, imgs))
