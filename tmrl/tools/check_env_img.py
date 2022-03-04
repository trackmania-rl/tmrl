# third-party imports
import cv2
import mss
import numpy as np
import logging

def check_env():
    sct = mss.mss()
    monitor = {"top": 32, "left": 1, "width": 256, "height": 127}
    # lidar = Lidar(monitor=monitor,
    #               road_point=(440, 479))
    # env_config = DEFAULT_CONFIG_DICT.copy()
    # env_config["interface"] = TM2020InterfaceLidar
    # env_config["wait_on_done"] = True
    # env = gym.make("real-time-gym-v0", config=env_config)
    # o = env.reset()
    while True:
        # o, r, d, i = env.step(None)
        # logging.info(r)
        img = np.asarray(sct.grab(monitor))[:, :, :3]
        cv2.imshow("PipeLine", img)
        cv2.waitKey(1)
        img = np.moveaxis(img, -1, 0)
        #logging.info(img.shape)


if __name__ == "__main__":
    check_env()
