
import numpy as np
import mss

import cv2


def check_env():
    sct = mss.mss()
    monitor = {"top": 32, "left": 1, "width": 256, "height": 127}
    # lidar = Lidar(monitor=monitor,
    #               road_point=(440, 479))
    # env_config = DEFAULT_CONFIG_DICT
    # env_config["interface"] = TM2020InterfaceLidar
    # env_config["wait_on_done"] = True
    # env = gym.make("rtgym:real-time-gym-v0", config=env_config)
    # o = env.reset()
    while True:
        # o, r, d, i = env.step(None)
        # print(r)
        img = np.asarray(sct.grab(monitor))[:, :, :3]
        cv2.imshow("PipeLine", img)
        cv2.waitKey(1)
        img = np.moveaxis(img, -1, 0)
        #print(img.shape)


if __name__ == "__main__":
    check_env()
