# standard library imports
import pickle
import time

# third-party imports
import keyboard
import numpy as np

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.utils.tools import TM2020OpenPlanetClient
import logging


PATH_REWARD = cfg.REWARD_PATH
DATASET_PATH = cfg.DATASET_PATH


def record_reward_dist(path_reward=PATH_REWARD):
    positions = []
    client = TM2020OpenPlanetClient()
    path = path_reward
    time_step = 0.01
    max_error = time_step * 1.0

    is_recording = False
    t1 = time.time()
    while True:
        t2 = time.time()
        if t2 - t1 >= time_step + max_error:
            logging.warning(f" more than time_step + max_error ({time_step + max_error}) passed between two time-steps ({t2 - t1}), updating t1.")
            t1 = time.time()
            # break
        while not t2 - t1 >= time_step:
            t2 = time.time()
        t1 = t1 + time_step

        if keyboard.is_pressed('e'):
            logging.info(f"start recording")
            is_recording = True
        if is_recording:
            data = client.retrieve_data()
            done = bool(data[8])
            if keyboard.is_pressed('q') or done:
                logging.info(f"Smoothing, get fixed dist and saving pickle file...")
                positions = np.array(positions)
                logging.info(f"position init {len(positions)}")

                final_positions = [positions[0]]
                dist_between_points = 0.1
                j = 1
                move_by = dist_between_points
                pt1 = final_positions[-1]
                while j < len(positions):
                    pt2 = positions[j]
                    pt, dst = line(pt1, pt2, move_by)
                    if pt is not None:  # a point was created
                        final_positions.append(pt)
                        move_by = dist_between_points
                        pt1 = pt
                    else:  # we passed pt2 without creating a new point
                        pt1 = pt2
                        j += 1
                        move_by = dst

                final_positions = np.array(final_positions)
                logging.info(f"position fin {len(final_positions)}")

                pickle.dump(final_positions, open(path, "wb"))
                logging.info(f"All done")
                return
            else:
                positions.append([data[2], data[3], data[4]])


def line(pt1, pt2, dist):
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return None, dist - norm  # we couldn't create a new point but we moved by a distance of norm
    else:
        vec_unit = vec / norm
        pt = pt1 + vec_unit * dist
        return pt, 0.0


if __name__ == "__main__":
    record_reward_dist(path_reward=PATH_REWARD)
