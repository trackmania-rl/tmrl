# standard library imports
import pickle
import time

# third-party imports
import numpy as np
from loguru import logger

import tmrl.config as cfg
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

PATH_REWARD = cfg.REWARD_PATH
DATASET_PATH = cfg.DATASET_PATH


def record_reward_dist(path_reward=PATH_REWARD, use_keyboard=False):
    if use_keyboard:
        import keyboard

    positions = []
    client = TM2020OpenPlanetClient()
    path = path_reward

    is_recording = False
    while True:
        if not is_recording:
            if not use_keyboard:
                logger.info("start recording")
                is_recording = True
            else:
                if keyboard.is_pressed("e"):
                    logger.info("start recording")
                    is_recording = True

        if is_recording:
            data = client.retrieve_data(
                sleep_if_empty=0.01
            )  # we need many points to build a smooth curve
            terminated = bool(data[8])

            early_stop = False if not use_keyboard else keyboard.is_pressed("q")

            if early_stop or terminated:
                logger.info("Computing reward function checkpoints from captured positions...")
                logger.info(f"Initial number of captured positions: {len(positions)}")
                positions = np.array(positions)

                final_positions = [positions[0]]
                dist_between_points = 0.1
                j = 1
                move_by = dist_between_points
                pt1 = final_positions[-1]
                while j < len(positions):
                    pt2 = positions[j]
                    pt, dst = line(pt1, pt2, move_by)
                    if pt is not None:  # a point was created
                        final_positions.append(pt)  # add the point to the list
                        move_by = dist_between_points
                        pt1 = pt
                    else:  # we passed pt2 without creating a new point
                        pt1 = pt2
                        j += 1
                        move_by = dst  # remaining distance

                final_positions = np.array(final_positions)
                logger.info(
                    f"Final number of checkpoints in the reward function: {len(final_positions)}"
                )

                with open(path, "wb") as f:
                    pickle.dump(final_positions, f)
                logger.info("All done")
                return
            else:
                positions.append([data[2], data[3], data[4]])
        else:
            time.sleep(0.05)  # waiting for user to press E


def line(pt1, pt2, dist):
    """
    Creates a point between pt1 and pt2, at distance dist from pt1.

    If dist is too large, returns None and the remaining distance (> 0.0).
    Else, returns the point and 0.0 as remaining distance.
    """
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return (
            None,
            dist - norm,
        )  # we couldn't create a new point but we moved by a distance of norm
    else:
        vec_unit = vec / norm
        pt = pt1 + vec_unit * dist
        return pt, 0.0


if __name__ == "__main__":
    record_reward_dist(path_reward=PATH_REWARD)
