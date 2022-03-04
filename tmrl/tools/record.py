# standard library imports
import os
import pickle
import time

# third-party imports
import cv2
import gym
import keyboard
import mss
import numpy as np
from inputs import get_gamepad
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.custom_gym_interfaces import (TM2020Interface, TM2020InterfaceLidar,
                                               TMInterface, TMInterfaceLidar)
from tmrl.custom.utils.tools import TM2020OpenPlanetClient, get_speed, load_digits
import logging
# TODO: add info dicts everywhere for CRC debugging

KEY_START_RECORD = 'e'
KEY_STOP_RECORD = 't'
KEY_FORWARD = 'up'
KEY_BACKWARD = 'down'
KEY_RIGHT = 'right'
KEY_LEFT = 'left'
KEY_RESET = 'r'

PATH_REWARD = cfg.REWARD_PATH
DATASET_PATH = cfg.DATASET_PATH


def record_tmnf_gamepad(path_dataset):
    """
    TODO: update
    """
    path = path_dataset
    direction = [0, 0, 0, 0]  # dir :  [acc [0,1], brake [0,1], left [0,1], right [0,1]]
    digits = load_digits()
    monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
    sct = mss.mss()
    time_step = 0.05
    max_error = time_step * 1.0  # if the error in timestep becomes larger than this, stop recording
    iteration = 0
    speeds = []
    dirs = []
    iters = []

    c = True
    while c:
        events = get_gamepad()
        if events:
            for event in events:
                if str(event.code) == "ABS_HAT0X":
                    c = False
                    logging.info('start recording')

    t1 = time.time()
    while not c:
        t2 = time.time()
        if t2 - t1 >= time_step + max_error:
            logging.warning(f" more than time_step + max_error ({time_step + max_error}) passed between two time-steps ({t2 - t1}). Stopping recording.")
            c = True
            break
        while not t2 - t1 >= time_step:
            t2 = time.time()
            # time.sleep(0.001)
            pass
        t1 = t1 + time_step

        img = np.asarray(sct.grab(monitor))[:, :, :3]
        speed = get_speed(img, digits)
        img = img[100:-150, :]
        img = cv2.resize(img, (190, 50))
        ev = get_gamepad()
        all_events = []
        while ev is not None:
            all_events = all_events + ev
            ev = get_gamepad()
        if len(all_events) > 0:
            for event in all_events:
                if str(event.code) == "BTN_SOUTH":
                    direction[0] = event.state
                elif str(event.code) == "BTN_TR" or str(event.code) == "BTN_WEST":
                    direction[1] = event.state
                elif str(event.code) == "ABS_X":
                    gd = event.state / 32768
                    if gd > 0:
                        direction[3] = gd
                        direction[2] = 0.0
                    else:
                        direction[3] = 0.0
                        direction[2] = -gd
                elif str(event.code) == "ABS_HAT0Y":
                    c = True
                    logging.info('stop recording')

        cv2.imwrite(path + str(iteration) + ".png", img)
        speeds.append(speed)
        direction = [float(i) for i in direction]
        dirs.append(direction)
        iters.append(iteration)
        iteration = iteration + 1
        # time.sleep(1)

    pickle.dump((iters, dirs, speeds), open(path + "data.pkl", "wb"))


def record_tmnf_keyboard(path_dataset):
    path = path_dataset
    direction = [0, 0, 0, 0]  # dir :  [acc [0,1], brake [0,1], left [0,1], right [0,1]]
    iteration = 0
    iters, speeds, distances, positions, dirs, dones, rews = [], [], [], [], [], [], []

    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TMInterface
    env = gym.make("real-time-gym-v0")
    env.reset()

    is_recording = False
    while True:
        obs, rew, done, info = env.step(None)
        direction[0] = float(keyboard.is_pressed(KEY_FORWARD))
        direction[1] = float(keyboard.is_pressed(KEY_BACKWARD))
        direction[2] = float(keyboard.is_pressed(KEY_RIGHT))
        direction[3] = float(keyboard.is_pressed(KEY_LEFT))
        if keyboard.is_pressed(KEY_RESET):
            logging.info(f"reset")
            env.reset()
            done = True
        if keyboard.is_pressed(KEY_START_RECORD):
            logging.info(f"start record")
            is_recording = True

        if is_recording:
            cv2.imwrite(path + str(iteration) + ".png", obs[1][-1])

            iters.append(iteration)
            speeds.append(obs[0])
            direc = np.array([direction[0], direction[1], direction[2] - direction[3]], dtype=np.float32)  # +1 for right and -1 for left
            dirs.append(direc)
            dones.append(done)
            rews.append(rew)

            iteration = iteration + 1

            if keyboard.is_pressed(KEY_STOP_RECORD):
                logging.info(f"Saving pickle file...")
                pickle.dump((iters, dirs, speeds, dones, rews), open(path + "data.pkl", "wb"))
                logging.info(f"All done")
                return


def record_tmnf_lidar_keyboard(path_dataset):
    path = path_dataset
    direction = [0, 0, 0, 0]  # dir :  [acc [0,1], brake [0,1], left [0,1], right [0,1]]
    iteration = 0
    iters, speeds, lidars, dirs, dones, rews = [], [], [], [], [], []

    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TMInterfaceLidar

    env = gym.make("real-time-gym-v0")
    env.reset()

    is_recording = False
    done = False
    while True:
        obs, rew, done, info = env.step(None) if not done else (
            env.reset(),
            0.0,
            False,
            {},
        )
        direction[0] = float(keyboard.is_pressed(KEY_FORWARD))
        direction[1] = float(keyboard.is_pressed(KEY_BACKWARD))
        direction[2] = float(keyboard.is_pressed(KEY_RIGHT))
        direction[3] = float(keyboard.is_pressed(KEY_LEFT))
        if keyboard.is_pressed(KEY_RESET):
            done = True
        if keyboard.is_pressed(KEY_START_RECORD):
            logging.info(f"start record")
            is_recording = True

        if is_recording:
            # logging.debug(f" ---")
            # logging.debug(f"lidar:{obs[1][-1]}")
            lidars.append(obs[1][-1])
            iters.append(iteration)
            # logging.debug(f"speed:{obs[0]}")
            speeds.append(obs[0])
            direc = np.array([direction[0], direction[1], direction[2] - direction[3]], dtype=np.float32)  # +1 for right and -1 for left
            # logging.debug(f"direction:{direc}")
            dirs.append(direc)
            # logging.debug(f"done:{done}")
            dones.append(done)
            # logging.debug(f"rew:{rew}")
            # logging.debug(f" ---")
            rews.append(rew)

            iteration = iteration + 1

            if keyboard.is_pressed(KEY_STOP_RECORD):
                logging.info(f"Saving pickle file...")
                pickle.dump((iters, dirs, speeds, lidars, dones, rews), open(path + "data.pkl", "wb"))
                logging.info(f"All done")
                return


def record_tm20_lidar(path_dataset):
    path = path_dataset
    iteration = 0
    iters, speeds, distances, positions, inputs, dones, rews = [], [], [], [], [], [], []

    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceLidar
    env_config["ep_max_length"] = 2000
    env = gym.make("real-time-gym-v0")
    env.reset()

    is_recording = False
    done = False
    while True:
        obs, rew, done, info = env.step(None) if not done else (
            env.reset(),
            0.0,
            False,
            {},
        )
        # obs = (obs[0], obs[1], obs[0][-3:])
        if keyboard.is_pressed(KEY_RESET):
            logging.info(f"reset")
            env.reset()
            done = True
        if keyboard.is_pressed(KEY_START_RECORD):
            logging.info(f"start record")
            is_recording = True

        if is_recording:
            cv2.imwrite(path + str(iteration) + ".png", obs[1][-1])
            iters.append(iteration)
            speeds.append(obs[0][0])
            distances.append(obs[0][1])
            positions.append([obs[0][2], obs[0][3], obs[0][4]])
            inputs.append([obs[0][6], obs[0][7], obs[0][5]])  # FIXME: check this thoroughly
            dones.append(done)
            rews.append(rew)
            iteration = iteration + 1

            if keyboard.is_pressed(KEY_STOP_RECORD):
                logging.info(f"Saving pickle file...")
                pickle.dump((iters, speeds, distances, positions, inputs, dones, rews), open(path + "data.pkl", "wb"))
                logging.info(f"All done")
                return


def record_tm20(path_dataset):
    """
    set the game to 40fps

    :return:
    """
    path = path_dataset
    iteration = 0
    iters, speeds, gear, rpm, dones, rews = [], [], [], [], [], []

    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020Interface
    env = gym.make("real-time-gym-v0")
    env.reset()

    is_recording = False
    video = cv2.VideoWriter(os.path.join(path, 'video.avi'), cv2.VideoWriter_fourcc(*'FFV1'), 20, (256, 127))

    while True:
        obs, rew, done, info = env.step(None)
        # obs = (obs[0], obs[1], obs[0][-3:])
        if keyboard.is_pressed(KEY_RESET):
            logging.info(f"reset")
            env.reset()
            done = True
        if keyboard.is_pressed(KEY_START_RECORD):
            logging.info(f"start record")
            is_recording = True

        if is_recording:
            #cv2.imwrite(os.path.join(path, str(iteration) + ".png"), np.moveaxis(obs[3][-1], 0, -1))
            video.write(np.moveaxis(obs[3][-1], 0, -1))
            iters.append(iteration)
            speeds.append(obs[0][0])
            gear.append(obs[1][0])
            rpm.append(obs[2][0])
            dones.append(done)
            rews.append(rew)
            iteration = iteration + 1

            if keyboard.is_pressed(KEY_STOP_RECORD):
                logging.info(f"Saving pickle file...")
                pickle.dump((iters, speeds, gear, rpm, dones, rews), open(path + "data.pkl", "wb"))
                # create video
                cv2.destroyAllWindows()
                video.release()
                logging.info(f"All done")
                return


def record_reward(path_reward=PATH_REWARD):
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
            positions.append([data[2], data[3], data[4]])
            if keyboard.is_pressed('q'):
                logging.info(f"Smoothing and saving pickle file...")
                positions = np.array(positions)
                epsilon = 0.001
                for i in range(len(positions)):
                    acc = np.sum(positions[i]) - np.sum(positions[0])
                    if acc > epsilon:
                        positions = positions[i:]
                        break
                position_1 = np.array(positions)
                position_2 = np.array(positions)
                for i in range(1, len(positions) - 1):
                    position_1[i] = (positions[i - 1] + positions[i] + positions[i + 1]) / 3.0
                for i in range(1, len(position_1) - 1):
                    position_2[i] = (position_1[i - 1] + position_1[i] + position_1[i + 1]) / 3.0
                pickle.dump(position_2, open(path + "reward.pkl", "wb"))
                logging.info(f"All done")
                return


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
    #record_tm20_lidar(PATH_DATASET)
    # record_tmnf_lidar_keyboard(PATH_DATASET)
    # record_tmnf_keyboard(PATH_DATASET)
    #record_tm20(DATASET_PATH)
    # record_reward(PATH_REWARD)
    record_reward_dist(path_reward=PATH_REWARD)
