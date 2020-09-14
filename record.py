from gym_tmrl.envs.tools import load_digits, get_speed
import numpy as np
import mss
import pickle
import time
import cv2
from inputs import get_gamepad
import gym
import keyboard
from gym_tmrl.envs.tmrl_env import TM2020OpenPlanetClient

import socket
import struct
from threading import Thread, Lock


def record_tmnf():
    path = r"C:/Users/Yann/Desktop/git/tmrl/data/"
    #path = r"D:/data/"
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
                    print('start recording')

    t1 = time.time()
    while not c:
        t2 = time.time()
        if t2 - t1 >= time_step + max_error:
            print(f"WARNING: more than time_step + max_error ({time_step + max_error}) passed between two time-steps ({t2 - t1}). Stopping recording.")
            c = True
            break
        while not t2 - t1 >= time_step:
            t2 = time.time()
            # time.sleep(0.001)
            pass
        t1 = t1+time_step

        img = np.asarray(sct.grab(monitor))[:, :, :3]
        speed = np.array([get_speed(img, digits), ], dtype='float32')
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
                    print('stop recording')

        cv2.imwrite(path + str(iteration) + ".png", img)
        speeds.append(speed)
        direction = [float(i) for i in direction]
        dirs.append(direction)
        iters.append(iteration)
        iteration = iteration + 1
        # time.sleep(1)

    pickle.dump((iters,dirs,speeds), open( path +"data.pkl", "wb" ))


def record_tm20():
    """
    set the game to 40fps

    :return:
    """
    path = r"D:/data2020/"
    iteration = 0
    iters, speeds, distances, positions, inputs, dones, rews = [], [], [], [], [], [], []
    env = gym.make("gym_tmrl:gym-tmrl-v0")
    env.reset()
    is_recording = False
    while True:
        obs, rew, done, info = env.step(None)
        # obs = (obs[0], obs[1], obs[0][-3:])
        if keyboard.is_pressed('r'):
            env.reset()
            done = True
        if keyboard.is_pressed('e'):
            print("start record")
            is_recording = True
        if is_recording:
            cv2.imwrite(path + str(iteration) + ".png", obs[1][-1])
            iters.append(iteration)
            speeds.append(obs[0][0])
            distances.append(obs[0][1])
            positions.append([obs[0][2], obs[0][3], obs[0][4]])
            inputs.append([obs[0][5], obs[0][6], obs[0][7]])
            dones.append(done)
            rews.append(rew)
            iteration = iteration + 1

            if keyboard.is_pressed('q'):
                print("Saving pickle file...")
                pickle.dump((iters, speeds, distances, positions, inputs, dones, rews), open(path + "data.pkl", "wb"))
                print("All done")
                return



def record_reward():
    positions = []
    client = TM2020OpenPlanetClient()
    path = r"D:/data2020reward/"
    time_step = 0.01
    max_error = time_step * 1.0

    is_recording = False
    t1 = time.time()
    while True:
        t2 = time.time()
        if t2 - t1 >= time_step + max_error:
            print(f"WARNING: more than time_step + max_error ({time_step + max_error}) passed between two time-steps ({t2 - t1}), updating t1.")
            t1 = time.time()
            # break
        while not t2 - t1 >= time_step:
            t2 = time.time()
        t1 = t1 + time_step

        if keyboard.is_pressed('e'):
            print("start recording")
            is_recording = True
        if is_recording:
            data = client.retrieve_data()
            positions.append([data[2], data[3], data[4]])
            if keyboard.is_pressed('q'):
                print("Smoothing and saving pickle file...")
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
                print("All done")
                return


if __name__ == "__main__":
    # record_tmnf()
    record_tm20()
    # record_reward()
