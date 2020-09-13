from gym_tmrl.envs.tools import load_digits, get_speed
import numpy as np
import mss
import pickle
import time
import cv2
from inputs import get_gamepad
import gym
import keyboard

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
    path = r"D:/data2020/"
    iteration = 0
    iters, speeds, distances, positions, inputs = [], [], [], [], []
    env = gym.make("gym_tmrl:gym-tmrl-v0")
    env.reset()
    is_recording = False
    while True:
        obs, rew, done, into = env.step(None)
        if keyboard.is_pressed('e'):
            print("start record")
            is_recording = True
        if is_recording:
            cv2.imwrite(path + str(iteration) + ".png", obs[1][-1])
            iteration = iteration + 1
            iters.append(iteration)
            speeds.append(obs[0][0])
            distances.append(obs[0][1])
            positions.append([obs[0][2], obs[0][3], obs[0][4]])
            inputs.append([obs[0][5], obs[0][6], obs[0][7]])

            if keyboard.is_pressed('q'):
                print("Saving pickle file...")
                pickle.dump((iters, speeds, distances, positions, inputs), open(path + "data.pkl", "wb"))
                print("All done")
                return


class TM2020OpenPlanetClient:
    def __init__(self,
                 host='127.0.0.1',
                 port=9000):
        self._host = host
        self._port = port

        # Threading attributes:
        self.__lock = Lock()
        self.__data = None
        self.__t_client = Thread(target=self.__client_thread, args=(), kwargs={}, daemon=True)
        self.__t_client.start()

    def __client_thread(self):
        """
        Thread of the client.
        This listens for incoming data until the object is destroyed
        TODO: handle disconnection
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            while True:  # main loop
                data_raw = b''
                while len(data_raw) != 32:
                    data_raw += s.recv(1024)
                self.__lock.acquire()
                self.__data = data_raw
                self.__lock.release()

    def retrieve_data(self, sleep_if_empty=0.1):
        """
        Retrieves the most recently received data
        Use this function to retrieve the most recently received data
        If block if nothing has been received so far
        """
        c = True
        while c:
            self.__lock.acquire()
            if self.__data is not None:
                data = struct.unpack('<ffffffff', self.__data)
                c = False
            self.__lock.release()
            if c:
                time.sleep(sleep_if_empty)
        return data


def record_reward():
    positions = []
    client = TM2020OpenPlanetClient()
    path = r"D:/data2020reward/"
    time_step = 0.1
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
            print("DEBUG truc de ouf")
            data = client.retrieve_data()
            print(f"pos:{[data[2], data[3], data[4]]}, speed:{data[0]}")
            positions.append([data[2], data[3], data[4]])

            if keyboard.is_pressed('q'):
                print("Saving pickle file...")
                pickle.dump(positions, open(path + "reward.pkl", "wb"))
                print("All done")
                return


if __name__ == "__main__":
    # record_tmnf()
    record_tm20()
    # record_reward()
