# standard library imports
import math
import os
import socket
import struct
import time
from pathlib import Path
from threading import Lock, Thread

# third-party imports
import cv2
import numpy as np

# local imports
from tmrl.config.config_constants import LIDAR_BLACK_THRESHOLD


class TM2020OpenPlanetClient:
    def __init__(self, host='127.0.0.1', port=9000):
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
            data_raw = b''
            while True:  # main loop
                while len(data_raw) < 44:
                    data_raw += s.recv(1024)
                div = len(data_raw) // 44
                data_used = data_raw[(div - 1) * 44:div * 44]
                data_raw = data_raw[div * 44:]
                self.__lock.acquire()
                self.__data = data_used
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
                data = struct.unpack('<fffffffffff', self.__data)
                c = False
            self.__lock.release()
            if c:
                time.sleep(sleep_if_empty)
        return data


def load_digits():
    p = Path(os.path.dirname(os.path.realpath(__file__))) / 'digits'
    zero = cv2.imread(str(p / '0.png'), 0)
    One = cv2.imread(str(p / '1.png'), 0)
    Two = cv2.imread(str(p / '2.png'), 0)
    Three = cv2.imread(str(p / '3.png'), 0)
    four = cv2.imread(str(p / '4.png'), 0)
    five = cv2.imread(str(p / '5.png'), 0)
    six = cv2.imread(str(p / '6.png'), 0)
    seven = cv2.imread(str(p / '7.png'), 0)
    eight = cv2.imread(str(p / '8.png'), 0)
    nine = cv2.imread(str(p / '9.png'), 0)
    digits = np.array([zero, One, Two, Three, four, five, six, seven, eight, nine])
    return digits


def get_speed(img, digits):
    img1 = np.array(img[464:, 887:908])
    img2 = np.array(img[464:, 909:930])
    img3 = np.array(img[464:, 930:951])

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1 > 250] = 255
    img1[img1 <= 250] = 0
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2[img2 > 250] = 255
    img2[img2 <= 250] = 0
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img3[img3 > 250] = 255
    img3[img3 <= 250] = 0

    best1, best2, best3 = 100000, 100000, 100000
    for idx, num in enumerate(digits):
        if np.sum(np.bitwise_xor(img1, num)) < best1:
            best1 = np.sum(np.bitwise_xor(img1, num))
            num1 = idx
        if np.sum(np.bitwise_xor(img2, num)) < best2:
            best2 = np.sum(np.bitwise_xor(img2, num))
            num2 = idx
        if np.sum(np.bitwise_xor(img3, num)) < best3:
            best3 = np.sum(np.bitwise_xor(img3, num))
            num3 = idx
        if np.max(img1) == 0:
            best1, num1 = 0, 0
        if np.max(img2) == 0:
            best2, num2 = 0, 0
        if np.max(img3) == 0:
            best3, num3 = 0, 0
    return float(100 * num1 + 10 * num2 + num3)


def armin(tab):
    nz = np.nonzero(tab)[0]
    if len(nz) != 0:
        return nz[0].item()
    else:
        return len(tab) - 1


class Lidar:
    def __init__(self, monitor, road_point):
        self.road_point = road_point
        self.monitor = monitor
        self.list_axis_x, self.list_axis_y = self._get_axis_lidar()
        self.black_threshold = LIDAR_BLACK_THRESHOLD

    def _get_axis_lidar(self):
        list_ax_x = []
        list_ax_y = []
        for angle in range(90, 280, 10):
            axis_x = []
            axis_y = []
            x = self.road_point[0]
            y = self.road_point[1]
            dx = math.cos(math.radians(angle))
            dy = math.sin(math.radians(angle))
            lenght = False
            dist = 20
            while not lenght:
                newx = int(x + dist * dx)
                newy = int(y + dist * dy)
                if newx <= 0 or newy <= 0 or newy >= self.monitor["width"] - 1:
                    lenght = True
                    list_ax_x.append(np.array(axis_x))
                    list_ax_y.append(np.array(axis_y))
                else:
                    axis_x.append(newx)
                    axis_y.append(newy)
                dist = dist + 1
        return list_ax_x, list_ax_y

    def lidar_20(self, im, show=False):
        img = np.array(im)
        distances = []
        if show:
            color = (255, 0, 0)
            thickness = 4
        for axis_x, axis_y in zip(self.list_axis_x, self.list_axis_y):
            index = armin(np.all(img[axis_x, axis_y] < self.black_threshold, axis=1))
            if show:
                img = cv2.line(img, (self.road_point[1], self.road_point[0]), (axis_y[index], axis_x[index]), color, thickness)
            index = np.float32(index)
            distances.append(index)
        res = np.array(distances, dtype=np.float32)
        if show:
            cv2.imshow("PipeLine", img)
            cv2.waitKey(1)
        return res


if __name__ == "__main__":
    pass
