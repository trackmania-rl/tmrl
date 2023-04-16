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
    def __init__(self, host='127.0.0.1', port=9000, struct_str='<' + 'f' * 11):
        self._struct_str = struct_str
        self.nb_floats = self._struct_str.count('f')
        self.nb_uint64 = self._struct_str.count('Q')
        self._nb_bytes = self.nb_floats * 4 + self.nb_uint64 * 8

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
                while len(data_raw) < self._nb_bytes:
                    data_raw += s.recv(1024)
                div = len(data_raw) // self._nb_bytes
                data_used = data_raw[(div - 1) * self._nb_bytes:div * self._nb_bytes]
                data_raw = data_raw[div * self._nb_bytes:]
                self.__lock.acquire()
                self.__data = data_used
                self.__lock.release()

    def retrieve_data(self, sleep_if_empty=0.01, timeout=10.0):
        """
        Retrieves the most recently received data
        Use this function to retrieve the most recently received data
        This blocks if nothing has been received so far
        """
        c = True
        t_start = None
        while c:
            self.__lock.acquire()
            if self.__data is not None:
                data = struct.unpack(self._struct_str, self.__data)
                c = False
                self.__data = None
            self.__lock.release()
            if c:
                if t_start is None:
                    t_start = time.time()
                t_now = time.time()
                assert t_now - t_start < timeout, f"OpenPlanet stopped sending data since more than {timeout}s."
                time.sleep(sleep_if_empty)
        return data


def save_ghost(host='127.0.0.1', port=10000):
    """
    Saves the current ghost

    Args:
        host (str): IP address of the ghost-saving server
        port (int): Port of the ghost-saving server
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))


def armin(tab):
    nz = np.nonzero(tab)[0]
    if len(nz) != 0:
        return nz[0].item()
    else:
        return len(tab) - 1


class Lidar:
    def __init__(self, im):
        self._set_axis_lidar(im)
        self.black_threshold = LIDAR_BLACK_THRESHOLD

    def _set_axis_lidar(self, im):
        h, w, _ = im.shape
        self.h = h
        self.w = w
        self.road_point = (44*h//49, w//2)
        min_dist = 20
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
            dist = min_dist
            while not lenght:
                newx = int(x + dist * dx)
                newy = int(y + dist * dy)
                if newx <= 0 or newy <= 0 or newy >= w - 1:
                    lenght = True
                    list_ax_x.append(np.array(axis_x))
                    list_ax_y.append(np.array(axis_y))
                else:
                    axis_x.append(newx)
                    axis_y.append(newy)
                dist = dist + 1
        self.list_axis_x = list_ax_x
        self.list_axis_y = list_ax_y

    def lidar_20(self, img, show=False):
        h, w, _ = img.shape
        if h != self.h or w != self.w:
            self._set_axis_lidar(img)
        distances = []
        if show:
            color = (255, 0, 0)
            thickness = 4
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        for axis_x, axis_y in zip(self.list_axis_x, self.list_axis_y):
            index = armin(np.all(img[axis_x, axis_y] < self.black_threshold, axis=1))
            if show:
                img = cv2.line(img, (self.road_point[1], self.road_point[0]), (axis_y[index], axis_x[index]), color, thickness)
            index = np.float32(index)
            distances.append(index)
        res = np.array(distances, dtype=np.float32)
        if show:
            cv2.imshow("Environment", img)
            cv2.waitKey(1)
        return res


if __name__ == "__main__":
    pass
