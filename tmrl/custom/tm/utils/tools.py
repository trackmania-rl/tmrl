# standard library imports
import math
import socket
import struct
import time
from threading import Lock, Thread

# third-party imports
import cv2
import numpy as np

# local imports
from tmrl.config import LIDAR_BLACK_THRESHOLD


class TM2020OpenPlanetClient:
    # Script attributes:
    def __init__(self, host="127.0.0.1", port=9000, struct_str=None, nb_floats=19):
        if struct_str is None:
            struct_str = "<" + "f" * nb_floats
        self._struct_str = struct_str
        self.nb_floats = self._struct_str.count("f")
        self.nb_int32 = self._struct_str.count("i")
        self.nb_uint64 = self._struct_str.count("Q")
        self._nb_bytes = self.nb_floats * 4 + self.nb_uint64 * 8 + self.nb_int32 * 4

        self._host = host
        self._port = port

        # Threading attributes:
        self.__lock = Lock()
        self.__data = None
        self._received_once = False  # used for longer first-packet timeout
        self._last_good_pos = None  # Buffer to replace [0, 0, 0] glitches from the plugin
        self._last_retrieve_position_patched = (
            False  # True when position was patched (for interface)
        )
        self._last_retrieve_invalid = (
            False  # True when sanity check failed (interface should terminate)
        )
        self._client_connected = (
            False  # True when connect() succeeded (may still be waiting for first frame)
        )
        self.__t_client = Thread(target=self.__client_thread, args=(), kwargs={}, daemon=True)
        self.__t_client.start()

    def __client_thread(self):
        """
        Thread of the client.
        Connects to the OpenPlanet plugin (retries until the plugin is listening),
        then receives data until the connection is closed.
        """
        retry_interval = 2.0
        retry_count = 0
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self._host, self._port))
                    self._client_connected = True
                    print(
                        f"Connected to OpenPlanet plugin at {self._host}:{self._port}. "
                        "Waiting for game data (be in a map with car on track, not main menu).",
                        flush=True,
                    )
                    data_raw = b""
                    while True:  # main loop
                        while len(data_raw) < self._nb_bytes:
                            chunk = s.recv(1024)
                            if not chunk:
                                self._client_connected = False
                                break  # connection closed, reconnect
                            data_raw += chunk
                        if len(data_raw) < self._nb_bytes:
                            self._client_connected = False
                            break  # connection closed
                        div = len(data_raw) // self._nb_bytes
                        data_used = data_raw[(div - 1) * self._nb_bytes : div * self._nb_bytes]
                        data_raw = data_raw[div * self._nb_bytes :]
                        self.__lock.acquire()
                        self.__data = data_used
                        self._received_once = True
                        self.__lock.release()
            except (ConnectionRefusedError, OSError):
                self._client_connected = False
                retry_count += 1
                if retry_count == 1 or retry_count % 5 == 0:
                    print(
                        f"Cannot connect to {self._host}:{self._port} (attempt {retry_count}). "
                        "TrackMania running? TQC_GrabData loaded (F3→Developer→Reload)? In map?",
                        flush=True,
                    )
                time.sleep(retry_interval)
                continue

    def retrieve_data(self, sleep_if_empty=0.01, timeout=10.0, first_packet_timeout=60.0):
        """
        Retrieves the most recently received data.
        Blocks if nothing has been received so far.
        Uses first_packet_timeout until the first packet is received (allows time for
        connection/reconnect); then uses timeout for subsequent waits.

        Timeouts and the post-return sanity check (speed range) are the first line of
        defense against corrupted samples entering the replay buffer. The interface
        should check _last_retrieve_invalid and _last_retrieve_position_patched and
        terminate the episode or flag the transition when appropriate.
        """
        c = True
        t_start = None
        data = None
        last_hint_log = 0.0
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
                elapsed = t_now - t_start
                effective_timeout = first_packet_timeout if not self._received_once else timeout
                if not self._received_once and elapsed - last_hint_log >= 15.0:
                    last_hint_log = elapsed
                    if self._client_connected:
                        print(
                            "Connected but no game data yet. Be IN A MAP with car on track "
                            "(drive or stand), not main menu or loading.",
                            flush=True,
                        )
                    else:
                        print(
                            f"Waiting for OpenPlanet plugin ({self._host}:{self._port}). "
                            "Start TrackMania, load map, TQC_GrabData (F3→Developer→Reload), "
                            "then be in map (car on track).",
                            flush=True,
                        )
                assert elapsed < effective_timeout, (
                    f"OpenPlanet stopped sending data since more than {effective_timeout}s. "
                    "Check: (1) TrackMania running, (2) TQC_GrabData (F3→Developer→Reload), "
                    "(3) IN A MAP with car on track (not menu/loading)."
                )
                time.sleep(sleep_if_empty)

        # FIX GLITCHES: replace [0,0,0] position with the last known good position
        self._last_retrieve_position_patched = False
        self._last_retrieve_invalid = False
        if data is not None:
            # Determine position indices from struct size:
            # TQC=20 floats uses [3,4,5], older formats use [2,3,4].
            pos_start_idx = 3 if self.nb_floats >= 20 else 2
            pos_x, pos_y, pos_z = (
                data[pos_start_idx],
                data[pos_start_idx + 1],
                data[pos_start_idx + 2],
            )

            # If position is exactly at or very near origin, consider it a glitch
            if math.sqrt(pos_x**2 + pos_y**2 + pos_z**2) < 1.0:
                if self._last_good_pos is not None:
                    # Create a new tuple with the patched position
                    data_list = list(data)
                    data_list[pos_start_idx] = self._last_good_pos[0]
                    data_list[pos_start_idx + 1] = self._last_good_pos[1]
                    data_list[pos_start_idx + 2] = self._last_good_pos[2]
                    data = tuple(data_list)
                    self._last_retrieve_position_patched = True
            else:
                # Update last known good position
                self._last_good_pos = (pos_x, pos_y, pos_z)

            # Sanity check: speed (index 2 in TQC format) should be in a plausible range.
            # Corrupted frames (e.g. all zeros or garbage) can otherwise enter the replay buffer.
            speed_idx = 2
            if speed_idx < len(data):
                try:
                    speed_val = float(data[speed_idx])
                    if not (0 <= speed_val <= 2500.0):
                        self._last_retrieve_invalid = True
                except (TypeError, ValueError):
                    self._last_retrieve_invalid = True
        return data


def save_ghost(host="127.0.0.1", port=10000):
    """
    Saves the current ghost

    Args:
        host (str): IP address of the ghost-saving server
        port (int): Port of the ghost-saving server
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))


def armin(tab):
    """
    Functionality: Finds the index of the first non-zero element in the input array tab.
    Returns: The index of the first non-zero element if found;
             otherwise, the index of the last element in the array.
    """
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
        """
        Functionality:
        Sets up the LiDAR axis based on the image passed.
        Creates LiDAR axes for scanning, angles from 90 to 280 degrees.
        """
        h, w, _ = im.shape
        self.h = h
        self.w = w
        self.road_point = (44 * h // 49, w // 2)
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
        """
        Functionality:
        Calculates LiDAR distances given an image.
        If the image dimensions differ from the previously set dimensions, updates the LiDAR axis.
        Loops through the predefined LiDAR axes and calculates the distances.
        Optionally displays LiDAR lines on the image if show is set to True.
        Returns: An array of distances calculated by the LiDAR.
        """
        h, w, _ = img.shape
        if h != self.h or w != self.w:
            self._set_axis_lidar(img)
        distances = []
        if show:
            color = (255, 0, 0)
            thickness = 4
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        for axis_x, axis_y in zip(self.list_axis_x, self.list_axis_y, strict=False):
            index = armin(np.all(img[axis_x, axis_y] < self.black_threshold, axis=1))
            if show:
                img = cv2.line(
                    img,
                    (self.road_point[1], self.road_point[0]),
                    (axis_y[index], axis_x[index]),
                    color,
                    thickness,
                )
            index = np.float32(index)
            distances.append(index)
        res = np.array(distances, dtype=np.float32)
        if show:
            cv2.imshow("Environment", img)
            cv2.waitKey(1)
        return res


if __name__ == "__main__":
    pass
