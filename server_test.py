import socket
import struct
import time
from threading import Thread, Lock


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


if __name__ == "__main__":
    client = TM2020OpenPlanetClient()
    while True:
        data = client.retrieve_data()
        print(f"data:{data}")
        time.sleep(0.5)
