import time
from threading import Thread, Lock
import random


def clip(val, min_val, max_val):
    return min(max(val, min_val), max_val)


class DummyRCDrone:
    def __init__(self,
                 mass=1.0,
                 friction=1.0,
                 communication_delay_min=0.02,
                 communication_delay_max=0.05,
                 max_vel=1.0,
                 world_size=1.0,
                 control_sleep=0.01):
        self.max_vel = max_vel
        self.mass = mass
        self.friction = friction
        self.communication_delay_min = communication_delay_min
        self.communication_delay_max = communication_delay_max
        self.world_size = world_size
        self.__lock_obs = Lock()
        self.__t_obs = time.time()
        self.__obs_pos_x = 0.0
        self.__obs_pos_y = 0.0
        self.__lock_act = Lock()
        self.__t_vel = time.time()
        self.__vel_x = 0.0
        self.__vel_y = 0.0
        self.control_sleep = control_sleep
        self.__run_th = Thread(target=self.__run, args=(), kwargs={}, daemon=True).start()

    def send_control(self, vel_x, vel_y):
        vel_x = clip(vel_x, - self.max_vel, self.max_vel)
        vel_y = clip(vel_y, - self.max_vel, self.max_vel)
        Thread(target=self.__send_act, args=(vel_x, vel_y), kwargs={}, daemon=True).start()

    def get_observation(self):
        self.__lock_obs.acquire()
        o_pos_x = self.__obs_pos_x
        o_pos_y = self.__obs_pos_y
        self.__lock_obs.release()
        return o_pos_x, o_pos_y

    def _send_obs(self, pos_x, pos_y):
        Thread(target=self.__send_obs, args=(pos_x, pos_y), kwargs={}, daemon=True).start()

    def __send_act(self, vel_x, vel_y):
        t = time.time()
        time.sleep(random.uniform(self.communication_delay_min, self.communication_delay_max))
        self.__lock_act.acquire()
        if t > self.__t_vel:
            self.__t_vel = t
            self.__vel_x = vel_x
            self.__vel_y = vel_y
        self.__lock_act.release()

    def __send_obs(self, pos_x, pos_y):
        t = time.time()
        time.sleep(random.uniform(self.communication_delay_min, self.communication_delay_max))
        self.__lock_obs.acquire()
        if t > self.__t_obs:
            self.__t_obs = t
            self.__obs_pos_x = pos_x
            self.__obs_pos_y = pos_y
        self.__lock_obs.release()

    def __run(self):
        pos_x = 0.0
        pos_y = 0.0
        while True:
            self.__lock_act.acquire()
            vel_x = self.__vel_x
            vel_y = self.__vel_y
            self.__lock_act.release()
            self._send_obs(pos_x, pos_y)
            pos_x = pos_x + vel_x * self.control_sleep
            pos_y = pos_y + vel_y * self.control_sleep
            clip(pos_x, - self.world_size, self.world_size)
            clip(pos_y, - self.world_size, self.world_size)
            time.sleep(self.control_sleep)
