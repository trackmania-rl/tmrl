import numpy as np
import time
from threading import Lock
from agents.custom.utils.udp_interface import UDPInterface
from threading import Thread
from copy import deepcopy


class DroneUDPInterface1:
    """
    This is the API needed for the algorithm to control the drone, kappa_id
    It also has a listener thread that retrieves observations from the drone
    Since the environment is single-agent, recovery functions block execution
    Communications are performed by UDP
    The controller sends [ZVelocity, arm(1.0)/disarm(0.0), brain_timestamp]
    The drone observations are [alt, vel, acc, ubatt, time_step_id]
    Cognifly controller is cognifly_vel_controller
    """
    def __init__(self, udp_send_ip, udp_recv_ip, udp_send_port, udp_recv_port,
                 min_altitude=0.0,
                 max_altitude=100.0,
                 low_batt=7.5):
        """
        Args:
            udp_send_ip: string
            udp_recv_ip: string
            udp_send_port: int: udp_send_port for UDPInterface
            udp_recv_port: int: udp_recv_port for UDPInterface
            min_altitude: float
            max_altitude: float
            low_batt: float: under this voltage, the drone is disarmed
        """
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.low_batt = low_batt
        self.udp_send_port = udp_send_port
        self.udp_recv_port = udp_recv_port
        self.udp_int = UDPInterface()
        # print(f"DEBUG: initializing sender with udp_send_ip:{udp_send_ip}, udp_send_port:{udp_send_port}")
        self.udp_int.init_sender(udp_send_ip, udp_send_port)
        # print(f"DEBUG: initializing receiver with udp_recv_ip:{udp_recv_ip}, udp_recv_port:{udp_recv_port}")
        self.udp_int.init_receiver(udp_recv_ip, udp_recv_port)
        self._obs = None
        self._lock = Lock()
        self._listener_thread = Thread(target=self.__listener_thread)
        self._listener_thread.setDaemon(True)  # will be terminated at exit
        self._listener_thread.start()
        # print("DEBUG: initializing obs...")
        while True:
            time.sleep(0.1)  # only for initialization
            self._lock.acquire()
            self.obs = deepcopy(self._obs)
            self._lock.release()
            if self.obs is not None:
                # print("DEBUG: found obs")
                break
        # print("DEBUG: obs initialized")

    def __listener_thread(self):
        # print(f"DEBUG: listener thread started")
        while True:
            # print("DEBUG: thread waiting recv")
            mes = self.udp_int.recv_msg()
            # print("DEBUG: recv")
            self._lock.acquire()
            for m in mes:
                if self._obs is None or self._obs[4] <= m[4]:
                    self._obs = m
            self._lock.release()

    def send_control(self, control, time_step_id):
        """
        Non-blocking function
        Applies the action given by the RL policy
        Args:
            control: float: throttle
        """
        # print(f"DEBUG: apply_train_control:{control}")
        ctrl = [control, 1.0, time_step_id]
        # print(f"DEBUG: ctrl:{ctrl}")
        self.udp_int.send_msg(ctrl)

    def take_off(self, takeoff_vel=10.0, target_alt=40.0, sleep_time=0.1):
        """
        Blocking function
        The drone takes off
        May use the optitrack
        """
        while True:
            ctrl = np.array([takeoff_vel, 1.0, -1.0])
            self.udp_int.send_msg(ctrl)
            time.sleep(sleep_time)
            self.update()
            if self.read_altitude() >= target_alt:
                break
        self.wait()

    def arm_disarm(self, arm=True, wait_time=2.0):
        """
        Blocking function
        Arms if arm==True, else disarms
        Args:
            arm: bool: whether to arm or disarm
            wait_time: float: time slept after sending the command
        """
        print(f"DEBUG: arm:{arm}")
        if arm:
            ctrl = np.array([0.0, 1.0, -1.0])
        else:
            ctrl = np.array([0.0, 0.0, -1.0])
        self.udp_int.send_msg(ctrl)
        time.sleep(wait_time)

    def land(self, land_vel, wait_time=2.0):
        """
        Blocking function
        The drone lands
        May use the optitrack
        Args:
            wait_time: float: time slept after sending the land command
        """
        ctrl = np.array([land_vel, 1.0, -1.0])
        self.udp_int.send_msg(ctrl)
        time.sleep(wait_time)
        self.wait()

    def wait(self):
        """
        Non-blocking function
        The drone stays 'paused', waiting in position
        May use the optitrack
        """
        ctrl = np.array([0.0, 1.0, -1.0])
        self.udp_int.send_msg(ctrl)

    def wait_for_new_battery(self):
        """
        Blocking function
        Blocks until a new battery is up and the drone is ready to take off
        """
        raise NotImplementedError

    def update(self):
        self._lock.acquire()
        self.obs = deepcopy(self._obs)
        self._lock.release()

    def read_battery(self):
        self._lock.acquire()
        res = self.obs[3]
        self._lock.release()
        return res

    def read_altitude(self):
        self._lock.acquire()
        res = self.obs[0]
        self._lock.release()
        return res

    def read_obs(self):
        self._lock.acquire()
        res = deepcopy(self.obs)
        self._lock.release()
        return res

    def apply_safety_constraints(self):
        """
        Checks whether the safety constraints are violated and applies the required action
        Returns:
            bool: whether the safety constraints have been violated
        """
        # check for battery level:
        if self.read_battery() <= self.low_batt:
            print(f"DEBUG: low battery: landing and disarming...")
            self.land()
            self.arm_disarm(arm=False)
            self.wait_for_new_battery()
        # check for arena boundaries:
        else:
            cur_drone_alt = self.read_altitude()
            if cur_drone_alt < self.min_altitude or cur_drone_alt > self.max_altitude:
                print(f"DEBUG: outside of arena: disarming")
                self.arm_disarm(arm=False)
                assert False, f"A security constraint has been violated: cur_drone_alt:{cur_drone_alt}"
        return False
