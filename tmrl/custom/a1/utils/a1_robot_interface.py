"""
This file defines the A1Robot Python object that enables controlling the real A1 robot.
A1Robot wraps the C++ Unitree Legged SDK.
The SDK was compiled as an .so file and translated to Python via pybind under the name robot_interface.
TMRL ships with this .so file compiled for Python 3.10.

Code adapted from https://github.com/ikostrikov/walk_in_the_park
"""

import logging

import numpy as np
import time
import copy
import math
import enum

from filterpy.kalman import KalmanFilter
from tmrl.custom.a1.utils.moving_window_filter import MovingWindowFilter
from tmrl.custon.a1.utils.action_filter import ActionFilterButter

try:
    from tmrl.custom.a1.utils.robot_interface import RobotInterface
except Exception as e:
    logging.warning(f"robot_interface.so works only on the A1 EDU robot with python 3.10.")
    raise e


TWO_PI = 2 * math.pi
MOTOR_WARN_TEMP_C = 50.0  # At 60C, Unitree will shut down a motor until it cools off.
NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2

ACTION_CONFIG = [
        ("FR_hip_motor",0.802851455917,-0.802851455917),
        ("FR_upper_joint",4.18879020479,-1.0471975512),
        ("FR_lower_joint",-0.916297857297,-2.69653369433),
        ("FL_hip_motor",0.802851455917,-0.802851455917),
        ("FL_upper_joint",4.18879020479,-1.0471975512),
        ("FL_lower_joint",-0.916297857297,-2.69653369433),
        ("RR_hip_motor",0.802851455917,-0.802851455917),
        ("RR_upper_joint",4.18879020479,-1.0471975512),
        ("RR_lower_joint",-0.916297857297,-2.69653369433),
        ("RL_hip_motor",0.802851455917,-0.802851455917),
        ("RL_upper_joint",4.18879020479,-1.0471975512),
        ("RL_lower_joint",-0.916297857297,-2.69653369433),
    ]

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0
MOTOR_KP = [
    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
    HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
MOTOR_KD = [
    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
    HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]

class MotorControlMode(enum.Enum):
    """The supported motor control modes."""
    POSITION = 1

    # Apply motor torques directly.
    TORQUE = 2

    # Apply a tuple (q, qdot, kp, kd, tau) for each motor. Here q, qdot are motor
    # position and velocities. kp and kd are PD gains. tau is the additional
    # motor torque. This is the most flexible control mode.
    HYBRID = 3


def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion into a 3x3 rotation matrix.

    Parameters:
        q (numpy.ndarray): A unit quaternion [q_w, q_x, q_y, q_z].

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """

    q_x, q_y, q_z, q_w = q

    qxx, qyy, qzz = q_x * q_x, q_y * q_y, q_z * q_z
    qxy, qxz, qyz = q_x * q_y, q_x * q_z, q_y * q_z
    qwx, qwy, qwz = q_w * q_x, q_w * q_y, q_w * q_z

    R = np.array([
        [1 - 2 * (qyy + qzz), 2 * (qxy - qwz),     2 * (qxz + qwy)],
        [2 * (qxy + qwz),     1 - 2 * (qxx + qzz), 2 * (qyz - qwx)],
        [2 * (qxz - qwy),     2 * (qyz + qwx),     1 - 2 * (qxx + qyy)]
    ])

    return R


# @numba.jit(nopython=True, cache=True)
def analytical_leg_jacobian(leg_angles, leg_id):
    """
    Computes the analytical Jacobian.
    Args:
        leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
        l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1)**(leg_id + 1)

    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * np.cos(t_eff)
    J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(
        t_eff) / l_eff - l_eff * np.cos(t_eff) / 2
    J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J


def map_to_minus_pi_to_pi(angles):
    mapped_angles = copy.deepcopy(angles)
    for i in range(len(angles)):
        mapped_angles[i] = math.fmod(angles[i], TWO_PI)
        if mapped_angles[i] >= math.pi:
            mapped_angles[i] -= TWO_PI
        elif mapped_angles[i] < -math.pi:
            mapped_angles[i] += TWO_PI
    return mapped_angles


class A1Robot:
    def __init__(self,
                 motor_control_mode=MotorControlMode.POSITION,
                 enable_clip_motor_commands=True,
                 motor_torque_limit = 35.5,
                 time_step=0.001,
                 accelerometer_variance=0.1,
                 sensor_variance=0.1,
                 initial_variance=0.1,
                 moving_window_filter_size=1,
                 action_filter_highcut=3.0):
        
        # A1 SDK interface
        self._interface = RobotInterface()
        self._initialize_interface()

        # Step counter
        self.time_step = time_step
        self._step_counter = 0

        # General
        self._motor_control_mode = motor_control_mode
        self._enable_clip_motor_commands = enable_clip_motor_commands
        self._motor_torque_limit = motor_torque_limit
        self._state = None
        self.orientation = None
        self.acceleration = None
        self.motor_angles = None
        self.motor_velocities = None
        self.motor_torques = None
        self.motor_temperatures = None
        self.base_position = np.zeros((3, ))
        self._last_position_update_time = time.time()
        self._motor_kps = np.asarray(MOTOR_KP)
        self._motor_kds = np.asarray(MOTOR_KD)

        # Velocity estimation
        self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        self.filter.x = np.zeros(3)
        self._initial_variance = initial_variance
        self.filter.P = np.eye(3) * self._initial_variance  # State covariance
        self.filter.Q = np.eye(3) * accelerometer_variance
        self.filter.R = np.eye(3) * sensor_variance
        self.filter.H = np.eye(3)  # measurement function (y=H*x)
        self.filter.F = np.eye(3)  # state transition matrix
        self.filter.B = np.eye(3)
        self._window_size = moving_window_filter_size
        self.moving_window_filter_x = MovingWindowFilter(window_size=self._window_size)
        self.moving_window_filter_y = MovingWindowFilter(window_size=self._window_size)
        self.moving_window_filter_z = MovingWindowFilter(window_size=self._window_size)
        self.estimated_velocity = np.zeros(3)
        self._last_timestamp = 0

        # Motor angle limits
        self._joint_angle_upper_limits = np.array([field[1] for field in ACTION_CONFIG])
        self._joint_angle_lower_limits = np.array([field[2] for field in ACTION_CONFIG])

        # Safety flag
        self._is_safe = True

        self.update()

        # action filter
        self._action_repeat = 1  # TODO remove
        self._action_filter_highcut = action_filter_highcut
        self._action_filter = self._build_action_filter([self._action_filter_highcut])
    
    def _initialize_interface(self):
        self._interface.send_command(np.zeros(60, dtype=np.float32))
        time.sleep(0.5)
    
    def _build_action_filter(self, highcut=None):
        sampling_rate = 1 / (self.time_step * self._action_repeat)
        a_filter = ActionFilterButter(
            sampling_rate=sampling_rate,
            num_joints=NUM_MOTORS,
            highcut=highcut)
        return a_filter
    
    def reset_velocity_estimator(self):
        logging.info("resetting velocity estimator")
        self.filter.x = np.zeros(3)
        self.filter.P = np.eye(3) * self._initial_variance
        self.moving_window_filter_x = MovingWindowFilter(window_size=self._window_size)
        self.moving_window_filter_y = MovingWindowFilter(window_size=self._window_size)
        self.moving_window_filter_z = MovingWindowFilter(window_size=self._window_size)
        self._last_timestamp = 0
    
    def _compute_delta_time(self, current_time):
        if self._last_timestamp == 0.:
            # First timestamp received, return an estimated delta_time.
            delta_time_s = self.time_step
        else:
            delta_time_s = current_time - self._last_timestamp
        self._last_timestamp = current_time
        return delta_time_s
    
    def get_motor_angles(self):
        return map_to_minus_pi_to_pi(self.motor_angles).copy()
    
    def compute_jacobian(self, leg_id):
        motor_angles = self.get_motor_angles()[leg_id * 3:(leg_id + 1) * 3]
        return analytical_leg_jacobian(motor_angles, leg_id)
    
    def update_velocity_estimate(self, current_time):
        """Propagate current state estimate with new accelerometer reading."""
        delta_time_s = self._compute_delta_time(current_time)
        sensor_acc = self.acceleration
        self._raw_acc = sensor_acc
        base_orientation = self.orientation
        rot_mat = quaternion_to_rotation_matrix(base_orientation)
        rot_mat = rot_mat.reshape((3, 3))

        calibrated_acc = sensor_acc + np.linalg.inv(rot_mat).dot(np.array([0., 0., -9.8]))
        self._calibrated_acc = calibrated_acc
        self.filter.predict(u=calibrated_acc * delta_time_s)

        # Correct estimation using contact legs
        observed_velocities = []
        self._observed_velocities = []
        foot_contact = self.get_foot_contacts()
        for leg_id in range(4):
            if foot_contact[leg_id]:
                jacobian = self.compute_jacobian(leg_id)
                # Only pick the jacobian related to joint motors
                joint_velocities = self.robot.GetMotorVelocities()[leg_id * 3:(leg_id + 1) * 3]
                leg_velocity_in_base_frame = jacobian.dot(joint_velocities)
                base_velocity_in_base_frame = -leg_velocity_in_base_frame[:3]
                observed_velocities.append(base_velocity_in_base_frame)
                self._observed_velocities.append(base_velocity_in_base_frame)
            else:
                self._observed_velocities.append(np.zeros(3))

        if len(observed_velocities) > 0:
            observed_velocities = np.mean(observed_velocities, axis=0)
            self.filter.update(observed_velocities)

        vel_x = self.moving_window_filter_x.calculate_average(self.filter.x[0])
        vel_y = self.moving_window_filter_y.calculate_average(self.filter.x[1])
        vel_z = self.moving_window_filter_z.calculate_average(self.filter.x[2])
        self.estimated_velocity = np.array([vel_x, vel_y, vel_z])
    
    def get_foot_contacts(self):
        return np.array(self._state.footForce) > 20
    
    def update_position(self):
        now = time.time()
        self.base_position += self.estimated_velocity * (now - self._last_position_update_time)
        self._last_position_update_time = now

    def update(self):
        self._state = self._interface.receive_observation()
        quat = self._state.imu.quaternion
        self.orientation = np.array([quat[1], quat[2], quat[3], quat[0]])
        self.acceleration = np.array(self._state.imu.accelerometer)
        self.motor_angles = np.array([motor.q for motor in self._state.motorState[:12]])
        self.motor_velocities = np.array([motor.dq for motor in self._state.motorState[:12]])
        self.motor_torques = np.array([motor.tauEst for motor in self._state.motorState[:12]])
        self.motor_temperatures = np.array([motor.temperature for motor in self._state.motorState[:12]])
        self.update_velocity_estimate(self._state.tick / 1000.)
        self.update_position()
    
    def _clip_motor_angles(self, desired_angles, current_angles):
        if self._enable_clip_motor_commands:
            angle_ub = np.minimum(
                self._joint_angle_upper_limits,
                current_angles + MAX_MOTOR_ANGLE_CHANGE_PER_STEP)
            angle_lb = np.maximum(
                self._joint_angle_lower_limits,
                current_angles - MAX_MOTOR_ANGLE_CHANGE_PER_STEP)
        else:
            angle_ub = self._joint_angle_upper_limits
            angle_lb = self._joint_angle_lower_limits
        return np.clip(desired_angles, angle_lb, angle_ub)
    
    def _clip_motor_commands(self, motor_commands, motor_control_mode):
        """Clips commands to respect any set joint angle and torque limits.

        Always clips position to be within ACTION_CONFIG. If
        self._enable_clip_motor_commands, also clips positions to be within
        MAX_MOTOR_ANGLE_CHANGE_PER_STEP of current positions.
        Always clips torques to be within self._motor_torque_limit (but the torque
        limits can be infinity).

        Args:
            motor_commands: np.array. Can be motor angles, torques, or hybrid.
            motor_control_mode: A MotorControlMode enum.

        Returns:
            Clipped motor commands.
        """
        if motor_control_mode == MotorControlMode.TORQUE:
            return np.clip(motor_commands, -1 * self._motor_torque_limit, self._motor_torque_limit)
        if motor_control_mode == MotorControlMode.POSITION:
            return self._clip_motor_angles(desired_angles=motor_commands, current_angles=self.motor_angles)
        if motor_control_mode == MotorControlMode.HYBRID:
            # Clip angles
            angles = motor_commands[np.array(range(NUM_MOTORS)) * 5]
            clipped_positions = self._clip_motor_angles(desired_angles=angles, current_angles=self.motor_angles)
            motor_commands[np.array(range(NUM_MOTORS)) * 5] = clipped_positions
            # Clip torques
            torques = motor_commands[np.array(range(NUM_MOTORS)) * 5 + 4]
            clipped_torques = np.clip(torques, -1 * self._motor_torque_limit, self._motor_torque_limit)
            motor_commands[np.array(range(NUM_MOTORS)) * 5 + 4] = clipped_torques
            return motor_commands
    
    def apply_action(self, motor_commands, motor_control_mode=None):
        """
        Clip and apply the motor commands using the motor model.
        Args:
            motor_commands: np.array.
            motor_control_mode: A member of the MotorControlMode enum.
        """
        if motor_control_mode is None:
            motor_control_mode = self._motor_control_mode

        motor_commands = self._clip_motor_commands(motor_commands, motor_control_mode)

        command = np.zeros(60, dtype=np.float32)
        if motor_control_mode == MotorControlMode.POSITION:
            for motor_id in range(NUM_MOTORS):
                command[motor_id * 5] = motor_commands[motor_id]
                command[motor_id * 5 + 1] = self._motor_kps[motor_id]
                command[motor_id * 5 + 3] = self._motor_kds[motor_id]
        elif motor_control_mode == MotorControlMode.TORQUE:
            for motor_id in range(NUM_MOTORS):
                command[motor_id * 5 + 4] = motor_commands[motor_id]
        elif motor_control_mode == MotorControlMode.HYBRID:
            command = np.array(motor_commands, dtype=np.float32)
        else:
            raise ValueError(f"Unknown motor control mode for A1 robot: {motor_control_mode}")

        self._interface.send_command(command)
    
    def check_motor_temperatures(self):
        if any(self.motor_temperatures > MOTOR_WARN_TEMP_C):
            logging.warning(f"Motors are getting hot. Temperatures: {[(name, temp) for name, temp in zip(MOTOR_NAMES, self._motor_temperatures.astype(int))]}")
    
    def _ResetActionFilter(self):
        self._action_filter.reset()

    def _filter_action(self, action):
        # initialize the filter history, since resetting the filter will fill
        # the history with zeros and this can cause sudden movements at the start
        # of each episode
        # import ipdb; ipdb.set_trace()
        if self._step_counter == 0:
            default_action = self.get_motor_angles()
            self._action_filter.init_history(default_action)

        filtered_action = self._action_filter.filter(action)
        return filtered_action
    
    def _step_minitaur(self, action, control_mode=None):
       action = self._filter_action(action)
        if control_mode == None:
            control_mode = self._motor_control_mode
        for i in range(self._action_repeat):
            proc_action = self.ProcessAction(action, i)
            self.step_internal(proc_action, control_mode)
            self._step_counter += 1

        self._last_action = action

    def _step_a1_robot(self, action, control_mode=None):
        self._step_minitaur(action, control_mode)
        self.check_motor_temperatures()

    # TODO: adapt the following
    
    def _ValidateMotorStates(self):
        # Check torque.
        if any(np.abs(self.motor_torques) > self._motor_torque_limits):
            raise robot_config.SafetyError(
                "Torque limits exceeded\ntorques: {}".format(
                    self.motor_torques))

        # Check joint velocities.
        if any(np.abs(self.GetTrueMotorVelocities()) > MAX_JOINT_VELOCITY):
            raise robot_config.SafetyError(
                "Velocity limits exceeded\nvelocities: {}".format(
                    self.GetTrueMotorVelocities()))

        # Joints often start out of bounds (in sim they're 0 and on real they're
        # slightly out of bounds), so we don't check angles during reset.
        if self._currently_resetting or self.running_reset_policy:
            return
        # Check joint positions.
        if (any(self.GetTrueMotorAngles() > (self._joint_angle_upper_limits +
                                             self.JOINT_EPSILON))
                or any(self.GetTrueMotorAngles() <
                       (self._joint_angle_lower_limits - self.JOINT_EPSILON))):
            raise robot_config.SafetyError(
                "Joint angle limits exceeded\nangles: {}".format(
                    self.GetTrueMotorAngles()))

    def step_internal(self, action, motor_control_mode=None):
        if self._is_safe:
            self.apply_action(action, motor_control_mode)
        self.update()
        self._state_action_counter += 1
        if not self._is_safe:
            return
        try:
            self._ValidateMotorStates()
        except (robot_config.SafetyError) as e:
            print(e)
            if self.running_reset_policy:
                # Let the resetter handle retries.
                raise e
            self._is_safe = False
            return
        self._Nap()

    def _Nap(self):
        """Sleep for the remainder of self.time_step."""
        now = time.time()
        sleep_time = self.time_step - (now - self._last_step_time_wall)
        if self._timesteps is not None:
            self._timesteps.append(now - self._last_step_time_wall)
        self._last_step_time_wall = now
        if sleep_time >= 0:
            time.sleep(sleep_time)

    def Brake(self):
        self.ReleasePose()
        self._robot_interface.brake()
        self.LogTimesteps()
        self._Nap()


if __name__ == "__main__":
    print("hello")

    a1 = A1Robot()

    ori = a1.orientation
    print(f"quaternion ori: {ori}")

    rot_mat = quaternion_to_rotation_matrix(ori)

    print(f"rot_mat: {rot_mat}")

    print(f"pos: {a1.base_position}")

