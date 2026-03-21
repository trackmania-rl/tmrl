"""Field index enums for memory data structures.

These enums replace magic numbers with named constants, making the code
self-documenting and easier to debug.
"""

from enum import IntEnum


class BufferField(IntEnum):
    """Indices for sample buffer tuples: (act, obs, rew, terminated, truncated, info)."""

    ACTION = 0
    OBSERVATION = 1
    REWARD = 2
    TERMINATED = 3
    TRUNCATED = 4
    INFO = 5


class GenericField(IntEnum):
    """Field indices for GenericTorchMemory.data list.

    Layout: [actions, observations, rewards, terminated, truncated, info, done]
    """

    ACTIONS = 0
    OBSERVATIONS = 1
    REWARDS = 2
    TERMINATED = 3
    TRUNCATED = 4
    INFO = 5
    DONE = 6


class TMLidarField(IntEnum):
    """Field indices for MemoryTMLidar.data list.

    Layout: [indexes, actions, speeds, lidar, eoes, rewards, infos, terminated, truncated]
    """

    INDEXES = 0
    ACTIONS = 1
    SPEEDS = 2
    LIDAR = 3
    EOES = 4
    REWARDS = 5
    INFOS = 6
    TERMINATED = 7
    TRUNCATED = 8


class TMLidarProgressField(IntEnum):
    """Field indices for MemoryTMLidarProgress.data list.

    Layout: [indexes, actions, speeds, lidar, eoes, rewards, infos, progress, terminated, truncated]
    """

    INDEXES = 0
    ACTIONS = 1
    SPEEDS = 2
    LIDAR = 3
    EOES = 4
    REWARDS = 5
    INFOS = 6
    PROGRESS = 7
    TERMINATED = 8
    TRUNCATED = 9


class TMLidarProgressImagesField(IntEnum):
    """Field indices for MemoryTMLidarProgressImages.data list.

    Layout: [indexes, actions, speeds, progress, lidar, images, eoes, rewards, infos,
             terminated, truncated]
    """

    INDEXES = 0
    ACTIONS = 1
    SPEEDS = 2
    PROGRESS = 3
    LIDAR = 4
    IMAGES = 5
    EOES = 6
    REWARDS = 7
    INFOS = 8
    TERMINATED = 9
    TRUNCATED = 10


class TMFullField(IntEnum):
    """Field indices for MemoryTMFull.data list.

    Layout: [indexes, actions, speeds, images, eoes, rewards, infos, gears, rpms,
             terminated, truncated]
    """

    INDEXES = 0
    ACTIONS = 1
    SPEEDS = 2
    IMAGES = 3
    EOES = 4
    REWARDS = 5
    INFOS = 6
    GEARS = 7
    RPMS = 8
    TERMINATED = 9
    TRUNCATED = 10


class TMBestField(IntEnum):
    """Field indices for MemoryTMBest.data list.

    Layout includes all telemetry fields from the game.
    """

    INDEXES = 0
    ACTIONS = 1
    POSITION = 2
    SPEED = 3
    ACCELERATION = 4
    JERK = 5
    RACE_PROGRESS = 6
    INPUT_STEER = 7
    INPUT_GAS_PEDAL = 8
    INPUT_BRAKE = 9
    GEAR = 10
    AIM_YAW = 11
    AIM_PITCH = 12
    SURFACE_ID = 13
    STEER_ANGLE = 14
    WHEEL_ROT = 15
    WHEEL_ROT_SPEED = 16
    DAMPER_LEN = 17
    SLIP_COEF = 18
    REACTOR_GROUND_MODE = 19
    GROUND_CONTACT = 20
    REACTOR_AIR_CONTROL = 21
    GROUND_DIST = 22
    CRASHED = 23
    FAILURE_COUNTER = 24
    IMGS = 25
    EOES = 26
    REWARDS = 27
    INFOS = 28
    TERMINATED = 29
    TRUNCATED = 30


class R2D2Field(IntEnum):
    """Field indices for MemoryR2D2.data list.

    Layout: [indexes, actions, checkpoints, speeds, accelerations, jerks,
             race_progress, input_steer, input_gas_pedal, input_brake, gear,
             aim_yaw, aim_pitch, steer_angle, slip_coef, failure_counter,
             imgs, eoes, rewards, infos, terminated, truncated]
    """

    INDEXES = 0
    ACTIONS = 1
    CHECKPOINTS = 2
    SPEEDS = 3
    ACCELERATIONS = 4
    JERKS = 5
    RACE_PROGRESS = 6
    INPUT_STEER = 7
    INPUT_GAS_PEDAL = 8
    INPUT_BRAKE = 9
    GEAR = 10
    AIM_YAW = 11
    AIM_PITCH = 12
    STEER_ANGLE = 13
    SLIP_COEF = 14
    FAILURE_COUNTER = 15
    IMGS = 16
    EOES = 17
    REWARDS = 18
    INFOS = 19
    TERMINATED = 20
    TRUNCATED = 21


class R2D2SophyField(IntEnum):
    """Field indices for MemoryR2D2Sophy.data list.

    Same telemetry as R2D2Field but without images and track_info replaces checkpoints.
    """

    INDEXES = 0
    ACTIONS = 1
    TRACK_INFO = 2
    SPEEDS = 3
    ACCELERATIONS = 4
    JERKS = 5
    RACE_PROGRESS = 6
    INPUT_STEER = 7
    INPUT_GAS_PEDAL = 8
    INPUT_BRAKE = 9
    GEAR = 10
    AIM_YAW = 11
    AIM_PITCH = 12
    STEER_ANGLE = 13
    SLIP_COEF = 14
    FAILURE_COUNTER = 15
    EOES = 16
    REWARDS = 17
    INFOS = 18
    TERMINATED = 19
    TRUNCATED = 20


class R2D2woImagesTrailingField(IntEnum):
    """Trailing field offsets for MemoryR2D2woImages.

    These are offsets from the obs_end boundary.
    Layout: [...obs fields..., eoes, rewards, infos, terminated, truncated]
    """

    EOES = 0
    REWARDS = 1
    INFOS = 2
    TERMINATED = 3
    TRUNCATED = 4


# ---------------------------------------------------------------------------
# Observation-tuple field enums
#
# These map indices within the *observation tuple* coming from the environment
# (i.e. b[BufferField.OBSERVATION][<index>]) to named constants.
# ---------------------------------------------------------------------------


class TMLidarObsField(IntEnum):
    """Observation indices for MemoryTMLidar: (speed, lidar)."""

    SPEEDS = 0
    LIDAR = 1


class TMLidarProgressObsField(IntEnum):
    """Observation indices for MemoryTMLidarProgress: (speed, progress, lidar)."""

    SPEEDS = 0
    PROGRESS = 1
    LIDAR = 2


class TMLidarProgressImagesObsField(IntEnum):
    """Observation indices for MemoryTMLidarProgressImages: (speed, progress, lidar, images)."""

    SPEEDS = 0
    PROGRESS = 1
    LIDAR = 2
    IMAGES = 3


class TMFullObsField(IntEnum):
    """Observation indices for MemoryTMFull: (speed, gears, rpms, images)."""

    SPEEDS = 0
    GEARS = 1
    RPMS = 2
    IMAGES = 3


class TMBestObsField(IntEnum):
    """Observation indices for MemoryTMBest (full game telemetry).

    Note: SURFACE_ID (11) is present in the raw observation but intentionally
    skipped when building data columns in MemoryTMBest.append_buffer.
    """

    POSITION = 0
    SPEED = 1
    ACCELERATION = 2
    JERK = 3
    RACE_PROGRESS = 4
    INPUT_STEER = 5
    INPUT_GAS_PEDAL = 6
    INPUT_BRAKE = 7
    GEAR = 8
    AIM_YAW = 9
    AIM_PITCH = 10
    SURFACE_ID = 11
    STEER_ANGLE = 12
    WHEEL_ROT = 13
    WHEEL_ROT_SPEED = 14
    DAMPER_LEN = 15
    SLIP_COEF = 16
    REACTOR_GROUND_MODE = 17
    GROUND_CONTACT = 18
    REACTOR_AIR_CONTROL = 19
    GROUND_DIST = 20
    CRASHED = 21
    CRASHED_LIST = 22
    FAILURE_COUNTER = 23
    IMGS = 24


class R2D2ObsField(IntEnum):
    """Observation indices for MemoryR2D2.

    Layout: (checkpoints, speeds, accelerations, jerks, race_progress,
             input_steer, input_gas_pedal, input_brake, gear, aim_yaw,
             aim_pitch, steer_angle, slip_coef, failure_counter, imgs)
    """

    CHECKPOINTS = 0
    SPEEDS = 1
    ACCELERATIONS = 2
    JERKS = 3
    RACE_PROGRESS = 4
    INPUT_STEER = 5
    INPUT_GAS_PEDAL = 6
    INPUT_BRAKE = 7
    GEAR = 8
    AIM_YAW = 9
    AIM_PITCH = 10
    STEER_ANGLE = 11
    SLIP_COEF = 12
    FAILURE_COUNTER = 13
    IMGS = 14


class R2D2SophyObsField(IntEnum):
    """Observation indices for MemoryR2D2Sophy.

    Same as R2D2ObsField but TRACK_INFO replaces CHECKPOINTS and no IMGS.
    """

    TRACK_INFO = 0
    SPEEDS = 1
    ACCELERATIONS = 2
    JERKS = 3
    RACE_PROGRESS = 4
    INPUT_STEER = 5
    INPUT_GAS_PEDAL = 6
    INPUT_BRAKE = 7
    GEAR = 8
    AIM_YAW = 9
    AIM_PITCH = 10
    STEER_ANGLE = 11
    SLIP_COEF = 12
    FAILURE_COUNTER = 13
