"""Canonical TQC / Sophy tuple observation space from trainer config.

Single place for Box shapes so replay alignment, env, and IQN stay consistent.
Keep in sync with :meth:`TM2020InterfaceIMPALASophy.get_observation_space` fields.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

import tmrl.config as cfg


def build_tqc_sophy_tuple_observation_space(points_number: int | None = None) -> spaces.Tuple:
    """Tuple-of-Box layout for TQCGRAB-style telemetry (no images, no act-in-obs tail)."""
    n = int(cfg.POINTS_NUMBER if points_number is None else points_number)
    track = spaces.Box(low=-100.0, high=100.0, shape=(6 * n,))
    speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
    acceleration = spaces.Box(low=-100.0, high=100.0, shape=(1,))
    jerk = spaces.Box(low=-10.0, high=10.0, shape=(1,))
    race_progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
    input_steer = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    input_gas_pedal = spaces.Box(low=0.0, high=1.0, shape=(1,))
    input_brake = spaces.Box(low=0.0, high=1.0, shape=(1,))
    gear = spaces.Box(low=0.0, high=6.0, shape=(1,))
    aim_yaw = spaces.Box(low=-4.0, high=4.0, shape=(1,))
    aim_pitch = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    steer_angle = spaces.Box(low=-30.0, high=30.0, shape=(2,))
    slip_coef = spaces.Box(low=0.0, high=1.0, shape=(2,))
    failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))
    spaces_list = [
        track,
        speed,
        acceleration,
        jerk,
        race_progress,
        input_steer,
        input_gas_pedal,
        input_brake,
        gear,
        aim_yaw,
        aim_pitch,
        steer_angle,
        slip_coef,
        failure_counter,
    ]
    if bool(cfg.REWARD_CONFIG.get("TRACK_CURVATURE_OBS", False)):
        curvature = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        spaces_list.append(curvature)
    return spaces.Tuple(tuple(spaces_list))
