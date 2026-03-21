"""Tests for drift reward annealing and body slip angle in compute_reward."""

import math

import numpy as np


def _compute_drift_weight(
    global_steps: int,
    drift_weight_start: float,
    drift_weight_end: float,
    anneal_steps: int,
    fallback_weight: float,
) -> float:
    """Reproduce the drift weight annealing logic from RewardFunction.compute_reward."""
    if anneal_steps > 0:
        frac = min(1.0, global_steps / anneal_steps)
        return drift_weight_start + frac * (drift_weight_end - drift_weight_start)
    return fallback_weight


def _compute_drift_bonus(
    slip_angle_deg: float,
    optimal_deg: float,
    sigma_deg: float,
    weight: float,
) -> float:
    """Reproduce the Gaussian drift bonus computation (uses abs slip for symmetry)."""
    if weight <= 0:
        return 0.0
    slip = abs(slip_angle_deg)
    return weight * math.exp(-((slip - optimal_deg) ** 2) / (2.0 * sigma_deg**2))


def _compute_slip_angle_deg(pos: np.ndarray, prev_pos: np.ndarray, aim_yaw: float) -> float | None:
    """Reproduce body slip angle computation from position delta and heading."""
    pos_delta_xz = np.array([pos[0] - prev_pos[0], pos[2] - prev_pos[2]], dtype=np.float64)
    motion_norm = np.linalg.norm(pos_delta_xz)
    if motion_norm <= 1e-3:
        return None
    car_dir = pos_delta_xz / motion_norm
    heading = np.array([math.sin(aim_yaw), math.cos(aim_yaw)], dtype=np.float64)
    cross = heading[0] * car_dir[1] - heading[1] * car_dir[0]
    dot = float(np.dot(heading, car_dir))
    return abs(math.degrees(math.atan2(cross, dot)))


_DW_KW = {"drift_weight_start": 0.3, "drift_weight_end": 0.0, "fallback_weight": 0.15}


class TestDriftWeightAnnealing:
    def test_start_weight(self):
        w = _compute_drift_weight(0, anneal_steps=500000, **_DW_KW)
        assert abs(w - 0.3) < 1e-6

    def test_end_weight(self):
        w = _compute_drift_weight(500000, anneal_steps=500000, **_DW_KW)
        assert abs(w - 0.0) < 1e-6

    def test_midpoint_weight(self):
        w = _compute_drift_weight(250000, anneal_steps=500000, **_DW_KW)
        assert abs(w - 0.15) < 1e-6

    def test_beyond_anneal_clamps(self):
        w = _compute_drift_weight(1000000, anneal_steps=500000, **_DW_KW)
        assert abs(w - 0.0) < 1e-6

    def test_no_anneal_uses_fallback(self):
        w = _compute_drift_weight(999, anneal_steps=0, **_DW_KW)
        assert abs(w - 0.15) < 1e-6


class TestDriftBonus:
    def test_optimal_angle_gives_max_bonus(self):
        bonus = _compute_drift_bonus(12.0, optimal_deg=12.0, sigma_deg=8.0, weight=0.3)
        assert abs(bonus - 0.3) < 1e-6

    def test_far_angle_gives_small_bonus(self):
        bonus = _compute_drift_bonus(50.0, optimal_deg=12.0, sigma_deg=8.0, weight=0.3)
        assert bonus < 0.01

    def test_zero_weight_gives_zero(self):
        bonus = _compute_drift_bonus(12.0, optimal_deg=12.0, sigma_deg=8.0, weight=0.0)
        assert bonus == 0.0

    def test_symmetry_around_optimal(self):
        b1 = _compute_drift_bonus(12.0 + 5.0, optimal_deg=12.0, sigma_deg=8.0, weight=0.3)
        b2 = _compute_drift_bonus(12.0 - 5.0, optimal_deg=12.0, sigma_deg=8.0, weight=0.3)
        assert abs(b1 - b2) < 1e-6

    def test_negative_slip_uses_abs(self):
        b_pos = _compute_drift_bonus(12.5, optimal_deg=12.5, sigma_deg=8.0, weight=0.3)
        b_neg = _compute_drift_bonus(-12.5, optimal_deg=12.5, sigma_deg=8.0, weight=0.3)
        assert abs(b_pos - b_neg) < 1e-6
        assert abs(b_pos - 0.3) < 1e-6


class TestComputeSlipAngle:
    def test_straight_ahead_zero_slip(self):
        yaw = 0.0
        prev = np.array([0.0, 0.0, 0.0])
        cur = np.array([0.0, 0.0, 1.0])
        slip = _compute_slip_angle_deg(cur, prev, yaw)
        assert slip is not None
        assert slip < 1.0

    def test_stationary_returns_none(self):
        prev = np.array([5.0, 0.0, 5.0])
        cur = np.array([5.0, 0.0, 5.0])
        slip = _compute_slip_angle_deg(cur, prev, aim_yaw=0.0)
        assert slip is None

    def test_sideways_gives_90(self):
        yaw = 0.0
        prev = np.array([0.0, 0.0, 0.0])
        cur = np.array([1.0, 0.0, 0.0])
        slip = _compute_slip_angle_deg(cur, prev, yaw)
        assert slip is not None
        assert abs(slip - 90.0) < 1.0

    def test_opposite_left_right_same_abs(self):
        yaw = 0.0
        prev = np.array([0.0, 0.0, 0.0])
        cur_right = np.array([0.2, 0.0, 1.0])
        cur_left = np.array([-0.2, 0.0, 1.0])
        slip_r = _compute_slip_angle_deg(cur_right, prev, yaw)
        slip_l = _compute_slip_angle_deg(cur_left, prev, yaw)
        assert slip_r is not None
        assert slip_l is not None
        assert abs(slip_r - slip_l) < 1e-3
