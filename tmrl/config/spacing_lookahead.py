"""Shared lookahead point count for TRACK_LOOK_AHEAD_PCT + TRACK_POINT_SPACING_M.

Must match :meth:`RewardFunction.__init__` in ``compute_reward.py`` so
``cfg.POINTS_NUMBER`` (trainer / IQN fingerprint) and ``RewardFunction._points_number``
(worker observation space) never diverge.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def polyline_arc_length_m(traj: Any) -> float | None:
    """Return total arc length of a polyline, or None if it cannot be computed."""
    if traj is None or not hasattr(traj, "__len__"):
        return None
    arr = np.asarray(traj)
    if len(arr) <= 1:
        return None
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return float(np.sum(diffs))


def points_number_from_spacing_config(
    traj_length_m: float,
    track_look_ahead_pct: float,
    track_point_spacing_m: float,
) -> int | None:
    """Compute POINTS_NUMBER from spacing settings (same formula as RewardFunction).

    Args:
        traj_length_m: Arc length of the reference trajectory (meters), before ``max(1,·)``.
        track_look_ahead_pct: Percent of trajectory length to look ahead.
        track_point_spacing_m: Nominal spacing between lookahead samples (meters).

    Returns:
        Integer in ``[1, 200]``, or ``None`` if spacing mode is disabled.
    """
    t_pct = float(track_look_ahead_pct)
    t_sp = float(track_point_spacing_m)
    if t_pct <= 0 or t_sp <= 0:
        return None
    total_traj = max(1.0, float(traj_length_m))
    look_ahead_dist = total_traj * (t_pct / 100.0)
    return min(200, max(1, math.ceil(look_ahead_dist / t_sp)))
