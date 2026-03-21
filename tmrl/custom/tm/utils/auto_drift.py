"""Auto-drift steering override for speed-drifts.

In TrackMania, speed-drifts require holding a precise steering angle that
depends on the current speed. The relationship is approximately:

    optimal_steer = clamp(base / (speed_kmh * sensitivity), -1, 1)

When the AI selects the auto-drift meta-action, the interface overrides the
discrete steering angle with the computed optimal angle for one tick while
preserving the gas/brake components from the selected action.

The constants below are tuned for TM2020 Stadium car physics.  They can be
refined with empirical testing or per-surface calibration.
"""

import math

import numpy as np

AUTO_DRIFT_K = 80.0
AUTO_DRIFT_MIN_SPEED_KMH = 80.0


def compute_drift_steer(speed_kmh: float) -> float:
    """Compute the optimal steering angle for a speed-drift.

    Uses an inverse relationship: steer = k / speed.  Faster speeds
    require tighter (smaller) steering inputs.  Examples with k=80:
      100 km/h -> 0.80,  200 km/h -> 0.40,  300 km/h -> 0.27

    Below AUTO_DRIFT_MIN_SPEED_KMH the car isn't fast enough to drift,
    so we return 0 (neutral steering).

    Args:
        speed_kmh: Current car speed in km/h.

    Returns:
        Steering value in [0, 1] suitable for the gamepad API.
        The sign is always positive (right); the interface should flip it
        based on the upcoming corner direction if needed.
    """
    if speed_kmh < AUTO_DRIFT_MIN_SPEED_KMH:
        return 0.0
    raw = AUTO_DRIFT_K / speed_kmh
    return float(np.clip(raw, 0.0, 1.0))


def build_yosh_action_table_with_drift(
    n_steer: int = 13,
    n_gas: int = 2,
) -> tuple[int, list[np.ndarray], int]:
    """Build the 78 + 6 action table: normal Yosh actions + auto-drift actions.

    Auto-drift actions have steer=NaN as a sentinel. The interface detects
    NaN steering and replaces it with compute_drift_steer(speed) at runtime.

    For each (gas, brake_mode) combination there is one auto-drift action,
    giving 2 gas x 3 brake = 6 extra actions.

    Returns:
        n_actions: total actions (84).
        table: list[np.array([gas, brake, steer])].
        auto_drift_start_idx: index of the first auto-drift action.
    """
    from tmrl.custom.tm.utils.discrete_control import (
        BRAKE_TAP_SENTINEL,
        build_yosh_action_table,
    )

    base_n, base_table = build_yosh_action_table(n_steer=n_steer, n_gas=n_gas)

    brake_values = [0.0, 1.0, BRAKE_TAP_SENTINEL]
    auto_drift_start = base_n
    for gi in range(n_gas):
        gas = float(gi)
        for bv in brake_values:
            base_table.append(np.array([gas, bv, math.nan], dtype=np.float32))

    n_actions = len(base_table)
    return n_actions, base_table, auto_drift_start


def is_auto_drift_action(control: np.ndarray) -> bool:
    """Return True if control has NaN steering (auto-drift sentinel)."""
    return bool(np.isnan(control[2]))
