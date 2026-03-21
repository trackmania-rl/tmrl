"""Replay memory utilities: FoG resampling and horizontal-flip action helpers."""

from __future__ import annotations

import numpy as np

# Steering is at index 2 in [gas, brake, steer] (control_gamepad / send_control).
ACTION_STEER_INDEX = 2


def fog_recency_resample(
    indices: tuple[int, ...] | list[int],
    buffer_len: int,
    decay_temperature: float = 3.0,
) -> tuple[int, ...]:
    """Forget-and-Grow (FoG) replay decay: resample *indices* with exponential recency bias.

    Each index *i* (representing its position in the circular buffer) is assigned a
    sampling weight  ``exp(decay_temperature * i / buffer_len)``.  More recent
    transitions (higher *i*) therefore receive exponentially higher probability,
    discouraging the agent from anchoring on the earliest, low-quality experiences
    ("primacy bias").

    Args:
        indices: Raw indices produced by the base ``sample_indices()`` call.
        buffer_len: Current number of valid items in the replay buffer.
        decay_temperature: Controls the steepness of the recency curve.
            Higher values forget old data more aggressively.  0 = uniform (no FoG).

    Returns:
        A tuple of resampled indices with the same length as *indices*.
    """
    if decay_temperature <= 0.0 or buffer_len <= 1 or len(indices) == 0:
        return tuple(indices)

    idx_arr = np.asarray(indices, dtype=np.int64)
    log_weights = decay_temperature * (idx_arr.astype(np.float64) / buffer_len)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = weights.sum()
    if total <= 0:
        return tuple(indices)
    probs = weights / total
    resampled = np.random.choice(idx_arr, size=len(idx_arr), replace=True, p=probs)
    return tuple(int(x) for x in resampled)


def _is_discrete_action(action) -> bool:
    """True when action is a scalar integer (DQN discrete index)."""
    arr = np.asarray(action)
    return arr.ndim == 0 and np.issubdtype(arr.dtype, np.integer)


def _hflip_discrete_action(action_idx, n_steer: int | None = None):
    """Mirror the steering component inside a Yosh-style discrete action index.

    Index layout: steer_idx * (n_gas * n_brake) + gas_idx * n_brake + brake_idx.
    Mirroring steering: new_steer = n_steer - 1 - steer_idx (symmetric about 0).

    Args:
        action_idx: Scalar integer action index.
        n_steer: Number of steer bins. If None, uses config IQN_N_STEER_BINS or 13.
    """
    from tmrl.custom.tm.utils.discrete_control import YOSH_N_BRAKE, YOSH_N_GAS

    if n_steer is None:
        try:
            import tmrl.config.constants as cfg

            n_steer = int(cfg.ALG_CONFIG.get("IQN_N_STEER_BINS", 13))
        except Exception:
            n_steer = 13
    n_steer = int(n_steer)
    idx = int(action_idx)
    gas_brake = YOSH_N_GAS * YOSH_N_BRAKE
    steer_idx = idx // gas_brake
    remainder = idx % gas_brake
    return np.int64((n_steer - 1 - steer_idx) * gas_brake + remainder)


def _hflip_action(action):
    """Negate the steering component (index 2) of an action array.

    Action order is [gas, brake, steer] (control_gamepad / send_control).
    Do NOT negate index 0 (gas) or index 1 (brake); only steer (index 2) flips.
    For discrete (DQN) actions, mirrors the steer index within the composite action.
    """
    if _is_discrete_action(action):
        return _hflip_discrete_action(action)
    action_arr = np.array(action, dtype=np.float32, copy=True)
    if len(action_arr) >= 3:
        action_arr[ACTION_STEER_INDEX] *= -1.0
    return action_arr
