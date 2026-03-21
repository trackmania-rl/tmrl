"""
Discrete action space that maps to continuous gamepad control.

Keeps the gamepad API (and thus crash/vibration feedback) unchanged:
the policy outputs a discrete action index; we map it to [forward, backward, steer]
and pass that to the existing control_gamepad(), so vibrations on guardrail hit still work.

Supports two presets:
  - Legacy (30 actions): 5 steer x 3 gas x 2 brake
  - Yosh-style (78 actions): 13 steer x 2 gas x 3 brake (off / full / 0.01s tap)
"""

import numpy as np

# Legacy default bins: steer (5), gas (3), brake (2) -> 30 actions
DEFAULT_N_STEER = 5
DEFAULT_N_GAS = 3
DEFAULT_N_BRAKE = 2

# Yosh-style bins: 13 steer x 2 gas x 3 brake = 78 actions
YOSH_N_STEER = 13
YOSH_N_GAS = 2
YOSH_N_BRAKE = 3

BRAKE_TAP_SENTINEL = -1.0
BRAKE_TAP_DURATION_S = 0.01


def build_discrete_to_continuous(
    n_steer: int = DEFAULT_N_STEER,
    n_gas: int = DEFAULT_N_GAS,
    n_brake: int = DEFAULT_N_BRAKE,
) -> tuple[int, list[np.ndarray]]:
    """
    Build discrete action space size and mapping from index to continuous control.

    Control is [forward (gas), backward (brake), steer] in [0,1], [0,1], [-1,1] respectively.
    Action index = steer_idx * (n_gas * n_brake) + gas_idx * n_brake + brake_idx.

    Returns:
        n_actions: total number of discrete actions.
        table: list of length n_actions; table[i] is np.array([gas, brake, steer]).
    """
    n_actions = n_steer * n_gas * n_brake
    table = []
    for si in range(n_steer):
        steer = np.linspace(-1.0, 1.0, n_steer)[si]
        for gi in range(n_gas):
            gas = np.linspace(0.0, 1.0, n_gas)[gi] if n_gas > 1 else 1.0
            for bi in range(n_brake):
                brake = np.linspace(0.0, 1.0, n_brake)[bi] if n_brake > 1 else 0.0
                table.append(np.array([gas, brake, steer], dtype=np.float32))
    return n_actions, table


def build_yosh_action_table(
    n_steer: int = YOSH_N_STEER,
    n_gas: int = YOSH_N_GAS,
) -> tuple[int, list[np.ndarray]]:
    """Build Yosh-style 78-action table: n_steer x n_gas x 3 brake modes.

    Brake modes:
      0 = no brake (0.0)
      1 = full brake (1.0)
      2 = 0.01 s brake tap (encoded as BRAKE_TAP_SENTINEL)

    The sentinel is detected by send_control to fire a timed pulse.

    Returns:
        n_actions: total discrete actions (default 78).
        table: list[np.array([gas, brake, steer])].
    """
    brake_values = np.array([0.0, 1.0, BRAKE_TAP_SENTINEL], dtype=np.float32)
    n_brake = len(brake_values)
    n_actions = n_steer * n_gas * n_brake
    table: list[np.ndarray] = []
    for si in range(n_steer):
        steer = np.linspace(-1.0, 1.0, n_steer, dtype=np.float32)[si]
        for gi in range(n_gas):
            gas = float(gi)  # 0.0 or 1.0 for binary accel
            for bi in range(n_brake):
                table.append(np.array([gas, brake_values[bi], steer], dtype=np.float32))
    return n_actions, table


def is_brake_tap(control: np.ndarray) -> bool:
    """Return True if the control vector encodes a 0.01 s brake tap."""
    return float(control[1]) == BRAKE_TAP_SENTINEL


def discrete_index_to_control(
    action_index: int,
    table: list[np.ndarray],
) -> np.ndarray:
    """
    Map a single discrete action index to continuous control [forward, backward, steer].

    Same format as expected by send_control / control_gamepad.
    """
    return table[action_index].copy()


def discrete_indices_to_control_batch(
    action_indices: np.ndarray,
    table: list[np.ndarray],
) -> np.ndarray:
    """Map a batch of discrete indices to (batch, 3) continuous controls."""
    return np.array([table[int(i)] for i in action_indices], dtype=np.float32)


def continuous_control_to_discrete_index(
    control: np.ndarray,
    table: list[np.ndarray],
) -> int:
    """
    Map continuous [gas, brake, steer] to the nearest discrete action index.

    Used when replay contains continuous actions (e.g. player runs) but the
    agent expects discrete indices (e.g. IQN). Brake tap sentinel is treated
    as far from any continuous value so we match to off or full brake only.
    """
    c = np.asarray(control, dtype=np.float32).flat
    gas, brake, steer = float(c[0]), float(c[1]), float(c[2])
    best_i = 0
    best_d2 = np.inf
    for i, t in enumerate(table):
        tg, tb, ts = float(t[0]), float(t[1]), float(t[2])
        # Brake: continuous [0,1] vs table 0, 1, or BRAKE_TAP_SENTINEL
        # Tap sentinel: never match from continuous; use large distance
        b_d2 = 2.0 if tb == BRAKE_TAP_SENTINEL else (brake - tb) ** 2
        d2 = (gas - tg) ** 2 + b_d2 + (steer - ts) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def continuous_control_to_discrete_indices_batch(
    controls: np.ndarray,
    table: list[np.ndarray],
) -> np.ndarray:
    """Map (batch, 3) continuous controls to (batch,) discrete indices."""
    controls = np.asarray(controls, dtype=np.float32)
    if controls.ndim == 1:
        return np.array(continuous_control_to_discrete_index(controls, table), dtype=np.int64)
    idx_list = [
        continuous_control_to_discrete_index(controls[i], table) for i in range(controls.shape[0])
    ]
    return np.array(idx_list, dtype=np.int64)
