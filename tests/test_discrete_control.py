"""Tests for discrete action space: Yosh-style 78-action table and brake tap."""

import numpy as np
from tmrl.custom.tm.utils.discrete_control import (
    BRAKE_TAP_SENTINEL,
    YOSH_N_BRAKE,
    YOSH_N_GAS,
    YOSH_N_STEER,
    build_discrete_to_continuous,
    build_yosh_action_table,
    discrete_index_to_control,
    is_brake_tap,
)


class TestBuildYoshActionTable:
    def test_default_table_size(self):
        n, table = build_yosh_action_table()
        assert n == YOSH_N_STEER * YOSH_N_GAS * YOSH_N_BRAKE
        assert n == 78
        assert len(table) == n

    def test_control_vector_shape(self):
        _, table = build_yosh_action_table()
        for ctrl in table:
            assert ctrl.shape == (3,)
            assert ctrl.dtype == np.float32

    def test_steer_range(self):
        _, table = build_yosh_action_table()
        steers = sorted({float(c[2]) for c in table})
        assert steers[0] == -1.0
        assert steers[-1] == 1.0

    def test_gas_is_binary(self):
        _, table = build_yosh_action_table()
        gas_vals = {float(c[0]) for c in table}
        assert gas_vals == {0.0, 1.0}

    def test_brake_modes(self):
        _, table = build_yosh_action_table()
        brake_vals = {float(c[1]) for c in table}
        assert brake_vals == {0.0, 1.0, float(BRAKE_TAP_SENTINEL)}

    def test_brake_tap_sentinel_detected(self):
        _, table = build_yosh_action_table()
        tap_controls = [c for c in table if is_brake_tap(c)]
        assert len(tap_controls) == YOSH_N_STEER * YOSH_N_GAS


class TestBuildDiscreteToContiguous:
    def test_legacy_table_size(self):
        n, table = build_discrete_to_continuous()
        assert n == 30
        assert len(table) == 30

    def test_custom_bins(self):
        n, table = build_discrete_to_continuous(n_steer=7, n_gas=2, n_brake=2)
        assert n == 28
        assert len(table) == 28


class TestDiscreteIndexToControl:
    def test_roundtrip(self):
        _, table = build_yosh_action_table()
        for i in range(len(table)):
            ctrl = discrete_index_to_control(i, table)
            assert np.allclose(ctrl, table[i])

    def test_returns_copy(self):
        _, table = build_yosh_action_table()
        ctrl = discrete_index_to_control(0, table)
        ctrl[0] = 999.0
        assert table[0][0] != 999.0


class TestIsBrakeTap:
    def test_positive(self):
        ctrl = np.array([1.0, BRAKE_TAP_SENTINEL, 0.5], dtype=np.float32)
        assert is_brake_tap(ctrl)

    def test_negative_no_brake(self):
        ctrl = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        assert not is_brake_tap(ctrl)

    def test_negative_full_brake(self):
        ctrl = np.array([1.0, 1.0, 0.5], dtype=np.float32)
        assert not is_brake_tap(ctrl)
