"""Tests for horizontal flip augmentation with discrete actions."""

import numpy as np
from tmrl.custom.memories import (
    _hflip_action,
    _hflip_discrete_action,
    _is_discrete_action,
)


class TestIsDiscreteAction:
    def test_scalar_int(self):
        assert _is_discrete_action(np.int64(5))

    def test_scalar_int32(self):
        assert _is_discrete_action(np.int32(5))

    def test_python_int(self):
        assert _is_discrete_action(0)

    def test_float_array(self):
        assert not _is_discrete_action(np.array([0.5, 0.0, 0.3], dtype=np.float32))

    def test_1d_int_array_is_not_scalar(self):
        assert not _is_discrete_action(np.array([5], dtype=np.int64))


class TestHflipDiscreteAction:
    def test_center_steer_unchanged(self):
        """For 13 steer bins, center is index 6. Flipping should stay at 6."""
        n_gas = 2
        n_brake = 3
        gas_brake = n_gas * n_brake
        center_steer = 6
        for gi in range(n_gas):
            for bi in range(n_brake):
                idx = center_steer * gas_brake + gi * n_brake + bi
                flipped = _hflip_discrete_action(idx)
                assert int(flipped) == idx

    def test_symmetry(self):
        """Flipping twice returns the original action."""
        for idx in range(78):
            double_flipped = _hflip_discrete_action(_hflip_discrete_action(idx))
            assert int(double_flipped) == idx

    def test_extremes_swap(self):
        """Leftmost steer (0) should map to rightmost (12) and vice versa."""
        gas_brake = 2 * 3
        idx_left = 0 * gas_brake
        idx_right = 12 * gas_brake
        assert int(_hflip_discrete_action(idx_left)) == idx_right
        assert int(_hflip_discrete_action(idx_right)) == idx_left

    def test_gas_brake_preserved(self):
        """Gas and brake components should not change during flip."""
        gas_brake = 2 * 3
        for idx in range(78):
            flipped = int(_hflip_discrete_action(idx))
            assert idx % gas_brake == flipped % gas_brake


class TestHflipAction:
    def test_discrete_delegates(self):
        action = np.int64(10)
        result = _hflip_action(action)
        expected = _hflip_discrete_action(10)
        assert int(result) == int(expected)

    def test_continuous_negates_steer(self):
        action = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        flipped = _hflip_action(action)
        assert flipped[0] == 1.0
        assert flipped[1] == 0.0
        assert flipped[2] == -0.5

    def test_continuous_double_flip_identity(self):
        action = np.array([0.8, 0.2, -0.3], dtype=np.float32)
        double = _hflip_action(_hflip_action(action))
        assert np.allclose(double, action)
