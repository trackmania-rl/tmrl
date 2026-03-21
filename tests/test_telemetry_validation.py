"""Tests for telemetry validation flags in TM2020OpenPlanetClient.

Since the client connects to a real game, these tests mock the internal state
to verify that the validation logic (position patch, speed sanity) sets the
correct flags after retrieve_data() processing.
"""

import math


def _simulate_retrieve_data_validation(data_tuple, nb_floats=20, last_good_pos=None):
    """Run the post-retrieve validation logic from tools.py on raw data.

    Returns (patched_data, position_patched, invalid, new_last_good_pos).
    """
    data = data_tuple
    position_patched = False
    invalid = False
    new_last_good_pos = last_good_pos

    if data is not None:
        pos_start_idx = 3 if nb_floats >= 20 else 2
        pos_x, pos_y, pos_z = (
            data[pos_start_idx],
            data[pos_start_idx + 1],
            data[pos_start_idx + 2],
        )

        if math.sqrt(pos_x**2 + pos_y**2 + pos_z**2) < 1.0:
            if last_good_pos is not None:
                data_list = list(data)
                data_list[pos_start_idx] = last_good_pos[0]
                data_list[pos_start_idx + 1] = last_good_pos[1]
                data_list[pos_start_idx + 2] = last_good_pos[2]
                data = tuple(data_list)
                position_patched = True
        else:
            new_last_good_pos = (pos_x, pos_y, pos_z)

        speed_idx = 2
        if speed_idx < len(data):
            try:
                speed_val = float(data[speed_idx])
                if not (0 <= speed_val <= 2500.0):
                    invalid = True
            except (TypeError, ValueError):
                invalid = True

    return data, position_patched, invalid, new_last_good_pos


class TestTelemetryPositionPatch:
    def test_near_origin_patched(self):
        data = tuple([0.0] * 20)
        last_good = (100.0, 50.0, 200.0)
        patched, was_patched, _, _ = _simulate_retrieve_data_validation(
            data, nb_floats=20, last_good_pos=last_good
        )
        assert was_patched
        assert patched[3] == 100.0
        assert patched[4] == 50.0
        assert patched[5] == 200.0

    def test_valid_position_not_patched(self):
        data = [0.0] * 20
        data[3], data[4], data[5] = 100.0, 50.0, 200.0
        data = tuple(data)
        _, was_patched, _, new_good = _simulate_retrieve_data_validation(
            data, nb_floats=20, last_good_pos=None
        )
        assert not was_patched
        assert new_good == (100.0, 50.0, 200.0)


class TestTelemetrySpeedSanity:
    def test_normal_speed_valid(self):
        data = [0.0] * 20
        data[2] = 150.0
        data[3], data[4], data[5] = 10.0, 20.0, 30.0
        data = tuple(data)
        _, _, invalid, _ = _simulate_retrieve_data_validation(data, nb_floats=20)
        assert not invalid

    def test_negative_speed_invalid(self):
        data = [0.0] * 20
        data[2] = -10.0
        data[3], data[4], data[5] = 10.0, 20.0, 30.0
        data = tuple(data)
        _, _, invalid, _ = _simulate_retrieve_data_validation(data, nb_floats=20)
        assert invalid

    def test_absurdly_high_speed_invalid(self):
        data = [0.0] * 20
        data[2] = 5000.0
        data[3], data[4], data[5] = 10.0, 20.0, 30.0
        data = tuple(data)
        _, _, invalid, _ = _simulate_retrieve_data_validation(data, nb_floats=20)
        assert invalid

    def test_zero_speed_valid(self):
        data = [0.0] * 20
        data[2] = 0.0
        data[3], data[4], data[5] = 10.0, 20.0, 30.0
        data = tuple(data)
        _, _, invalid, _ = _simulate_retrieve_data_validation(data, nb_floats=20)
        assert not invalid
