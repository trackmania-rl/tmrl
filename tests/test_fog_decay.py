"""Tests for Forget-and-Grow (FoG) replay decay resampling."""

import numpy as np
from tmrl.custom.memories import fog_recency_resample


class TestFogRecencyResample:
    def test_zero_temperature_returns_original(self):
        indices = tuple(range(100))
        result = fog_recency_resample(indices, buffer_len=1000, decay_temperature=0.0)
        assert result == indices

    def test_output_length_matches_input(self):
        indices = tuple(range(50))
        result = fog_recency_resample(indices, buffer_len=1000, decay_temperature=3.0)
        assert len(result) == len(indices)

    def test_recency_bias_shifts_distribution(self):
        """With high temperature, resampled indices should skew towards the end."""
        np.random.seed(42)
        indices = tuple(range(1000))
        result = fog_recency_resample(indices, buffer_len=1000, decay_temperature=5.0)
        mean_original = np.mean(indices)
        mean_resampled = np.mean(result)
        assert mean_resampled > mean_original

    def test_extreme_temperature_selects_newest(self):
        np.random.seed(42)
        indices = tuple(range(100))
        result = fog_recency_resample(indices, buffer_len=100, decay_temperature=50.0)
        assert np.mean(result) > 80

    def test_empty_indices(self):
        result = fog_recency_resample((), buffer_len=100, decay_temperature=3.0)
        assert result == ()

    def test_single_element_buffer(self):
        result = fog_recency_resample((0,), buffer_len=1, decay_temperature=3.0)
        assert result == (0,)

    def test_negative_temperature_treated_as_uniform(self):
        indices = tuple(range(50))
        result = fog_recency_resample(indices, buffer_len=100, decay_temperature=-1.0)
        assert result == indices

    def test_all_indices_valid(self):
        """All resampled indices must be from the original set."""
        indices = tuple(range(0, 500, 5))
        result = fog_recency_resample(indices, buffer_len=500, decay_temperature=3.0)
        original_set = set(indices)
        for idx in result:
            assert idx in original_set
