"""Tests for player-run observation alignment to trainer observation_space."""

import gymnasium as gym
import numpy as np
from tmrl.networking import Buffer
from tmrl.tools.player_runs import (
    align_buffer_observations_to_space,
    align_observation_to_space,
    observation_matches_space,
)


def _tqc_like_space(*, track_n: int, with_curvature: bool) -> gym.spaces.Tuple:
    track = gym.spaces.Box(-100.0, 100.0, shape=(6 * track_n,))
    parts = [
        track,
        gym.spaces.Box(0.0, 1.0, shape=(1,)),
        gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        gym.spaces.Box(0.0, 1.0, shape=(1,)),
        gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        gym.spaces.Box(0.0, 1.0, shape=(1,)),
        gym.spaces.Box(0.0, 1.0, shape=(1,)),
        gym.spaces.Box(0.0, 1.0, shape=(1,)),
        gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        gym.spaces.Box(-30.0, 30.0, shape=(2,)),
        gym.spaces.Box(0.0, 1.0, shape=(2,)),
        gym.spaces.Box(0.0, 1.0, shape=(1,)),
    ]
    if with_curvature:
        parts.append(gym.spaces.Box(-1.0, 1.0, shape=(track_n,), dtype=np.float32))
    return gym.spaces.Tuple(tuple(parts))


def test_align_trim_track_and_curvature_to_smaller_n():
    target = _tqc_like_space(track_n=2, with_curvature=True)
    src = _tqc_like_space(track_n=4, with_curvature=True)
    obs = tuple(
        np.ones(sp.shape, dtype=np.float32) * (0.1 * (i + 1)) for i, sp in enumerate(src.spaces)
    )
    assert not observation_matches_space(obs, target)
    out = align_observation_to_space(obs, target)
    assert observation_matches_space(out, target)
    assert np.asarray(out[0]).size == 12
    assert np.asarray(out[-1]).size == 2


def test_align_drops_trailing_when_target_has_no_curvature():
    target = _tqc_like_space(track_n=2, with_curvature=False)
    src = _tqc_like_space(track_n=2, with_curvature=True)
    obs = tuple(np.ones(sp.shape, dtype=np.float32) for sp in src.spaces)
    assert not observation_matches_space(obs, target)
    out = align_observation_to_space(obs, target)
    assert observation_matches_space(out, target)
    assert len(out) == len(target.spaces)


def test_align_buffer_in_place():
    target = _tqc_like_space(track_n=2, with_curvature=True)
    src = _tqc_like_space(track_n=3, with_curvature=True)
    buf = Buffer()
    obs = tuple(np.ones(sp.shape, dtype=np.float32) for sp in src.spaces)
    buf.append_sample((np.zeros(3, dtype=np.float32), obs, 0.0, False, False, {}))
    n = align_buffer_observations_to_space(buf, target)
    assert n == 1
    _, o, *_ = buf.memory[0]
    assert observation_matches_space(o, target)
