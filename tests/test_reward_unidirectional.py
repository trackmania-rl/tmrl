"""Tests for unidirectional waypoint reward logic.

Verifies that:
- Progress reward is only for moving forward beyond the furthest reached index
- Revisiting past waypoints produces zero reward
- The furthest_reached_idx ratchet only advances
"""

import numpy as np


def _compute_reward_progress(
    best_index: int,
    furthest_reached_idx: int,
    cumulative_dist: np.ndarray,
    total_traj_length: float,
    progress_reward_full_lap: float,
) -> tuple[float, int]:
    """Simplified version of the reward progress logic from compute_reward.py."""
    datalen = len(cumulative_dist)
    idx_furthest = min(furthest_reached_idx, datalen - 1)
    dist_furthest = cumulative_dist[idx_furthest]
    dist_best = cumulative_dist[min(best_index, datalen - 1)]
    distance_gained = max(0.0, float(dist_best - dist_furthest))
    reward_progress = float(distance_gained * (progress_reward_full_lap / total_traj_length))
    new_furthest = max(furthest_reached_idx, best_index)
    return reward_progress, new_furthest


class TestUnidirectionalReward:
    def setup_method(self):
        self.cumulative_dist = np.array([0.0, 10.0, 25.0, 45.0, 70.0, 100.0])
        self.total_length = 100.0
        self.full_lap_reward = 10.0

    def test_forward_progress_gives_reward(self):
        rew, new_f = _compute_reward_progress(
            best_index=3,
            furthest_reached_idx=1,
            cumulative_dist=self.cumulative_dist,
            total_traj_length=self.total_length,
            progress_reward_full_lap=self.full_lap_reward,
        )
        assert rew > 0.0
        assert new_f == 3

    def test_backward_gives_zero_reward(self):
        rew, new_f = _compute_reward_progress(
            best_index=1,
            furthest_reached_idx=3,
            cumulative_dist=self.cumulative_dist,
            total_traj_length=self.total_length,
            progress_reward_full_lap=self.full_lap_reward,
        )
        assert rew == 0.0
        assert new_f == 3, "Furthest should not decrease"

    def test_same_position_zero_reward(self):
        rew, new_f = _compute_reward_progress(
            best_index=2,
            furthest_reached_idx=2,
            cumulative_dist=self.cumulative_dist,
            total_traj_length=self.total_length,
            progress_reward_full_lap=self.full_lap_reward,
        )
        assert rew == 0.0
        assert new_f == 2

    def test_furthest_never_decreases(self):
        furthest = 0
        for idx in [1, 3, 2, 4, 1, 5]:
            _, furthest = _compute_reward_progress(
                best_index=idx,
                furthest_reached_idx=furthest,
                cumulative_dist=self.cumulative_dist,
                total_traj_length=self.total_length,
                progress_reward_full_lap=self.full_lap_reward,
            )
        assert furthest == 5

    def test_full_lap_reward_correct(self):
        rew, _ = _compute_reward_progress(
            best_index=5,
            furthest_reached_idx=0,
            cumulative_dist=self.cumulative_dist,
            total_traj_length=self.total_length,
            progress_reward_full_lap=self.full_lap_reward,
        )
        assert abs(rew - self.full_lap_reward) < 1e-6
