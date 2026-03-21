"""Reward and termination logic for TrackMania 2020 RL.

RewardFunction computes step reward and termination from position, speed,
and inputs. It uses a reference trajectory and optional track boundaries
for progress, speed-along-track, and off-track/stall detection.
"""

import atexit
import math
import os
import pickle
import shutil
import tempfile

import numpy as np
from loguru import logger

import tmrl.config as cfg
import wandb
from tmrl.config.spacing_lookahead import points_number_from_spacing_config

OFF_TRACK_PROGRESS_ZERO_MULTIPLIER = 2.0
SPEED_REWARD_MIN_KMH = 5.0
TIME_STEP_SECONDS = 0.05
_BARRIER_SEARCH_WINDOW = 15


def _point_to_segment_dist_2d(
    px: float, pz: float, ax: float, az: float, bx: float, bz: float
) -> float:
    """Orthogonal distance from point (px,pz) to segment (ax,az)-(bx,bz) in the XZ plane."""
    dx, dz = bx - ax, bz - az
    len_sq = dx * dx + dz * dz
    if len_sq < 1e-18:
        return math.sqrt((px - ax) ** 2 + (pz - az) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / len_sq))
    cx = ax + t * dx
    cz = az + t * dz
    return math.sqrt((px - cx) ** 2 + (pz - cz) ** 2)


def _min_dist_to_polyline_xz(
    px: float, pz: float, polyline_xz: np.ndarray, center_idx: int, window: int
) -> float:
    """Min orthogonal distance from (px,pz) to polyline segments near center_idx.

    Args:
        px, pz: Query point in XZ plane.
        polyline_xz: Shape (N, 2) array of polyline vertices in XZ.
        center_idx: Index around which to search.
        window: Half-window size for the search.

    Returns:
        Minimum perpendicular distance to any segment in the window.
    """
    n = len(polyline_xz)
    lo = max(0, center_idx - window)
    hi = min(n - 1, center_idx + window)
    best = float("inf")
    for i in range(lo, hi):
        d = _point_to_segment_dist_2d(
            px,
            pz,
            polyline_xz[i, 0],
            polyline_xz[i, 1],
            polyline_xz[i + 1, 0],
            polyline_xz[i + 1, 1],
        )
        if d < best:
            best = d
    return best


def _resample_polyline_by_arc_length(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a polyline to a fixed number of points uniformly by arc length.

    Args:
        points: Array of shape (N, 3) or (N, 2) forming the polyline.
        num_points: Desired number of points in the resampled polyline.

    Returns:
        Resampled polyline of shape (num_points, ...) and dtype float64.
    """
    points = np.asarray(points, dtype=np.float64)
    num_input = len(points)
    if num_input <= 1 or num_points <= 1:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_length = np.zeros(num_input)
    np.cumsum(segment_lengths, out=cumulative_length[1:])
    total_length = float(cumulative_length[-1])
    if total_length <= 0:
        return points.copy()
    arc_positions = np.linspace(0.0, total_length, num_points, endpoint=True)
    result = []
    segment_index = 0
    for arc_s in arc_positions:
        if arc_s >= total_length:
            result.append(points[-1].copy())
            continue
        while segment_index + 1 < num_input and cumulative_length[segment_index + 1] < arc_s:
            segment_index += 1
        if segment_index + 1 >= num_input:
            result.append(points[-1].copy())
            continue
        seg_start = cumulative_length[segment_index]
        seg_end = cumulative_length[segment_index + 1]
        interp = (arc_s - seg_start) / (seg_end - seg_start) if seg_end > seg_start else 0.0
        interp = np.clip(interp, 0.0, 1.0)
        result.append((1.0 - interp) * points[segment_index] + interp * points[segment_index + 1])
    return np.array(result, dtype=np.float64)


class RewardFunction:
    """Reward and termination logic for TrackMania 2020 RL.

    Uses OpenPlanet API data. Rewards progress along a reference trajectory,
    speed along the track, and applies penalties for crash, steering jerk,
    and constant penalty. Handles termination (stall, off-track, end-of-track).
    """

    def __init__(
        self,
        reward_data_path: str,
        nb_obs_forward: int = 8,
        nb_obs_backward: int = 8,
        min_nb_steps_before_failure: int = int(2.5 * 20),
        max_dist_from_traj: float = 23.5,
        crash_penalty: float = 10.0,
        constant_penalty: float = 0.0,
    ) -> None:
        """Initialize reward function with trajectory and config.

        Args:
            reward_data_path: Path to the reference trajectory pickle file.
            nb_obs_forward: Number of trajectory points to look forward for progress.
            nb_obs_backward: Number of trajectory points to look backward (rewind check).
            min_nb_steps_before_failure: Minimum steps before failure can trigger.
            max_dist_from_traj: Max distance from trajectory before off-track failure.
            crash_penalty: Penalty applied on crash event.
            constant_penalty: Per-step penalty.
        """
        self.reward_data_path = reward_data_path
        if not os.path.exists(reward_data_path):
            logger.warning(f"Reward trajectory not found at {reward_data_path}. Using dummy.")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            self._dummy_trajectory = True
        else:
            with open(reward_data_path, "rb") as f:
                self.data = pickle.load(f)
            self._dummy_trajectory = len(self.data) <= 2

        self.datalen = len(self.data)

        if not os.path.exists(cfg.TRACK_PATH_LEFT):
            self.left_track = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        else:
            with open(cfg.TRACK_PATH_LEFT, "rb") as f:
                self.left_track = np.asarray(pickle.load(f), dtype=np.float64)

        if not os.path.exists(cfg.TRACK_PATH_RIGHT):
            self.right_track = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        else:
            with open(cfg.TRACK_PATH_RIGHT, "rb") as f:
                self.right_track = np.asarray(pickle.load(f), dtype=np.float64)

        self._has_boundaries = (
            len(self.left_track) >= 2 and len(self.right_track) >= 2 and self.datalen >= 2
        )
        if self._has_boundaries:
            if len(self.left_track) != self.datalen:
                self.left_track = _resample_polyline_by_arc_length(self.left_track, self.datalen)
            if len(self.right_track) != self.datalen:
                self.right_track = _resample_polyline_by_arc_length(self.right_track, self.datalen)
            self._left_xz = self.left_track[:, [0, 2]].astype(np.float64).copy()
            self._right_xz = self.right_track[:, [0, 2]].astype(np.float64).copy()
            self._road_center_xz = (self._left_xz + self._right_xz) / 2.0
            self._road_half_widths = np.linalg.norm(self._left_xz - self._right_xz, axis=1) / 2.0
            self._road_half_widths = np.maximum(self._road_half_widths, 0.5)
            _hw = self._road_half_widths
            logger.info(
                "Track boundaries loaded: {} points, road half-width "
                "min={:.1f}m mean={:.1f}m max={:.1f}m",
                self.datalen,
                float(_hw.min()),
                float(_hw.mean()),
                float(_hw.max()),
            )
        else:
            self._left_xz = np.zeros((2, 2), dtype=np.float64)
            self._right_xz = np.zeros((2, 2), dtype=np.float64)
            self._road_center_xz = np.zeros((2, 2), dtype=np.float64)
            self._road_half_widths = np.ones(2, dtype=np.float64) * 12.0

        self.cur_idx = 0
        self.prev_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward

        cfg_min_steps = int(cfg.REWARD_CONFIG.get("MIN_STEPS", min_nb_steps_before_failure))
        self.min_nb_steps_before_failure = max(1, int(cfg_min_steps))
        self.max_dist_from_traj = float(cfg.REWARD_CONFIG.get("MAX_STRAY", max_dist_from_traj))

        max_time_no_progress = float(cfg.REWARD_CONFIG.get("MAX_TIME_NO_PROGRESS_SECONDS", 0.0))
        if max_time_no_progress > 0:
            rtgym = getattr(cfg, "ENV_CONFIG", {}).get("RTGYM_CONFIG", {})
            raw_dt = float(rtgym.get("time_step_duration", 0.05))
            if raw_dt >= 1.0:
                self._time_step_duration = raw_dt / 1000.0
            else:
                self._time_step_duration = raw_dt
            self._max_no_progress_steps = max(
                1,
                int(round(max_time_no_progress / self._time_step_duration)),  # noqa: RUF046
            )
            steps_if_005 = max(1, int(round(max_time_no_progress / 0.05)))  # noqa: RUF046
            if max_time_no_progress >= 1.0 and self._max_no_progress_steps < steps_if_005 // 2:
                logger.warning(
                    "No-progress steps {} low for {:.1f}s (dt={:.3f}s); dt=0.05s -> {} steps.",
                    self._max_no_progress_steps,
                    max_time_no_progress,
                    self._time_step_duration,
                    steps_if_005,
                )
                self._time_step_duration = 0.05
                self._max_no_progress_steps = steps_if_005
            self._use_time_no_progress = True
            self._last_progress_step = 0
            config_path = getattr(cfg, "CONFIG_FILE_PATH", "config.json")
            logger.info(
                "Reward: time-based no-progress timeout {:.1f}s ({} steps). Config: {}",
                max_time_no_progress,
                self._max_no_progress_steps,
                config_path,
            )
        else:
            self._time_step_duration = TIME_STEP_SECONDS
            self._max_no_progress_steps = 0
            self._use_time_no_progress = False
            self._last_progress_step = 0
            config_path = getattr(cfg, "CONFIG_FILE_PATH", "config.json")
            logger.info(
                "Reward: no-progress only via MAX_TIME_NO_PROGRESS_SECONDS. Set in {} to enable.",
                config_path,
            )

        self.step_counter = 0
        self.failure_counter = 0

        self.average_distance = self.calculate_average_distance()
        self._cumulative_dist = np.zeros(max(1, self.datalen))
        if self.datalen > 1:
            diffs = np.linalg.norm(np.diff(self.data, axis=0), axis=1)
            np.cumsum(diffs, out=self._cumulative_dist[1:])
        self._total_traj_length = (
            max(1.0, float(self._cumulative_dist[-1])) if self.datalen >= 1 else 1.0
        )

        self._progress_reward_full_lap = float(
            cfg.REWARD_CONFIG.get("PROGRESS_REWARD_FULL_LAP", 200.0)
        )
        self._speed_reward_weight = float(cfg.REWARD_CONFIG.get("SPEED_REWARD_WEIGHT", 0.25))
        self._speed_reward_exponent = float(cfg.REWARD_CONFIG.get("SPEED_REWARD_EXPONENT", 1.0))
        self._speed_reward_alignment_floor = float(
            cfg.REWARD_CONFIG.get("SPEED_REWARD_ALIGNMENT_FLOOR", 0.0)
        )
        self._max_speed_kmh = float(cfg.REWARD_CONFIG.get("MAX_SPEED_KMH", 100.0))
        self._constant_penalty = float(cfg.REWARD_CONFIG.get("CONSTANT_PENALTY", constant_penalty))
        self._drift_reward_weight = float(cfg.REWARD_CONFIG.get("DRIFT_REWARD_WEIGHT", 0.0))
        self._drift_optimal_angle_deg = float(
            cfg.REWARD_CONFIG.get("DRIFT_OPTIMAL_ANGLE_DEG", 12.0)
        )
        self._drift_sigma_deg = float(cfg.REWARD_CONFIG.get("DRIFT_SIGMA_DEG", 8.0))
        self._drift_threshold_kmh = float(cfg.REWARD_CONFIG.get("DRIFT_THRESHOLD_KMH", 80.0))
        self._max_track_width = float(cfg.REWARD_CONFIG.get("MAX_TRACK_WIDTH", 35.0))
        self.crash_penalty = float(cfg.REWARD_CONFIG.get("CRASH_PENALTY", 2.0))
        self._reward_clip_floor = float(cfg.REWARD_CONFIG.get("REWARD_CLIP_FLOOR", 5.0))
        self._reward_scale = float(cfg.REWARD_CONFIG.get("REWARD_SCALE", 1.0))
        self._end_of_track_reward = float(cfg.REWARD_CONFIG.get("END_OF_TRACK_REWARD", 10.0))
        self._track_curvature_obs = bool(cfg.REWARD_CONFIG.get("TRACK_CURVATURE_OBS", False))
        self._progress_min_alignment = float(cfg.REWARD_CONFIG.get("PROGRESS_MIN_ALIGNMENT", 0.0))
        self._velocity_alignment_reward_weight = float(
            cfg.REWARD_CONFIG.get("VELOCITY_ALIGNMENT_REWARD_WEIGHT", 0.0)
        )
        self._barrier_touch_penalty = float(cfg.REWARD_CONFIG.get("BARRIER_TOUCH_PENALTY", 0.0))
        self._barrier_touch_radius = float(cfg.REWARD_CONFIG.get("BARRIER_TOUCH_RADIUS", 0.25))
        self._barrier_touch_min_speed_kmh = float(
            cfg.REWARD_CONFIG.get("BARRIER_TOUCH_MIN_SPEED_KMH", 5.0)
        )

        self._drift_weight_start = float(
            cfg.REWARD_CONFIG.get("DRIFT_REWARD_WEIGHT_START", self._drift_reward_weight)
        )
        self._drift_weight_end = float(cfg.REWARD_CONFIG.get("DRIFT_REWARD_WEIGHT_END", 0.0))
        self._drift_anneal_steps = int(cfg.REWARD_CONFIG.get("DRIFT_ANNEAL_STEPS", 0))
        self._global_env_steps: int = 0

        self._term_reason: str | None = None
        self.lap_cur_cooldown = cfg.LAP_COOLDOWN
        self.new_lap = False
        self.episode_reward = 0.0
        self.furthest_race_progress = 0.0
        self.furthest_reached_idx = 0
        self._logged_run_this_episode = False
        self._use_wandb = getattr(cfg, "WANDB_WORKER", True)
        self._checkpoint_stride = max(
            1, min(len(self.data), int(cfg.POINTS_DISTANCE / max(self.average_distance, 0.01)))
        )
        _track_pct = float(cfg.REWARD_CONFIG.get("TRACK_LOOK_AHEAD_PCT", 0.0))
        _track_spacing = float(cfg.REWARD_CONFIG.get("TRACK_POINT_SPACING_M", 0.0))
        self._points_number: int | None
        if _track_pct > 0 and _track_spacing > 0 and self.datalen > 1:
            self._points_number = points_number_from_spacing_config(
                self._total_traj_length, _track_pct, _track_spacing
            )
            self._point_spacing_m = _track_spacing
        else:
            self._point_spacing_m = 0.0
            self._points_number = None

        self._debug_reward = bool(cfg.REWARD_CONFIG.get("DEBUG_REWARD_COMPONENTS", False))
        self._debug_log_interval = int(cfg.REWARD_CONFIG.get("DEBUG_LOG_INTERVAL", 100))
        self.prev_pos: np.ndarray | None = None
        self._reset_debug_accumulators()

        if self._use_wandb:
            wandb_dir = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)
            cfg.ensure_wandb_api_key()
            try:
                wandb.init(
                    project=cfg.WANDB_PROJECT,
                    entity=cfg.WANDB_ENTITY,
                    id=cfg.WANDB_RUN_ID + " WORKER",
                    config=cfg.create_config(),
                    job_type="worker",
                    dir=wandb_dir,
                )
            except Exception as exc:
                logger.warning(f"wandb error: {exc}")

    def get_n_next_checkpoints_xy(
        self, position: list[float], number_of_next_points: int
    ) -> list[float]:
        """Next N checkpoint (x, z) coordinates relative to position, scaled by 10.

        Args:
            position: Current (x, y, z) position of the agent.
            number_of_next_points: Number of future trajectory points to return.

        Returns:
            Flattened list of (delta_x, delta_z) per point, each scaled by 10.
        """
        max_idx = len(self.data) - 1
        next_indices = [
            min(self.cur_idx + step * self._checkpoint_stride, max_idx)
            for step in range(1, number_of_next_points + 1)
        ]
        route_to_next_poses = []
        for pos_index in next_indices:
            for axis in (0, -1):
                route_to_next_poses.append((self.data[pos_index][axis] - position[axis]) * 10.0)
        return route_to_next_poses

    def _nearest_index_in_window(
        self, pos: np.ndarray, start_idx: int, end_idx: int
    ) -> tuple[int, float]:
        """Find nearest trajectory index to `pos` in the half-open window [start_idx, end_idx)."""
        if end_idx <= start_idx:
            idx = min(max(start_idx, 0), self.datalen - 1)
            return idx, float(np.linalg.norm(pos - self.data[idx]))

        segment = self.data[start_idx:end_idx]
        dists = np.linalg.norm(segment - pos, axis=1)
        best_local = int(np.argmin(dists))
        best_index = start_idx + best_local
        min_dist = float(dists[best_local])
        return best_index, min_dist

    def get_track_info(
        self, position: list[float], points_number: int
    ) -> (
        tuple[list[float], list[float], list[float]]
        | tuple[list[float], list[float], list[float], list[float]]
    ):
        """Track boundary (left, center, right) positions relative to current position.

        Args:
            position: Current (x, y, z) position of the agent.
            points_number: Number of look-ahead points (ignored if spacing-based points used).

        Returns:
            Tuple of (left_positions, center_positions, right_positions), each
            a flat list of relative coordinates.
        """
        max_idx = min(len(self.data), len(self.left_track), len(self.right_track)) - 1
        if getattr(self, "_point_spacing_m", 0) > 0 and getattr(self, "_points_number", 0) > 0:
            next_indices = []
            cur_idx = self.cur_idx
            cur_dist = self._cumulative_dist[min(cur_idx, self.datalen - 1)]
            n_pts = self._points_number or 0
            for _ in range(n_pts):
                target_dist = cur_dist + self._point_spacing_m
                while (
                    cur_idx < self.datalen - 1 and self._cumulative_dist[cur_idx + 1] <= target_dist
                ):
                    cur_idx += 1
                if cur_idx < self.datalen - 1:
                    cur_idx += 1
                    cur_dist = self._cumulative_dist[cur_idx]
                else:
                    cur_dist = self._cumulative_dist[-1] + self._point_spacing_m
                next_indices.append(min(cur_idx, max_idx))
            points_number = len(next_indices)
        else:
            next_indices = [
                self.cur_idx + step * self._checkpoint_stride + 1 for step in range(points_number)
            ]
            for idx in range(len(next_indices)):
                if next_indices[idx] > max_idx:
                    next_indices[idx] = max_idx

        left_positions, center_positions, right_positions = [], [], []
        for pos_index in next_indices:
            for axis in (0, -1):
                left_val = self.left_track[pos_index][axis]
                right_val = self.right_track[pos_index][axis]
                center_val = (left_val + right_val) / 2.0
                left_positions.append(left_val - position[axis])
                center_positions.append(center_val - position[axis])
                right_positions.append(right_val - position[axis])

        if getattr(self, "_track_curvature_obs", False) and len(next_indices) > 0:
            curvatures = []
            for _k, pos_index in enumerate(next_indices):
                i0 = max(0, pos_index - 1)
                i1 = pos_index
                i2 = min(self.datalen - 1, pos_index + 1)
                if i0 >= i2 or self.datalen < 2:
                    curvatures.append(0.0)
                    continue
                p0 = self.data[i0]
                p1 = self.data[i1]
                p2 = self.data[i2]
                v1 = np.array([p1[0] - p0[0], p1[2] - p0[2]], dtype=np.float64)
                v2 = np.array([p2[0] - p1[0], p2[2] - p1[2]], dtype=np.float64)
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    curvatures.append(0.0)
                    continue
                v1, v2 = v1 / n1, v2 / n2
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = math.acos(dot)
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                sign = 1.0 if cross >= 0 else -1.0
                arc = 0.5 * (n1 + n2)
                kappa = sign * (angle / arc) if arc > 1e-9 else 0.0
                curvatures.append(float(kappa))
            return left_positions, center_positions, right_positions, curvatures
        return left_positions, center_positions, right_positions

    def calculate_average_distance(self) -> float:
        """Mean segment length between consecutive trajectory points."""
        distances = np.linalg.norm(np.diff(self.data, axis=0), axis=1)
        return float(np.mean(distances))

    def _reset_debug_accumulators(self):
        """Resets debug metric accumulators."""
        self._dbg_speeds_kmh = []
        self._dbg_speed_rewards = []
        self._dbg_progress_rewards = []
        self._dbg_barrier_penalties = []
        self._dbg_crash_steps = 0
        self._dbg_barrier_touch_steps = 0
        self._dbg_end_of_track_awarded = False

    def _log_episode_debug_summary(self):
        """Logs a per-component reward breakdown for the finished episode."""
        if not self._dbg_speeds_kmh:
            return
        steps = len(self._dbg_speeds_kmh)
        speeds = np.array(self._dbg_speeds_kmh)
        logger.info(f" Episode summary ({steps} steps):")
        logger.info(
            f"  Speed km/h: mean={float(np.mean(speeds)):.1f} max={float(np.max(speeds)):.1f}"
        )

        progress_sum = (
            float(np.sum(self._dbg_progress_rewards)) if self._dbg_progress_rewards else 0.0
        )
        speed_sum = float(np.sum(self._dbg_speed_rewards)) if self._dbg_speed_rewards else 0.0
        barrier_sum = (
            float(np.sum(self._dbg_barrier_penalties)) if self._dbg_barrier_penalties else 0.0
        )
        bt = getattr(self, "_dbg_barrier_touch_steps", 0)

        const_sum = -self._constant_penalty * steps
        eot = (
            self._end_of_track_reward if getattr(self, "_dbg_end_of_track_awarded", False) else 0.0
        )

        logger.info(
            f"  progress: {progress_sum:+.1f}  |  speed: {speed_sum:+.1f}  |"
            f"  constant: {const_sum:+.1f}"
        )
        approx = progress_sum + speed_sum + barrier_sum + const_sum + eot
        logger.info(
            f"  barrier: {barrier_sum:+.1f}"
            f" ({bt}/{steps} steps, {100 * bt / max(1, steps):.1f}%)"
            f"  |  end_of_track: {eot:+.1f}"
            f"  |  approx_total: {approx:+.1f}"
        )

    def compute_reward(
        self,
        pos,
        crashed: bool = False,
        speed: float | None = None,
        next_cp: bool = False,
        next_lap: bool = False,
        end_of_track: bool = False,
        input_brake: float | None = None,
        aim_yaw: float | None = None,
        input_steer: float | None = None,
        gear: float | None = None,
        slip_angle_deg: float | None = None,
    ) -> tuple[float, bool, int, float]:
        """
        Computes the reward and termination status for the current step.

        Args:
            pos: Current position.
            crashed (bool): Whether the agent crashed. Defaults to False.
            speed (float, optional): Current speed in km/h. Defaults to None.
            next_cp (bool): Whether a checkpoint was passed. Defaults to False.
            next_lap (bool): Whether a lap was completed. Defaults to False.
            end_of_track (bool): Whether the end of track was reached. Defaults to False.
            input_brake (float, optional): Current brake input. Defaults to None.
            aim_yaw (float, optional): Current vehicle yaw. Defaults to None.
            input_steer (float, optional): Current steering input. Defaults to None.
            gear (float, optional): Current gear. Defaults to None.
            slip_angle_deg (float, optional): Slip angle (deg) for drift reward. Defaults to None.

        Returns:
            Tuple of (reward, terminated, failure_counter, episode_reward).
        """
        terminated = False
        self.step_counter += 1
        self.prev_idx = self.cur_idx
        _speed_kmh = float(speed) if speed is not None else 0.0

        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        start_fwd = self.cur_idx
        end_fwd = min(self.cur_idx + self.nb_obs_forward, self.datalen)
        best_index, min_dist = self._nearest_index_in_window(pos, start_fwd, end_fwd)
        reward_progress = 0.0
        if self.datalen > 1 and self._total_traj_length > 0:
            idx_furthest = min(self.furthest_reached_idx, self.datalen - 1)
            dist_furthest = self._cumulative_dist[idx_furthest]
            dist_best = self._cumulative_dist[min(best_index, self.datalen - 1)]
            distance_gained = max(0.0, float(dist_best - dist_furthest))
            reward_progress = float(
                distance_gained * (self._progress_reward_full_lap / self._total_traj_length)
            )
            if min_dist > OFF_TRACK_PROGRESS_ZERO_MULTIPLIER * self.max_dist_from_traj:
                reward_progress = 0.0
            else:
                self.furthest_reached_idx = max(self.furthest_reached_idx, best_index)
        if reward_progress > 0:
            self._last_progress_step = self.step_counter

        if best_index == self.cur_idx:
            start_bwd = max(0, self.cur_idx - self.nb_obs_backward + 1)
            end_bwd = self.cur_idx + 1
            best_index, min_dist = self._nearest_index_in_window(pos, start_bwd, end_bwd)
        self.cur_idx = best_index
        self._global_env_steps += 1

        _speed_reward_added = 0.0
        if self.prev_pos is None:
            self.prev_pos = pos.copy()
        pos_delta = pos - self.prev_pos
        self.prev_pos = pos.copy()
        pos_delta_xz = np.array([pos_delta[0], pos_delta[2]], dtype=np.float64)
        motion_norm = np.linalg.norm(pos_delta_xz)
        _computed_slip_deg: float | None = None
        if motion_norm > 1e-3:
            car_dir = pos_delta_xz / motion_norm
            if aim_yaw is not None:
                heading = np.array(
                    [math.sin(float(aim_yaw)), math.cos(float(aim_yaw))],
                    dtype=np.float64,
                )
                cross = heading[0] * car_dir[1] - heading[1] * car_dir[0]
                dot = float(np.dot(heading, car_dir))
                _computed_slip_deg = abs(math.degrees(math.atan2(cross, dot)))
        else:
            car_yaw = float(aim_yaw) if aim_yaw is not None else 0.0
            car_dir = np.array([np.sin(car_yaw), np.cos(car_yaw)], dtype=np.float64)
        next_idx = min(best_index + 1, self.datalen - 1)
        alignment_effective = 0.0
        if next_idx > best_index:
            track_vec = self.data[next_idx] - self.data[best_index]
            track_vec_xz = np.array([track_vec[0], track_vec[2]], dtype=np.float64)
            norm = np.linalg.norm(track_vec_xz)
            if norm > 0:
                track_dir = track_vec_xz / norm
                alignment = np.dot(track_dir, car_dir)
                alignment_effective = max(
                    getattr(self, "_speed_reward_alignment_floor", 0.0),
                    max(0.0, alignment),
                )

        reward = reward_progress

        if (
            _speed_kmh > SPEED_REWARD_MIN_KMH
            and min_dist <= self.max_dist_from_traj
            and self._speed_reward_weight > 0
        ):
            useful_speed_factor = (_speed_kmh / self._max_speed_kmh) * alignment_effective
            if useful_speed_factor > 0:
                exp = getattr(self, "_speed_reward_exponent", 1.0)
                _speed_reward_added = self._speed_reward_weight * (useful_speed_factor**exp)
                reward += _speed_reward_added

        _effective_slip = slip_angle_deg if slip_angle_deg is not None else _computed_slip_deg
        if _effective_slip is not None and _speed_kmh >= self._drift_threshold_kmh:
            _effective_slip = abs(float(_effective_slip))
            if self._drift_anneal_steps > 0:
                frac = min(1.0, self._global_env_steps / self._drift_anneal_steps)
                drift_w = self._drift_weight_start + frac * (
                    self._drift_weight_end - self._drift_weight_start
                )
            else:
                drift_w = self._drift_reward_weight
            if drift_w > 0:
                opt = self._drift_optimal_angle_deg
                sigma = max(1e-3, self._drift_sigma_deg)
                drift_bonus = drift_w * math.exp(
                    -((_effective_slip - opt) ** 2) / (2.0 * sigma * sigma)
                )
                reward += drift_bonus

        if (
            getattr(self, "_use_time_no_progress", False)
            and self.step_counter > self.min_nb_steps_before_failure
            and (self.step_counter - self._last_progress_step) >= self._max_no_progress_steps
        ):
            terminated = True
            self._term_reason = "no_progress_timeout"

        if (
            min_dist > self._max_track_width
            and self.step_counter > self.min_nb_steps_before_failure
        ):
            terminated = True
            self._term_reason = "off_track"

        if crashed:
            reward -= abs(self.crash_penalty)

        if self._constant_penalty > 0:
            reward -= self._constant_penalty

        if end_of_track:
            if self.datalen > 1 and self._total_traj_length > 0:
                dist_best = self._cumulative_dist[min(best_index, self.datalen - 1)]
                remaining_dist = max(0.0, float(self._total_traj_length - dist_best))
                reward += remaining_dist * (
                    self._progress_reward_full_lap / self._total_traj_length
                )
            reward += self._end_of_track_reward
            self._dbg_end_of_track_awarded = True

        if self._debug_reward:
            self._dbg_speeds_kmh.append(_speed_kmh)
            self._dbg_progress_rewards.append(reward_progress)
            self._dbg_speed_rewards.append(_speed_reward_added)
            self._dbg_barrier_penalties.append(0.0)

        reward = max(-self._reward_clip_floor, reward)
        reward = reward * self._reward_scale

        if terminated and self._term_reason is None:
            self._term_reason = "unknown_term"
        if end_of_track and self._term_reason is None:
            self._term_reason = "end_of_track"

        if getattr(self, "_use_time_no_progress", False):
            self.failure_counter = self.step_counter - self._last_progress_step
        else:
            self.failure_counter = 0
        race_progress = self.compute_race_progress()
        if race_progress > self.furthest_race_progress:
            self.furthest_race_progress = race_progress

        self.episode_reward += reward

        return reward, terminated, self.failure_counter, self.episode_reward

    def log_model_run(self, terminated: bool, end_of_track: bool) -> None:
        """Log episode outcome to console and optionally to Weights & Biases.

        Args:
            terminated: Whether the episode was terminated (stall, off-track, etc.).
            end_of_track: Whether the agent reached the end of the track.
        """
        if (terminated or end_of_track) and not self._logged_run_this_episode:
            self._logged_run_this_episode = True
            if end_of_track:
                self.furthest_race_progress = 1.0

            run_time_seconds = self.step_counter * getattr(
                self, "_time_step_duration", TIME_STEP_SECONDS
            )
            term_reason = getattr(self, "_term_reason", None)
            logger.info(
                "Total reward of the run: {:.4f} (Steps: {}, Time: {:.2f}s, reason: {})",
                self.episode_reward,
                self.step_counter,
                run_time_seconds,
                term_reason,
            )
            if self._debug_reward:
                self._log_episode_debug_summary()

            if self._use_wandb:
                self._episode_count = getattr(self, "_episode_count", 0) + 1
                log_dict: dict[str, float | int | str] = {
                    "run/reward": self.episode_reward,
                    "run/time_seconds": run_time_seconds,
                    "run/steps": self.step_counter,
                    "run/best_race_progress": self.furthest_race_progress,
                    "run/episode_count": self._episode_count,
                    "run/finish_time": run_time_seconds if end_of_track else 0.0,
                    "run/finished_track": int(end_of_track),
                }
                if term_reason is not None:
                    log_dict["run/term_reason"] = term_reason
                bt = getattr(self, "_dbg_barrier_touch_steps", 0)
                if bt > 0:
                    log_dict["run/barrier_touch_pct"] = 100 * bt / max(1, self.step_counter)
                wandb.log(log_dict)

    def compute_race_progress(self) -> float:
        """Current race progress as fraction of trajectory length (0.0 to 1.0)."""
        return self.cur_idx / len(self.data)

    def reset(self) -> None:
        """Reset reward state for a new episode."""
        self.cur_idx = 0
        self.prev_idx = 0
        self.furthest_reached_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self._last_progress_step = 0
        self._term_reason = None
        self.episode_reward = 0.0
        self.new_lap = False
        self.lap_cur_cooldown = cfg.LAP_COOLDOWN
        self.furthest_race_progress = 0.0
        self._logged_run_this_episode = False
        self.prev_pos = None
        if self._debug_reward:
            self._reset_debug_accumulators()
