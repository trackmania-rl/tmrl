"""
Build reward trajectory (centerline) from track left + track right.

Reference points = (track_left + track_right) / 2 -- center of track (SOTA: Gran Turismo Sophy,
F1Tenth). Produces a smooth axis, symmetric boundaries; agent discovers optimal racing line
on its own. Track boundaries defined by left/right; MAX_TRACK_WIDTH can be replaced by
checking distance to left/right.

Points spaced every --spacing-m meters (default 0.2 m = 20 cm).
Optionally --base-reward <path>: same number of points as in that reward file.

Default align=cross-section: each left point is paired with the closest right point in the
same "cross-section" of the track, because left/right recorded start-to-finish have
different lengths (inner vs outer barrier) -- naive left[i]+right[i] produces a bad center.

Usage:
  python scripts/build_centerline_reward.py [--spacing-m 0.2] [--smooth]
  python scripts/build_centerline_reward.py --debug-plot  # saves to output_files/debug/
  python scripts/build_centerline_reward.py --base-reward /path/to/reward_test-3.pkl

Input formats: .pkl (N,3) x,y,z or .csv (N,2) x,z.
Output: reward_<MAP>.pkl (N,3), ready for RewardFunction.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
from loguru import logger


def _timer(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    logger.debug("[{}] {:.3f}s", label, t1 - t0)
    return t1


def _ensure_3d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"Expected 2D array (N,2) or (N,3), got shape {points.shape}")
    if points.shape[1] == 2:
        return np.column_stack([points[:, 0], np.zeros(len(points)), points[:, 1]])
    if points.shape[1] != 3:
        raise ValueError(f"Expected 2 or 3 columns, got {points.shape[1]}")
    return points


MAX_TRACK_POINTS_LOAD = 100_000


def _cumulative_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(max(1, len(points)))
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    out = np.zeros(len(points))
    np.cumsum(diffs, out=out[1:])
    return out


def resample_by_arc_length(points: np.ndarray, num_points: int) -> np.ndarray:
    """Vectorized resample: np.searchsorted + vectorized lerp."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2 or num_points < 2:
        return points.copy()
    cum = _cumulative_distances(points)
    total = float(cum[-1])
    if total <= 0:
        return points.copy()
    s_values = np.linspace(0.0, total, num_points, endpoint=True)
    seg_idx = np.searchsorted(cum, s_values, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, n - 2)
    seg_start = cum[seg_idx]
    seg_end = cum[seg_idx + 1]
    seg_len = seg_end - seg_start
    seg_len[seg_len <= 0] = 1.0
    t = np.clip((s_values - seg_start) / seg_len, 0.0, 1.0)
    result = (1.0 - t[:, None]) * points[seg_idx] + t[:, None] * points[seg_idx + 1]
    return result


def resample_by_spacing_m(points: np.ndarray, spacing_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or spacing_m <= 0:
        return points.copy()
    cum = _cumulative_distances(points)
    total = float(cum[-1])
    if total <= 0:
        return points.copy()
    num_points = max(2, round(total / spacing_m))
    return resample_by_arc_length(points, num_points)


def _pair_cross_section(left_pts: np.ndarray, right_pts: np.ndarray) -> np.ndarray:
    """
    For each left point find the closest right point that lies at a similar
    arc-length fraction (same "cross-section" of the track).

    Pure nearest-neighbor matching fails when the track loops back near itself:
    a left point on one section gets paired with a right point from a
    geometrically close but topologically different section, producing diagonal
    cuts.  Constraining the search to a +/- WINDOW_FRAC window in normalised
    arc-length prevents this.
    """
    cum_left = _cumulative_distances(left_pts)
    cum_right = _cumulative_distances(right_pts)
    total_left = cum_left[-1] if cum_left[-1] > 0 else 1.0
    total_right = cum_right[-1] if cum_right[-1] > 0 else 1.0
    frac_left = cum_left / total_left
    frac_right = cum_right / total_right

    window_frac = 0.10

    right_paired = np.empty_like(left_pts)
    for i in range(len(left_pts)):
        f = frac_left[i]
        lo = int(np.searchsorted(frac_right, f - window_frac))
        hi = int(np.searchsorted(frac_right, f + window_frac))
        if lo >= hi:
            idx = int(np.argmin(np.abs(frac_right - f)))
            right_paired[i] = right_pts[idx]
        else:
            segment = right_pts[lo:hi]
            dists = np.linalg.norm(segment - left_pts[i], axis=1)
            right_paired[i] = segment[np.argmin(dists)]
    return right_paired


def smooth_centerline(points: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points.copy()
    from scipy.ndimage import gaussian_filter1d

    sigma = max(0.5, min(sigma, len(points) / 4))
    smoothed = np.zeros_like(points)
    for col in range(points.shape[1]):
        smoothed[:, col] = gaussian_filter1d(points[:, col], sigma=sigma, mode="nearest")
    return smoothed


def load_track(path: str) -> np.ndarray:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        points = _ensure_3d(np.asarray(data))
    elif ext == ".csv":
        data = np.loadtxt(path, delimiter=",")
        points = _ensure_3d(data)
    else:
        raise ValueError(f"Unsupported format: {path} (use .pkl or .csv)")
    if len(points) > MAX_TRACK_POINTS_LOAD:
        logger.info("Resampling {} -> {} points ({})", len(points), MAX_TRACK_POINTS_LOAD, path)
        points = resample_by_arc_length(points, MAX_TRACK_POINTS_LOAD)
    return points


def build_centerline(
    left: np.ndarray,
    right: np.ndarray,
    target_points: int | None = None,
    spacing_m: float | None = None,
    smooth: bool = False,
    align: str = "cross-section",
) -> np.ndarray:
    left = _ensure_3d(np.asarray(left, dtype=np.float64))
    right = _ensure_3d(np.asarray(right, dtype=np.float64))
    n_l, n_r = len(left), len(right)
    if n_l < 2 or n_r < 2:
        raise ValueError("Both left and right need at least 2 points.")
    if target_points is None:
        target_points = min(n_l, n_r)
    target_points = max(2, target_points)

    if align == "cross-section":
        left_r = resample_by_arc_length(left, target_points)
        right_paired = _pair_cross_section(left_r, right)
        center = (left_r + right_paired) * 0.5
    else:
        left_r = resample_by_arc_length(left, target_points) if n_l != target_points else left
        right_r = resample_by_arc_length(right, target_points) if n_r != target_points else right
        center = (left_r + right_r) * 0.5

    if smooth:
        center = smooth_centerline(center)
    if spacing_m is not None and spacing_m > 0:
        center = resample_by_spacing_m(center, spacing_m)
    return center


def _save_debug_plot(
    left: np.ndarray,
    right: np.ndarray,
    center: np.ndarray,
    path: str,
    align: str,
    center_raw: np.ndarray | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not found, skipping --debug-plot")
        return
    max_plot = 20_000

    def _ds(arr):
        if len(arr) <= max_plot:
            return arr
        idx = np.linspace(0, len(arr) - 1, max_plot, dtype=int)
        return arr[idx]

    left_ds, right_ds, center_ds = _ds(left), _ds(right), _ds(center)
    cr = _ds(center_raw) if center_raw is not None else None

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        left_ds[:, 0], left_ds[:, 1], left_ds[:, 2], "b-", alpha=0.7, label="left", linewidth=0.8
    )
    ax.plot(
        right_ds[:, 0],
        right_ds[:, 1],
        right_ds[:, 2],
        "C1-",
        alpha=0.7,
        label="right",
        linewidth=0.8,
    )
    if cr is not None:
        ax.plot(cr[:, 0], cr[:, 1], cr[:, 2], "g-", alpha=0.9, linewidth=1.5, label="center (raw)")
    ax.plot(
        center_ds[:, 0],
        center_ds[:, 1],
        center_ds[:, 2],
        "c--",
        alpha=0.9,
        linewidth=1.2,
        label="center (final)",
        zorder=4,
    )
    ax.scatter(left[0, 0], left[0, 1], left[0, 2], c="b", s=80, marker="o", zorder=5)
    ax.scatter(right[0, 0], right[0, 1], right[0, 2], c="C1", s=80, marker="o", zorder=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z")
    ax.set_title(f"Centerline debug (align={align})")
    with contextlib.suppress(Exception):
        ax.set_box_aspect([1, 0.01, 1])
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    path_2d = path.rsplit(".", 1)
    path_2d = (
        (path_2d[0] + "_topdown." + path_2d[1]) if len(path_2d) == 2 else (path + "_topdown.png")
    )
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.plot(left_ds[:, 0], left_ds[:, 2], "b-", alpha=0.7, label="left", linewidth=0.8)
    ax2.plot(right_ds[:, 0], right_ds[:, 2], "C1-", alpha=0.7, label="right", linewidth=0.8)
    if cr is not None:
        ax2.plot(cr[:, 0], cr[:, 2], "g-", alpha=0.9, linewidth=1.5, label="center (raw)")
    ax2.plot(
        center_ds[:, 0],
        center_ds[:, 2],
        "c--",
        alpha=0.9,
        linewidth=1.2,
        label="center (final)",
        zorder=4,
    )
    ax2.plot(left[0, 0], left[0, 2], "bo", markersize=8)
    ax2.plot(right[0, 0], right[0, 2], "o", color="C1", markersize=8)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_aspect("equal")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Top-down (X-Z). Raw = before smooth/spacing.")
    fig2.savefig(path_2d, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Debug plot saved: {} and {}", path, path_2d)


@dataclass
class CenterlineArgs:
    """Build reward trajectory (centerline) from track_left + track_right."""

    left: str | None = None
    """Path to left track boundary (.pkl or .csv). Default: from TmrlData config."""

    right: str | None = None
    """Path to right track boundary (.pkl or .csv). Default: from TmrlData config."""

    out: str | None = None
    """Output path. Default: TmrlData reward path or reward_<MAP>.pkl."""

    spacing_m: float = 0.2
    """Point spacing in meters (default: 0.2 = 20 cm)."""

    base_reward: str | None = None
    """Match point count from this existing reward file instead of using --spacing-m."""

    align: Literal["cross-section", "index"] = "cross-section"
    """How to pair left/right points: cross-section (nearest in same track section) or index."""

    smooth: bool = True
    """Apply Gaussian smoothing to the centerline."""

    debug_plot: str | None = None
    """Save 3D + top-down debug plot. Relative/omitted path → output_files/debug/."""

    dry_run: bool = False
    """Only print stats, do not write file."""


def main() -> int:
    t_total = time.perf_counter()
    args = tyro.cli(CenterlineArgs)

    if args.left is None or args.right is None:
        try:
            import tmrl.config as cfg

            args.left = args.left or cfg.TRACK_PATH_LEFT
            args.right = args.right or cfg.TRACK_PATH_RIGHT
            default_out = cfg.REWARD_PATH
            map_name = cfg.MAP_NAME
        except Exception as e:
            logger.error("--left and --right required when TmrlData config unavailable: {}", e)
            return 1
    else:
        default_out = None
        map_name = "unknown"

    t0 = time.perf_counter()
    try:
        left = load_track(args.left)
        right = load_track(args.right)
    except (FileNotFoundError, ValueError) as e:
        logger.error("{}", e)
        return 1
    t0 = _timer("Load tracks", t0)

    n_l, n_r = len(left), len(right)
    spacing_m = None
    target_points = None
    if args.base_reward:
        base_path = os.path.abspath(args.base_reward)
        if not os.path.isfile(base_path):
            logger.error("base-reward file not found: {}", base_path)
            return 1
        with open(base_path, "rb") as f:
            base_data = pickle.load(f)
        target_points = len(np.asarray(base_data))
        logger.info("Base reward: {} ({} points)", base_path, target_points)
    else:
        spacing_m = float(args.spacing_m)
        if spacing_m <= 0:
            logger.error("--spacing-m must be positive")
            return 1
        target_points = max(n_l, n_r, 2)

    center = build_centerline(
        left,
        right,
        target_points=target_points,
        spacing_m=spacing_m,
        smooth=args.smooth,
        align=args.align,
    )
    t0 = _timer("Build centerline", t0)

    if args.debug_plot:
        import tmrl.config as cfg

        debug_dir = str(cfg.DEBUG_FOLDER)
        os.makedirs(debug_dir, exist_ok=True)
        if os.path.isabs(args.debug_plot):
            debug_path = args.debug_plot
        else:
            debug_path = os.path.join(debug_dir, args.debug_plot or "centerline_debug.png")
        center_raw = build_centerline(
            left,
            right,
            target_points=target_points,
            spacing_m=None,
            smooth=False,
            align=args.align,
        )
        _save_debug_plot(
            left, right, center, os.path.abspath(debug_path), args.align, center_raw=center_raw
        )
        t0 = _timer("Debug plot", t0)

    cum = _cumulative_distances(center)
    total_len = float(cum[-1]) if len(center) >= 2 else 0.0
    avg_seg = total_len / (len(center) - 1) if len(center) > 1 else 0.0

    logger.info("Track left:  {} ({} pts)", args.left, n_l)
    logger.info("Track right: {} ({} pts)", args.right, n_r)
    mode = f"spacing {args.spacing_m} m" if not args.base_reward else "match base-reward count"
    logger.info(
        "Centerline:  {} pts ({}){}", len(center), mode, " (smoothed)" if args.smooth else ""
    )
    logger.info("  total length: {:.2f} m, avg segment: {:.4f} m", total_len, avg_seg)

    if args.dry_run:
        logger.info("(dry-run: not writing file)")
        return 0

    out_path = args.out or default_out
    if not out_path:
        out_path = f"reward_{map_name}.pkl"
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(center, f, protocol=pickle.HIGHEST_PROTOCOL)
    t0 = _timer("Save pkl", t0)
    logger.info("Saved to: {}", out_path)

    logger.debug("[TOTAL] {:.3f}s", time.perf_counter() - t_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
