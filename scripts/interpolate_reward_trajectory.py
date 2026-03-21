"""
Interpolate a reward trajectory pkl to add more points (e.g. 10x) along the track.
Finer spacing gives a more granular progress signal (e.g. on difficult turns)
without changing total reward scale: progress is still (distance_gained * 100 / total_length).

Usage:
  python scripts/interpolate_reward_trajectory.py --input /path/to/reward_<MAP_NAME>.pkl
      [--factor 10] [--out path] [--dry-run]

Example (TmrlData on Windows WSL):
  python scripts/interpolate_reward_trajectory.py
      --input /mnt/c/Users/szulc/TmrlData/reward/reward_test-3.pkl --factor 10
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass

import numpy as np
import tyro
from loguru import logger


def _cumulative_distances(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length along the polyline (length at each point index)."""
    if len(points) < 2:
        return np.zeros(max(1, len(points)))
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    out = np.zeros(len(points))
    np.cumsum(diffs, out=out[1:])
    return out


def interpolate_trajectory(points: np.ndarray, factor: int) -> np.ndarray:
    """
    Return a new point array with roughly factor*len(points) points, uniformly spaced by arc length.
    Linear interpolation along the original polyline; total arc length is preserved.
    """
    n = len(points)
    if n < 2:
        return points.copy()
    cum = _cumulative_distances(points)
    total = float(cum[-1])
    if total <= 0:
        return points.copy()

    num_new = max(2, int(factor * n))
    s_values = np.linspace(0.0, total, num_new, endpoint=True)

    new_pts = []
    j = 0
    for s in s_values:
        if s >= total:
            new_pts.append(points[-1].copy())
            continue
        while j + 1 < n and cum[j + 1] < s:
            j += 1
        if j + 1 >= n:
            new_pts.append(points[-1].copy())
            continue
        seg_start = cum[j]
        seg_end = cum[j + 1]
        t = 0.0 if seg_end <= seg_start else (s - seg_start) / (seg_end - seg_start)
        t = np.clip(t, 0.0, 1.0)
        pt = (1.0 - t) * points[j] + t * points[j + 1]
        new_pts.append(pt)
    return np.array(new_pts, dtype=points.dtype)


@dataclass
class InterpolateArgs:
    """Interpolate reward trajectory pkl to add more points (e.g. 10x)."""

    input: str
    """Path to reward_<MAP_NAME>.pkl (e.g. TmrlData/reward/reward_test-3.pkl)."""

    factor: int = 10
    """Target number of points = factor * original length."""

    out: str | None = None
    """Output path (default: overwrite input)."""

    dry_run: bool = False
    """Only print stats, do not write file."""


def main() -> int:
    args = tyro.cli(InterpolateArgs)

    in_path = args.input
    try:
        with open(in_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logger.error("File not found: {}", in_path)
        return 1
    except Exception as e:
        logger.error("Error loading {}: {}", in_path, e)
        return 1

    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 3:
        logger.error("Expected (N, 3) array, got shape {}", getattr(data, "shape", "?"))
        return 1

    n_old = len(data)
    cum_old = _cumulative_distances(data)
    total_old = float(cum_old[-1]) if n_old >= 2 else 0.0
    avg_dist_old = total_old / (n_old - 1) if n_old > 1 else 0.0

    new_data = interpolate_trajectory(data, args.factor)
    n_new = len(new_data)
    cum_new = _cumulative_distances(new_data)
    total_new = float(cum_new[-1]) if n_new >= 2 else 0.0
    avg_dist_new = total_new / (n_new - 1) if n_new > 1 else 0.0

    length_ratio = total_new / total_old if total_old > 0 else 1.0

    logger.info(
        "Before: points={}, total_length={:.2f}, avg_segment={:.4f}", n_old, total_old, avg_dist_old
    )
    logger.info(
        "After:  points={}, total_length={:.2f}, avg_segment={:.4f}", n_new, total_new, avg_dist_new
    )
    logger.info("Length ratio (after/before): {:.6f} (should be ~1.0)", length_ratio)
    if abs(length_ratio - 1.0) > 0.01:
        logger.warning("Total length changed by >1% -- check interpolation.")
    else:
        logger.info("Reward scale unchanged (total lap raw reward still ~100).")

    if args.dry_run:
        logger.info("(dry-run: not writing file)")
        return 0

    out_path = args.out if args.out else in_path
    try:
        with open(out_path, "wb") as f:
            pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error("Error writing {}: {}", out_path, e)
        return 1
    logger.info("Saved to: {}", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
