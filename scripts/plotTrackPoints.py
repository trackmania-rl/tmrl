"""Plot left and right track boundaries as interactive 3D scatter.

Loads track_<MAP>_left.pkl and track_<MAP>_right.pkl from config (or given paths)
and displays both with Plotly. Optional Gaussian smoothing.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tyro
from scipy.ndimage import gaussian_filter1d


@dataclass
class PlotTrackArgs:
    """Plot left/right track boundaries from TmrlData or given paths."""

    left_path: str | None = None
    """Path to left boundary .pkl (default: from config TRACK_PATH_LEFT)."""

    right_path: str | None = None
    """Path to right boundary .pkl (default: from config TRACK_PATH_RIGHT)."""

    smooth_sigma: float = 0.0
    """If > 0, apply Gaussian smoothing (sigma) to both tracks before plotting."""

    renderer: str = "browser"
    """Plotly renderer: 'browser', 'notebook', etc."""


def load_track_points(path: str) -> np.ndarray:
    """Load track boundary from a pickle file.

    Args:
        path: Path to .pkl with (N, 3) points.

    Returns:
        Array of shape (N, 3).
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Track file not found: {path}")
    with open(path_obj, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data, dtype=np.float64)


def smooth_points(points: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth each coordinate with a 1D Gaussian along the polyline.

    Args:
        points: Array of shape (N, 3).
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        Smoothed array of shape (N, 3).
    """
    if sigma <= 0 or len(points) < 3:
        return points
    smoothed = np.column_stack(
        [
            gaussian_filter1d(points[:, 0], sigma, mode="nearest"),
            gaussian_filter1d(points[:, 1], sigma, mode="nearest"),
            gaussian_filter1d(points[:, 2], sigma, mode="nearest"),
        ]
    )
    return smoothed


def main() -> int:
    """Load left/right tracks and show interactive 3D plot."""
    args = tyro.cli(PlotTrackArgs)

    if args.left_path is None or args.right_path is None:
        try:
            import tmrl.config as cfg

            left_path = args.left_path or cfg.TRACK_PATH_LEFT
            right_path = args.right_path or cfg.TRACK_PATH_RIGHT
        except Exception as exc:
            print(
                "Cannot load config (run from project root or set PYTHONPATH). "
                f"Provide --left-path and --right-path. Error: {exc}",
                file=sys.stderr,
            )
            return 1
    else:
        left_path = args.left_path
        right_path = args.right_path

    try:
        left_track = load_track_points(left_path)
        right_track = load_track_points(right_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.smooth_sigma > 0:
        left_track = smooth_points(left_track, args.smooth_sigma)
        right_track = smooth_points(right_track, args.smooth_sigma)

    plotly_fig = go.Figure()
    plotly_fig.add_trace(
        go.Scatter3d(
            x=left_track[:, 0],
            y=left_track[:, 1],
            z=left_track[:, 2],
            mode="markers",
            marker={"size": 5, "color": "blue", "opacity": 0.8},
            name="Left track",
        )
    )
    plotly_fig.add_trace(
        go.Scatter3d(
            x=right_track[:, 0],
            y=right_track[:, 1],
            z=right_track[:, 2],
            mode="markers",
            marker={"size": 5, "color": "red", "opacity": 0.8},
            name="Right track",
        )
    )
    plotly_fig.update_layout(
        title="Left and right track boundaries (interactive 3D)",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectratio": {"x": 1, "y": 0.01, "z": 1},
        },
    )
    plotly_fig.show(renderer=args.renderer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
