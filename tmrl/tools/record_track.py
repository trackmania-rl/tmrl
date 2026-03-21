# standard library imports
import os
import pickle
import time
from typing import cast

# third-party imports
import keyboard
import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

import tmrl.config as cfg
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

# Must match TQC_GrabData plugin (20 floats). Default 19 misaligns the byte stream
# after the first frame and corrupts all subsequent data.
TQC_GRAB_NB_FLOATS = 20

MIN_POSITIONS_FOR_TRACK = 50
MIN_TRACK_LENGTH_M = (
    100.0  # Only stop when you've driven at least this far (avoids saving "start line only")
)


def _track_length_m(positions):
    if len(positions) < 2:
        return 0.0
    pts = np.asarray(positions)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.sum(diffs))


def _filter_origin_points(positions: np.ndarray) -> np.ndarray:
    """
    Remove only [0,0,0] glitch points (norm < 1.0).
    retrieve_data() already patches most of these, but the very first packets
    before _last_good_pos is set can slip through.
    No jump-distance filter — it caused false positives when the first buffered
    point was stale and all real positions got rejected as "jumps".
    """
    pts = cast(np.ndarray, np.asarray(positions, dtype=np.float64))
    if len(pts) < 2:
        return pts
    norms = np.linalg.norm(pts, axis=1)
    mask = norms >= 1.0
    filtered = cast(np.ndarray, pts[mask])
    if len(filtered) < 2:
        return pts
    return filtered


def record_track(path_track=cfg.TRACK_PATH_LEFT):
    positions = []
    client = TM2020OpenPlanetClient(port=9000, nb_floats=TQC_GRAB_NB_FLOATS)
    path = path_track

    is_recording = False
    while True:
        if keyboard.is_pressed("e"):
            logger.info("start recording")
            is_recording = True
        if is_recording:
            data = client.retrieve_data(
                sleep_if_empty=0.01
            )  # we need many points to build a smooth curve
            length_m = _track_length_m(positions)
            # Stop only when user presses Q (ignore game 'terminated')
            if keyboard.is_pressed("q"):
                if len(positions) < MIN_POSITIONS_FOR_TRACK:
                    logger.warning(
                        f"Too few positions ({len(positions)}). Drive the full track, "
                        f"then press Q. Need at least {MIN_POSITIONS_FOR_TRACK}."
                    )
                    continue
                if length_m < MIN_TRACK_LENGTH_M:
                    logger.warning(
                        f"Track too short ({length_m:.0f} m). "
                        f"Drive at least {MIN_TRACK_LENGTH_M:.0f} m, then press Q."
                    )
                    continue
                logger.info("Computing reward function checkpoints from captured positions...")
                logger.info(f"Initial number of captured positions: {len(positions)}")
                positions = np.array(positions)

                positions = _filter_origin_points(positions)

                length_after = _track_length_m(positions)
                logger.info(
                    f"After filtering: {len(positions)} positions, path length {length_after:.0f} m"
                )

                # We no longer need the old `line` + `while` loop.
                # `space_points` now does exact arc-length resampling on filtered,
                # chronologically ordered points.
                spaced_points = space_points(positions)
                smoothed_points = smooth_points(spaced_points)

                logger.info(
                    f"Final number of checkpoints of recorded track: {len(smoothed_points)}"
                )
                if len(smoothed_points) < 2:
                    logger.error("Not enough distinct points. Drive the full track, then press Q.")
                    continue
                with open(path, "wb") as f:
                    pickle.dump(smoothed_points, f)
                logger.info("All done")
                return
            else:
                positions.append([data[3], data[4], data[5]])
        else:
            time.sleep(0.05)  # waiting for user to press E


# Dense spacing (m) for track boundaries so geometry is preserved (arcs, 180° turns).
# Previously, `space_points` used `len(reward_file)`, which under-sampled
# and turned curves into sharp corners.
TRACK_BOUNDARY_SPACING_M = 0.25


def space_points(points, spacing_m=TRACK_BOUNDARY_SPACING_M):
    """
    Resample track boundary by arc length with dense spacing so curves stay smooth.
    Uses spacing_m (default 0.25 m).
    """
    if len(points) < 2:
        return points.copy()

    # Calculate exact distance between consecutive points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

    # Filter out duplicate points (distance == 0) to avoid CubicSpline errors
    mask = distances > 1e-6
    if not np.any(mask):
        return points.copy()

    valid_points = [points[0]]
    for i, m in enumerate(mask):
        if m:
            valid_points.append(points[i + 1])
    points = np.array(valid_points)

    if len(points) < 2:
        return points.copy()

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_distances = np.cumsum(distances)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)
    total_length = float(cumulative_distances[-1])
    if total_length <= 0:
        return points
    # Dense geometry by spacing_m; cap to avoid huge outputs if path length is wrong
    desired_num_points = max(2, round(total_length / spacing_m))
    desired_num_points = min(desired_num_points, 200_000)
    new_distances = np.linspace(0, total_length, desired_num_points, endpoint=True)
    cs_x = CubicSpline(cumulative_distances, x)
    cs_y = CubicSpline(cumulative_distances, y)
    cs_z = CubicSpline(cumulative_distances, z)
    new_x = cs_x(new_distances)
    new_y = cs_y(new_distances)
    new_z = cs_z(new_distances)
    return np.column_stack((new_x, new_y, new_z))


def interp_points_with_cubic_spline(sub_array, data_density=3):
    if len(sub_array) < 2:
        return sub_array.copy()
    original_x, original_y, original_z = sub_array.T

    # Calculate the new x-values based on data density (e.g., double the points)
    original_i = np.arange(0, int(data_density * len(original_x)), step=data_density)
    if len(original_i) < 2:
        return sub_array.copy()
    new_i = np.arange(0, int(data_density * len(original_x) - 1))

    print("Original i:", len(original_i))
    print("Original x:", len(original_x))
    print("Original y:", len(original_y))
    print("Original z:", len(original_z))
    print("new_i:", len(new_i))

    # Perform cubic spline interpolation for each vector (x, y, z)
    cs_x = CubicSpline(original_i, original_x)
    cs_y = CubicSpline(original_i, original_y)
    cs_z = CubicSpline(original_i, original_z)

    # Interpolate the y-values for the new_x values for each vector
    new_x_values = cs_x(new_i)
    new_y_values = cs_y(new_i)
    new_z_values = cs_z(new_i)

    # Combine the new x, y, and z values into a single NumPy array
    new_data = np.array([new_x_values, new_y_values, new_z_values])

    # Transpose the new_data array to have x, y, z as rows
    new_data = new_data.T

    return new_data


def smooth_points(points, sigma=3):
    """
    Smooths the given points using a Gaussian filter.

    Args:
        points (np.array): The array of points to be smoothed.
        sigma (int): The standard deviation for the Gaussian kernel.

    Returns:
        np.array: The smoothed array of points.
    """

    # Apply Gaussian filter for each dimension independently
    smoothed_x = gaussian_filter1d(points[:, 0], sigma)
    smoothed_y = gaussian_filter1d(points[:, 1], sigma)
    smoothed_z = gaussian_filter1d(points[:, 2], sigma)

    # Combine the smoothed coordinates back into a single array
    smoothed_points = np.column_stack((smoothed_x, smoothed_y, smoothed_z))

    return smoothed_points


def line(pt1, pt2, dist):
    """
    Creates a point between pt1 and pt2, at distance dist from pt1.

    If dist is too large, returns None and the remaining distance (> 0.0).
    Else, returns the point and 0.0 as remaining distance.
    """
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return (
            None,
            dist - norm,
        )  # we couldn't create a new point but we moved by a distance of norm
    else:
        vec_unit = vec / norm
        pt = pt1 + vec_unit * dist
        return pt, 0.0


if __name__ == "__main__":
    if not os.path.exists(cfg.REWARD_PATH):
        logger.debug(f" reward not found at path:{cfg.REWARD_PATH}")
    which_track = input("Choose which track do you want to record [left/right] [l/r]: ").lower()
    assert which_track in ("l", "r", "right", "left"), "Input must be left, right, l or r"
    print('Press "e" if you are ready do record track')
    if which_track in ("l", "left"):
        record_track(path_track=cfg.TRACK_PATH_LEFT)
    elif which_track in ("r", "right"):
        record_track(path_track=cfg.TRACK_PATH_RIGHT)
