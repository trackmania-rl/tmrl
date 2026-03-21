"""Sample compressors for local buffer storage.

These functions compress observations before storing them in local buffers
for network transmission, reducing bandwidth requirements.

Buffer sample order: (prev_act, obs, rew, terminated, truncated, info)
where prev_act is the action that yielded obs (i.e. prev_obs -> prev_act -> obs).
"""

import numpy as np

LIDAR_RECENT_WINDOW = 19


def get_local_buffer_sample_lidar(prev_act, obs, rew, terminated, truncated, info):
    """Compress for LIDAR interface: keep only speed and most recent LIDAR."""
    obs_mod = (obs[0], obs[1][-LIDAR_RECENT_WINDOW:])
    return prev_act, obs_mod, np.float32(rew), terminated, truncated, info


def get_local_buffer_sample_lidar_progress(prev_act, obs, rew, terminated, truncated, info):
    """Compress for LIDAR+progress interface: keep speed, progress, recent LIDAR."""
    obs_mod = (obs[0], obs[1], obs[2][-LIDAR_RECENT_WINDOW:])
    return prev_act, obs_mod, np.float32(rew), terminated, truncated, info


def get_local_buffer_sample_lidar_progress_images(prev_act, obs, rew, terminated, truncated, info):
    """Compress for LIDAR+images interface: cast reward to float32."""
    return prev_act, obs, np.float32(rew), terminated, truncated, info


def get_local_buffer_sample_mobilenet(prev_act, obs, rew, terminated, truncated, info):
    """Compress for MobileNet interface: cast reward to float32."""
    return prev_act, obs, np.float32(rew), terminated, truncated, info


def get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, terminated, truncated, info):
    """Compress for full TM2020 image interface: quantize images to uint8."""
    obs_mod = (obs[0], obs[1], obs[2], (obs[3][-1] * 256.0).astype(np.uint8))
    return prev_act, obs_mod, rew, terminated, truncated, info
