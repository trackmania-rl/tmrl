# third-party imports
import numpy as np
import logging
# OBSERVATION PREPROCESSING ==================================


def obs_preprocessor_tm_act_in_obs(obs):
    """
    Preprocessor for TM2020 with images, converting RGB images to grayscale
    """
    images = obs[3]
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_images = np.dot(images[..., :3], rgb_weights)
    obs = (obs[0], obs[1], obs[2], grayscale_images, *obs[4:])  # >= 1 action
    return obs


def obs_preprocessor_tm_lidar_act_in_obs(obs):
    """
    Preprocessor for the LIDAR environment, flattening LIDARs
    """
    obs = (obs[0], np.ndarray.flatten(obs[1]), *obs[2:])  # >= 1  action
    return obs


def obs_preprocessor_tm_lidar_progress_act_in_obs(obs):
    """
    Preprocessor for the LIDAR environment, flattening LIDARs
    """
    obs = (obs[0], obs[1], np.ndarray.flatten(obs[2]), *obs[3:])  # >= 1  action
    return obs


# SAMPLE PREPROCESSING =======================================
# these can be called when sampling from the replay memory, on the whole sample after observation preprocesing
# this is useful in particular for data augmentation
# be careful whatever you do here is consistent, because consistency after this will NOT be checked by CRC


def sample_preprocessor_tm_lidar_act_in_obs(last_obs, act, rew, new_obs, done):
    return last_obs, act, rew, new_obs, done
