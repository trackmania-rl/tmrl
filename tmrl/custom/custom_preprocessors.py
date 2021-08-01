# third-party imports
import numpy as np
import logging
# OBSERVATION PREPROCESSING ==================================


def obs_preprocessor_tm_act_in_obs(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    obs = (obs[0], obs[1], obs[2], obs[3], *obs[4:])  # >= 1 action
    # logging.debug(f" (not same as old): preprocessed obs:{obs}")
    return obs


def obs_preprocessor_tm_lidar_act_in_obs(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    obs = (obs[0], np.ndarray.flatten(obs[1]), *obs[2:])  # >= 1  action
    # logging.debug(f" (not same as old): preprocessed obs:{obs}")
    return obs


def obs_preprocessor_cognifly(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    return obs


# SAMPLE PREPROCESSING =======================================
# these can be called when sampling from the replay memory, on the whole sample after observation preprocesing
# this is useful in particular for data augmentation
# be careful whatever you do here is consistent, because consistency after this will NOT be checked by CRC


def sample_preprocessor_tm_lidar_act_in_obs(last_obs, act, rew, new_obs, done):
    return last_obs, act, rew, new_obs, done
