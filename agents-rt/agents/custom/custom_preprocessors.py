import numpy as np


# OBSERVATION PREPROCESSING ==================================

def obs_preprocessor_tm_act_in_obs(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    obs = (obs[0], obs[1], obs[2], obs[3], obs[4])
    return obs


def obs_preprocessor_tm_lidar_act_in_obs(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    obs = (obs[0], np.ndarray.flatten(obs[1]), obs[2], obs[3])  # 2 actions
    # print(f"DEBUG (prepro): obs:{obs}")
    return obs


def obs_preprocessor_cognifly(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    # print(f"DEBUG: prepro obs: ---")
    # print(f"DEBUG: alt:{obs[0][0]:.2f}")
    # print(f"DEBUG: vel:{obs[1][0]:.2f}")
    # print(f"DEBUG: acc:{obs[2][0]:.2f}")
    # print(f"DEBUG: tar:{obs[3][0]:.2f}")
    # print(f"DEBUG: del:{obs[4][0]:.2f}")
    # print(f"DEBUG: a1:{obs[5][0]:.2f}")
    # print(f"DEBUG: a2:{obs[6][0]:.2f}")
    # print(f"DEBUG: a3:{obs[7][0]:.2f}")
    # print(f"DEBUG: a4:{obs[8][0]:.2f}")
    return obs


# SAMPLE PREPROCESSING =======================================
# these can be called when sampling from the replay memory, on the whole sample after observation preprocesing
# this is useful in particular for data augmentation
# be careful whatever you do here is consistent, because consistency after this will NOT be checked by CRC

def sample_preprocessor_tm_lidar_act_in_obs(last_obs, act, rew, new_obs, done):
    return last_obs, act, rew, new_obs, done
