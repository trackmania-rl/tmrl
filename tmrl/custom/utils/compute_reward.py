# standard library imports
import os
import pickle

# third-party imports
import numpy as np
import logging

class RewardFunction:
    def __init__(self, reward_data_path, nb_obs_forward=10, nb_obs_backward=10, nb_zero_rew_before_early_done=10, min_nb_steps_before_early_done=int(3.5 * 20)):
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)
        self.cur_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_early_done = nb_zero_rew_before_early_done
        self.min_nb_steps_before_early_done = min_nb_steps_before_early_done
        self.step_counter = 0
        self.early_done_counter = 0

    def compute_reward(self, pos):
        done = False
        self.step_counter += 1
        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = 0
        while True:
            dist = np.linalg.norm(pos - self.data[index])
            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            # stop condition
            if index >= len(self.data) or temp <= 0:
                break
        reward = best_index - self.cur_idx
        if best_index == self.cur_idx:  # if the best index didn't change, we rewind (more Markovian reward)
            min_dist = np.inf
            index = self.cur_idx
            while True:
                dist = np.linalg.norm(pos - self.data[index])
                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                # stop condition
                if index <= 0 or temp <= 0:
                    break
            if self.step_counter > self.min_nb_steps_before_early_done:
                self.early_done_counter += 1
                if self.early_done_counter > self.nb_zero_rew_before_early_done:
                    done = True
        else:
            self.early_done_counter = 0
        self.cur_idx = best_index
        return reward, done

    def reset(self):
        self.cur_idx = 0
        self.step_counter = 0
        self.early_done_counter = 0
