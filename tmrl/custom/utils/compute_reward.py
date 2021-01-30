import numpy as np
import pickle


class RewardFunction:
    def __init__(self, reward_data_path, nb_obs_forward=10, nb_obs_backward=10):
        with open(reward_data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.cur_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward

    def compute_reward(self, pos):
        min_dist = np.inf
        index = self.cur_idx
        while True:
            dist = np.linalg.norm(pos-self.data[index])
            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            # stop condition
            if index == len(self.data) or temp == 0:
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
                if index == 0 or temp == 0:
                    break
        self.cur_idx = best_index
        return reward

    def reset(self):
        self.cur_idx = 0
