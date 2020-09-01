from random import randint
from agents.util import collate
import pickle
from pathlib import Path
import cv2
import numpy as np


class Memory:
    keep_reset_transitions: int = 0

    def __init__(self, memory_size, batchsize, device, remove_size=100, path_loc=r"D:\data", imgs_obs=4):
        self.device = device
        self.batchsize = batchsize
        self.capacity = memory_size
        self.remove_size = remove_size
        self.imgs_obs = imgs_obs

        self.last_observation = None
        self.last_action = None

        # init memory
        self.path = Path(path_loc)
        with open(self.path / 'data.pkl', 'rb') as f:
            self.data = pickle.load(f)

        if len(self) > self.capacity:
            # TODO: crop to memory_size
            print(f"WARNING: the dataset length ({len(self)}) is longer than memory_size ({self.capacity})")

    def append(self, r, done, info, obs, action):
        return self
        # if self.last_observation is not None:
        #
        #     if self.keep_reset_transitions:
        #         store = True
        #     else:
        #         # info["reset"] = True means the episode reset shouldn't be treated as a true terminal state
        #         store = not info.get('TimeLimit.truncated', False) and not info.get('reset', False)
        #
        #     if store:
        #         self.memory.append((self.last_observation, self.last_action, r, obs, done))
        # self.last_observation = obs
        # self.last_action = action
        #
        # # remove old entries if necessary (delete generously so we don't have to do it often)
        # if len(self.data) > self.capacity:
        #     del self.data[:self.capacity // self.remove_size + 1]
        # return self

    def __len__(self):
        return len(self.data[0])-self.imgs_obs-1

    def __getitem__(self, item):
        idx_last = item+self.imgs_obs-1
        idx_now = item+self.imgs_obs
        imgs = self.load_imgs(item)

        l=(
            (self.data[2][idx_last], imgs[:-1]),  # last_observation
            self.data[1][idx_last],  # last_action
            self.data[2][idx_now][0],  # r
            (self.data[2][idx_now], imgs[1:]),  # obs
            0.0,  # done
        )
        return l

    def load_imgs(self, item):
        res = []
        for i in range(item, item+self.imgs_obs+1):
            img = cv2.imread(str(self.path / (str(self.data[0][i]) + ".png")))
            res.append(np.moveaxis(img, -1, 0))
        return np.array(res)

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batchsize))

    def sample(self, indices=None):
        indices = self.sample_indices() if indices is None else indices
        batch = [self[idx] for idx in indices]
        batch = collate(batch, self.device)
        return batch


if __name__ == "__main__":
    dataset = Memory(memory_size=10, batchsize=3, device='cpu', remove_size=100, path_loc=r"C:\Users\Yann\Desktop\git\tmrl\data", imgs_obs=4)
    print('last_observation', dataset[0][0][0])
    print('last_observation', dataset[0][0][1].shape)  #
    print('last_action', dataset[0][1])
    print('r', dataset[0][2])
    print('obs', dataset[0][3][0])
    print('obs', dataset[0][3][1].shape)  #
    print('done', dataset[0][4])
    exit()