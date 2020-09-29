from random import randint
from agents.util import collate
import pickle
from pathlib import Path
import cv2
import numpy as np


class MemoryTM2020:
    keep_reset_transitions: int = 0

    def __init__(self, memory_size, batchsize, device, remove_size=100, path_loc=r"D:\data", imgs_obs=4, act_in_obs=True, obs_preprocessor: callable = None):
        self.device = device
        self.batchsize = batchsize
        self.memory_size = memory_size
        self.remove_size = remove_size
        self.imgs_obs = imgs_obs
        self.act_in_obs = act_in_obs
        self.obs_preprocessor = obs_preprocessor

        self.last_observation = None
        self.last_action = None

        # init memory
        self.path = Path(path_loc)
        with open(self.path / 'data.pkl', 'rb') as f:
            self.data = pickle.load(f)

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            print(f"WARNING: the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")

    def append(self, r, done, info, obs, action):
        return self  # TODO

    def __len__(self):
        return len(self.data[0])-self.imgs_obs-1

    def __getitem__(self, item):
        idx_last = item+self.imgs_obs-1
        idx_now = item+self.imgs_obs
        imgs = self.load_imgs(item)

        last_obs = (np.array([self.data[1][idx_last], ], dtype=np.float32), imgs[:-1], np.array(self.data[4][idx_last], dtype=np.float32)) if self.act_in_obs else (np.array([self.data[1][idx_last], ], dtype=np.float32), imgs[:-1])
        last_act = np.array(self.data[4][idx_last], dtype=np.float32)
        rew = np.float32(self.data[6][idx_now])
        new_obs = (np.array([self.data[1][idx_now], ], dtype=np.float32), imgs[1:], np.array(self.data[4][idx_now], dtype=np.float32)) if self.act_in_obs else (np.array([self.data[1][idx_now], ], dtype=np.float32), imgs[1:])
        done = np.float32(self.data[5][idx_now])

        if self.obs_preprocessor is not None:
            last_obs = self.obs_preprocessor(last_obs)
            new_obs = self.obs_preprocessor(new_obs)
        return last_obs, last_act, rew, new_obs, done

    def load_imgs(self, item):
        res = []
        for i in range(item, item+self.imgs_obs+1):
            img = cv2.imread(str(self.path / (str(self.data[0][i]) + ".png")))
            img = img[20:-30, :]
            img = img.astype('float32') / 255.0
            res.append(np.moveaxis(img, -1, 0))
        return np.array(res)

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batchsize))

    def sample(self, indices=None):
        indices = self.sample_indices() if indices is None else indices
        batch = [self[idx] for idx in indices]
        batch = collate(batch, self.device)
        return batch


class MemoryTMNF:
    keep_reset_transitions: int = 0

    def __init__(self, memory_size, batchsize, device, remove_size=100, path_loc=r"D:\data", imgs_obs=4, act_in_obs=True, obs_preprocessor: callable = None):
        self.device = device
        self.batchsize = batchsize
        self.memory_size = memory_size
        self.remove_size = remove_size
        self.imgs_obs = imgs_obs
        self.act_in_obs = act_in_obs
        self.obs_preprocessor = obs_preprocessor

        self.last_observation = None
        self.last_action = None

        # init memory
        self.path = Path(path_loc)
        with open(self.path / 'data.pkl', 'rb') as f:
            self.data = pickle.load(f)
            print(f"DEBUG: len data:{len(self.data)}")
            print(f"DEBUG: len data[0]:{len(self.data[0])}")

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            print(f"WARNING: the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")

    def append(self, buffer):
        return self

    def __len__(self):
        return len(self.data[0])-self.imgs_obs-1

    def __getitem__(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        So we load 5 images from here...
        """
        idx_last = item + self.imgs_obs-1
        idx_now = item + self.imgs_obs
        imgs = self.load_imgs(item)

        last_obs = (self.data[2][idx_last], imgs[:-1], self.data[1][idx_last]) if self.act_in_obs else (self.data[2][idx_last], imgs[:-1])
        last_act = self.data[1][idx_last]
        rew = np.float32(self.data[4][idx_now])
        new_obs = (self.data[2][idx_now], imgs[1:], self.data[1][idx_now]) if self.act_in_obs else (self.data[2][idx_now], imgs[1:])
        done = np.float32(0.0)

        if self.obs_preprocessor is not None:
            last_obs = self.obs_preprocessor(last_obs)
            new_obs = self.obs_preprocessor(new_obs)
        return last_obs, last_act, rew, new_obs, done

    def load_imgs(self, item):
        res = []
        for i in range(item, item+self.imgs_obs+1):
            img = cv2.imread(str(self.path / (str(self.data[0][i]) + ".png")))
            img = img.astype('float32')
            res.append(img)
        return np.array(res)

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batchsize))

    def sample(self, indices=None):
        indices = self.sample_indices() if indices is None else indices
        batch = [self[idx] for idx in indices]
        # print(f"DEBUG: batch:{batch}")
        batch = collate(batch, self.device)
        return batch


class MemoryTMNFLidar(MemoryTMNF):
    def __getitem__(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        So we load 5 images from here...
        """
        idx_last = item + self.imgs_obs-1
        idx_now = item + self.imgs_obs
        imgs = self.load_imgs(item)

        last_obs = (self.data[2][idx_last], imgs[:-1], self.data[1][idx_last]) if self.act_in_obs else (self.data[2][idx_last], imgs[:-1])
        last_act = self.data[1][idx_last]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], imgs[1:], self.data[1][idx_now]) if self.act_in_obs else (self.data[2][idx_now], imgs[1:])
        done = np.float32(0.0)

        if self.obs_preprocessor is not None:
            last_obs = self.obs_preprocessor(last_obs)
            new_obs = self.obs_preprocessor(new_obs)
        return last_obs, last_act, rew, new_obs, done

    def load_imgs(self, item):
        res = []
        for i in range(item, item+self.imgs_obs+1):
            img = self.data[3][i]
            res.append(img)
        return np.array(res)

    def append(self, buffer):
        # TODO
        print(f"DEBUG: appending buffer to replay memory")
        return self


def load_and_print_pickle_file(path=r"C:\Users\Yann\Desktop\git\tmrl\data\data.pkl"):  # r"D:\data2020"
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"nb samples: {len(data[0])}")
    for i, d in enumerate(data):
        print(f"[{i}][0]: {d[0]}")


if __name__ == "__main__":
    load_and_print_pickle_file()

    # dataset = MemoryTM2020(memory_size=10, batchsize=3, device='cpu', remove_size=100, path_loc=r"D:\data2020", imgs_obs=4)
    # print('last_observation vitesse ', dataset[0][0][0])
    # print('last_observation img', dataset[0][0][1].shape, np.min(dataset[0][0][1]), np.max(dataset[0][0][1]))
    # print('last_action', dataset[0][1])
    # print('r', dataset[0][2])
    # print('obs', dataset[0][3][0])
    # print('obs', dataset[0][3][1].shape)  #
    # print('done', dataset[0][4])
    # exit()

    # print('last_observation', dataset[0][0][0])
    # print('last_observation', dataset[0][0][1].shape)  #
    # print('last_action', dataset[0][1])
    # print('r', dataset[0][2])
    # print('obs', dataset[0][3][0])
    # print('obs', dataset[0][3][1].shape)  #
    # print('done', dataset[0][4])
    # exit()
