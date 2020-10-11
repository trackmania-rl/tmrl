from random import randint
from agents.util import collate
import pickle
from pathlib import Path
import cv2
import numpy as np
import os


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

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0

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

        last_act = np.array(self.data[4][idx_last], dtype=np.float32)
        last_obs = (np.array([self.data[1][idx_last], ], dtype=np.float32), imgs[:-1], last_act) if self.act_in_obs else (np.array([self.data[1][idx_last], ], dtype=np.float32), imgs[:-1])
        rew = np.float32(self.data[6][idx_now])
        new_act = np.array(self.data[4][idx_now], dtype=np.float32)
        new_obs = (np.array([self.data[1][idx_now], ], dtype=np.float32), imgs[1:], new_act) if self.act_in_obs else (np.array([self.data[1][idx_now], ], dtype=np.float32), imgs[1:])
        done = np.float32(self.data[5][idx_now])

        if self.obs_preprocessor is not None:
            last_obs = self.obs_preprocessor(last_obs)
            new_obs = self.obs_preprocessor(new_obs)
        return last_obs, new_act, rew, new_obs, done

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

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0

        # self.last_observation = None
        # self.last_action = None

        # init memory
        self.path = Path(path_loc)
        if os.path.isfile(self.path / 'data.pkl'):
            with open(self.path / 'data.pkl', 'rb') as f:
                self.data = list(pickle.load(f))
                print(f"DEBUG: len data:{len(self.data)}")
                print(f"DEBUG: len data[0]:{len(self.data[0])}")
        else:
            print("DEBUG: no data found, initializing empty replay memory")
            self.data = []

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            print(f"WARNING: the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")

    def append(self, buffer):
        return self

    def __len__(self):
        if len(self.data) < self.imgs_obs + 1:
            return 0
        return len(self.data[0])-self.imgs_obs-1

    def __getitem__(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        So we load 5 images from here...
        """
        idx_last = item + self.imgs_obs-1
        idx_now = item + self.imgs_obs
        imgs = self.load_imgs(item)

        last_act = self.data[1][idx_last]
        last_obs = (self.data[2][idx_last], imgs[:-1], last_act) if self.act_in_obs else (self.data[2][idx_last], imgs[:-1])
        rew = np.float32(self.data[4][idx_now])
        new_act = self.data[1][idx_now]
        new_obs = (self.data[2][idx_now], imgs[1:], new_act) if self.act_in_obs else (self.data[2][idx_now], imgs[1:])
        done = np.float32(0.0)

        if self.obs_preprocessor is not None:
            last_obs = self.obs_preprocessor(last_obs)
            new_obs = self.obs_preprocessor(new_obs)
        return last_obs, new_act, rew, new_obs, done

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
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        """
        # print(f"DEBUG: getitem ---")
        idx_last = item + self.imgs_obs-1
        idx_now = item + self.imgs_obs
        # print(f"DEBUG: item:{item}, idx_last:{idx_last}, idx_now:{idx_now}")
        imgs = self.load_imgs(item)
        # print(f"DEBUG: imgs:{imgs}")
        last_act = self.data[1][idx_last]
        # print(f"DEBUG WARNING!!!: last_act:{last_act}")
        last_obs = (self.data[2][idx_last], imgs[:-1], last_act) if self.act_in_obs else (self.data[2][idx_last], imgs[:-1])
        # print(f"DEBUG: last_obs:{last_obs}")
        rew = np.float32(self.data[5][idx_now])
        # print(f"DEBUG: rew:{rew}")
        new_act = self.data[1][idx_now]
        # print(f"DEBUG WARNING!!!: new_act:{new_act}")
        new_obs = (self.data[2][idx_now], imgs[1:], new_act) if self.act_in_obs else (self.data[2][idx_now], imgs[1:])
        # print(f"DEBUG: new_obs:{new_obs}")
        done = np.float32(0.0)

        if self.obs_preprocessor is not None:
            last_obs = self.obs_preprocessor(last_obs)
            new_obs = self.obs_preprocessor(new_obs)

        print(last_obs, new_act, rew, new_obs)
        return last_obs, new_act, rew, new_obs, done

    def load_imgs(self, item):
        res = []
        for i in range(item, item+self.imgs_obs+1):
            img = self.data[3][i]
            res.append(img)
        return np.array(res)

    def append(self, buffer):
        """
        buffer is a list of (obs, act, rew, done, info)
        """
        # print(f"DEBUG: appending buffer to replay memory")
        # print(f"DEBUG: len(self.data):{len(self.data)}, len(buffer):{len(buffer)}")
        if len(buffer) > 0:
            # print(f"DEBUG: self.data[0][-1]:{self.data[0][-1]}")
            # print(f"DEBUG: len(self.data[0]):{len(self.data[0])}")
            # print(f"DEBUG: type(self.data[0]):{type(self.data[0])}, type(self.data[0][0]):{type(self.data[0][0])}")
            # print(f"DEBUG: type(self.data[1]):{type(self.data[1])}, type(self.data[1][0]):{type(self.data[1][0])}, type(buffer.memory[0][1]):{type(buffer.memory[0][1])}")
            # print(f"DEBUG: type(self.data[2]):{type(self.data[2])}, type(self.data[2][0]):{type(self.data[2][0])}, type(buffer.memory[0][0][0]):{type(buffer.memory[0][0][0])}")
            # print(f"DEBUG: type(self.data[3]):{type(self.data[3])}, type(self.data[3][0]):{type(self.data[3][0])}, type(buffer.memory[0][0][1]):{type(buffer.memory[0][0][1])}")
            # print(f"DEBUG: type(self.data[4]):{type(self.data[4])}, type(self.data[4][0]):{type(self.data[4][0])}, type(buffer.memory[0][3]):{type(buffer.memory[0][3])}")
            # print(f"DEBUG: type(self.data[5]):{type(self.data[5])}, type(self.data[5][0]):{type(self.data[5][0])}, type(buffer.memory[0][2]):{type(buffer.memory[0][2])}")

            first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

            d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]
            d1 = [b[1] for b in buffer.memory]
            d2 = [b[0][0] for b in buffer.memory]
            d3 = [b[0][1] for b in buffer.memory]
            d4 = [b[3] for b in buffer.memory]
            d5 = [b[2] for b in buffer.memory]

            if self.__len__() > 0:
                self.data[0] += d0
                self.data[1] += d1
                self.data[2] += d2
                self.data[3] += d3
                self.data[4] += d4
                self.data[5] += d5
            else:
                self.data.append(d0)
                self.data.append(d1)
                self.data.append(d2)
                self.data.append(d3)
                self.data.append(d4)
                self.data.append(d5)

            to_trim = self.__len__() - self.memory_size
            if to_trim > 0:
                print(f"DEBUG: trimming {to_trim} elements")
                self.data[0] = self.data[0][to_trim:]
                self.data[1] = self.data[1][to_trim:]
                self.data[2] = self.data[2][to_trim:]
                self.data[3] = self.data[3][to_trim:]
                self.data[4] = self.data[4][to_trim:]
                self.data[5] = self.data[5][to_trim:]

            self.stat_train_return = buffer.stat_train_return
            self.stat_test_return = buffer.stat_test_return
            # print(f"DEBUG: self.stat_train_return:{self.stat_train_return}, self.stat_test_return:{self.stat_test_return}")
        # else:
        #     print(f"DEBUG: empty buffer")
        return self


def load_and_print_pickle_file(path=r"C:\Users\Yann\Desktop\git\tmrl\data\data.pkl"):  # r"D:\data2020"
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"nb samples: {len(data[0])}")
    for i, d in enumerate(data):
        print(f"[{i}][0]: {d[0]}")
    print("full data:")
    for i, d in enumerate(data):
        print(f"[{i}]: {d}")


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
