import numpy as np
import cv2
from agents.memory_dataloading import MemoryDataloading


# LOCAL BUFFERS ================================================

def get_local_buffer_sample(prev_act, obs, rew, done, info):
    """
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, done, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the dataloading memory
    the user must define both this function and the append() method of the dataloading memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1][-1])  # speed and most recent image only
    rew_mod = np.float32(rew)
    return prev_act, obs_mod, rew_mod, done, info


# MEMORY DATALOADING ===========================================

class MemoryTMNF(MemoryDataloading):
    def __init__(self, memory_size, batchsize, device, remove_size=100, path_loc=r"D:\data", imgs_obs=4, act_in_obs=True, obs_preprocessor: callable = None):
        self.imgs_obs = imgs_obs
        self.act_in_obs = act_in_obs
        super().__init__(memory_size, batchsize, device, path_loc, remove_size, obs_preprocessor)

    def append(self, buffer):
        return self

    def __len__(self):
        if len(self.data) < self.imgs_obs + 1:
            return 0
        return len(self.data[0])-self.imgs_obs-1

    def get_transition(self, item):
        idx_last = item + self.imgs_obs - 1
        idx_now = item + self.imgs_obs
        imgs = self.load_imgs(item)

        last_act = self.data[1][idx_last]
        last_obs = (self.data[2][idx_last], imgs[:-1], last_act) if self.act_in_obs else (
        self.data[2][idx_last], imgs[:-1])
        rew = np.float32(self.data[4][idx_now])
        new_act = self.data[1][idx_now]
        new_obs = (self.data[2][idx_now], imgs[1:], new_act) if self.act_in_obs else (self.data[2][idx_now], imgs[1:])
        done = np.float32(0.0)
        return last_obs, new_act, rew, new_obs, done

    def load_imgs(self, item):
        res = []
        for i in range(item, item+self.imgs_obs+1):
            img = cv2.imread(str(self.path / (str(self.data[0][i]) + ".png")))
            img = img.astype('float32')
            res.append(img)
        return np.array(res)


class MemoryTMNFLidar(MemoryTMNF):
    def get_transition(self, item):
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
        done = np.float32(self.data[4][idx_now])
        return last_obs, new_act, rew, new_obs, done

    def load_imgs(self, item):
        res = []
        for i in range(item, item+self.imgs_obs+1):
            img = self.data[3][i]
            # print(f"DEBUG:i:{i}/{self.__len__()}, img:{img}")
            res.append(img)
        return np.stack(res)

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

            d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
            d1 = [b[0] for b in buffer.memory]  # actions
            d2 = [b[1][0] for b in buffer.memory]  # speeds
            d3 = [b[1][1] for b in buffer.memory]  # images
            d4 = [b[3] for b in buffer.memory]  # dones
            d5 = [b[2] for b in buffer.memory]  # rewards

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


class MemoryTM2020(MemoryDataloading):  # TODO
    def __init__(self, memory_size, batchsize, device, remove_size=100, path_loc=r"D:\data", imgs_obs=4,
                 act_in_obs=True, obs_preprocessor: callable = None):
        super().__init__(self, memory_size, batchsize, device, path_loc, remove_size, obs_preprocessor)
        self.imgs_obs = imgs_obs
        self.act_in_obs = act_in_obs

    def append(self, buffer):
        return self

    def __len__(self):
        return len(self.data[0])-self.imgs_obs-1

    def get_transition(self, item):
        idx_last = item + self.imgs_obs - 1
        idx_now = item + self.imgs_obs
        imgs = self.load_imgs(item)

        last_act = np.array(self.data[4][idx_last], dtype=np.float32)
        last_obs = (
        np.array([self.data[1][idx_last], ], dtype=np.float32), imgs[:-1], last_act) if self.act_in_obs else (
        np.array([self.data[1][idx_last], ], dtype=np.float32), imgs[:-1])
        rew = np.float32(self.data[6][idx_now])
        new_act = np.array(self.data[4][idx_now], dtype=np.float32)
        new_obs = (np.array([self.data[1][idx_now], ], dtype=np.float32), imgs[1:], new_act) if self.act_in_obs else (
        np.array([self.data[1][idx_now], ], dtype=np.float32), imgs[1:])
        done = np.float32(self.data[5][idx_now])
        return last_obs, new_act, rew, new_obs, done

    def load_imgs(self, item):
        res = []
        for i in range(item, item+self.imgs_obs+1):
            img = cv2.imread(str(self.path / (str(self.data[0][i]) + ".png")))
            img = img[20:-30, :]
            img = img.astype('float32') / 255.0
            res.append(np.moveaxis(img, -1, 0))
        return np.array(res)