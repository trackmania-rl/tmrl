# third-party imports
import numpy as np
import cv2

# local imports
from tmrl.memory_dataloading import MemoryDataloading, TrajMemoryDataloading

# LOCAL BUFFER COMPRESSION ==============================


def get_local_buffer_sample(prev_act, obs, rew, done, info):
    """
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition (but not influencing the unaugmented observation in real-time envs)
        obs, rew, done, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the dataloading memory
    the user must define both this function and the append() method of the dataloading memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1][-1])  # speed and most recent image only
    rew_mod = np.float32(rew)
    done_mod = done
    return prev_act, obs_mod, rew_mod, done_mod, info


def get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, done, info):
    """
    Sample compressor for MemoryTM2020
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, done, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the dataloading memory
    the user must define both this function and the append() method of the dataloading memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    prev_act_mod = prev_act
    compressed_img = cv2.imencode('.PNG', np.moveaxis(obs[3][-1], 0, -1))
    obs_mod = (obs[0], obs[1], obs[2], compressed_img)  # speed, gear, rpm, last image
    rew_mod = np.float32(rew)
    done_mod = done
    info_mod = info
    return prev_act_mod, obs_mod, rew_mod, done_mod, info_mod


# FUNCTIONS ====================================================


def last_true_in_list(li):
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


def replace_hist_before_done(hist, done_idx_in_hist):
    last_idx = len(hist) - 1
    assert done_idx_in_hist <= last_idx, f"DEBUG: done_idx_in_hist:{done_idx_in_hist}, last_idx:{last_idx}"
    if 0 <= done_idx_in_hist < last_idx:
        for i in reversed(range(len(hist))):
            if i <= done_idx_in_hist:
                hist[i] = hist[i + 1]


# MEMORY DATALOADING ===========================================


class MemoryTMNF(MemoryDataloading):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 nb_steps=1,
                 use_dataloader=False,
                 num_workers=0,
                 pin_memory=False,
                 obs_preprocessor: callable = None,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         use_dataloader=use_dataloader,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         obs_preprocessor=obs_preprocessor,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):  # TODO
        return self

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):  # TODO
        pass
        # return last_obs, new_act, rew, new_obs, done


class MemoryTMNFLidar(MemoryTMNF):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_dones = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_done_idx = last_true_in_list(last_dones)  # last occurrence of True
        assert last_done_idx is None or last_dones[last_done_idx], f"DEBUG: last_done_idx:{last_done_idx}"
        last_infos = self.data[6][idx_now - self.min_samples:idx_now]
        last_ignored_dones = ["__no_done" in i for i in last_infos]
        last_ignored_done_idx = last_true_in_list(last_ignored_dones)  # last occurrence of True
        assert last_ignored_done_idx is None or last_ignored_dones[last_ignored_done_idx] and not last_dones[last_ignored_done_idx], f"DEBUG: last_ignored_done_idx:{last_ignored_done_idx}, last_ignored_dones:{last_ignored_dones}, last_dones:{last_dones}"
        if last_ignored_done_idx is not None:
            last_done_idx = last_ignored_done_idx  # FIXME: might not work in extreme cases where a done is ignored right after another done

        if last_done_idx is not None:
            replace_hist_before_done(hist=new_act_buf, done_idx_in_hist=last_done_idx - self.start_acts_offset - 1)
            replace_hist_before_done(hist=last_act_buf, done_idx_in_hist=last_done_idx - self.start_acts_offset)
            replace_hist_before_done(hist=imgs_new_obs, done_idx_in_hist=last_done_idx - self.start_imgs_offset - 1)
            replace_hist_before_done(hist=imgs_last_obs, done_idx_in_hist=last_done_idx - self.start_imgs_offset)

        last_obs = (self.data[2][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], imgs_new_obs, *new_act_buf)
        done = self.data[4][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, done, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, done, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # lidar
        d4 = [b[3] for b in buffer.memory]  # dones
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[4] for b in buffer.memory]  # infos

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]

        return self


class TrajMemoryTMNF(TrajMemoryDataloading):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 traj_len=1,
                 nb_steps=1,
                 use_dataloader=False,
                 num_workers=0,
                 pin_memory=False,
                 obs_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.traj_len = traj_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.min_samples += self.traj_len - 1
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         use_dataloader=use_dataloader,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         obs_preprocessor=obs_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):  # TODO
        return self

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_trajectory(self, item):  # TODO
        pass
        # return last_obs, new_act, rew, new_obs, done


class TrajMemoryTMNFLidar(TrajMemoryTMNF):
    def get_trajectory(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        idx_now = item + self.min_samples
        all_acts = self.load_acts_traj(item)
        # new_act_buf = acts[1:]
        all_imgs = self.load_imgs_traj(item)

        # rew = np.float32(self.data[5][idx_now])

        rew_traj = [np.float32(self.data[5][idx_now + i]) for i in range(self.traj_len)]

        # new_act = self.data[1][idx_now]

        # new_obs = (self.data[2][idx_now], imgs[1:], *new_act_buf)

        augm_obs_traj = [(self.data[2][idx_now + i], all_imgs[1 + i:self.imgs_obs + i + 1], *all_acts[1 + i:self.act_buf_len + i + 1]) for i in range(self.traj_len)]

        # done = self.data[4][idx_now]

        done_traj = [self.data[4][idx_now + i] for i in range(self.traj_len)]

        # info = self.data[6][idx_now]

        info_traj = [self.data[6][idx_now + i] for i in range(self.traj_len)]

        return augm_obs_traj, rew_traj, done_traj, info_traj

    def load_imgs_traj(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + self.traj_len)]
        return np.stack(res)

    def load_acts_traj(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + self.traj_len)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, done, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # lidar
        d4 = [b[3] for b in buffer.memory]  # dones
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[4] for b in buffer.memory]  # infos

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]

        return self


class MemoryTM2020(MemoryDataloading):  # TODO: reset transitions
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 nb_steps=1,
                 use_dataloader=False,
                 num_workers=0,
                 pin_memory=False,
                 obs_preprocessor: callable = None,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         use_dataloader=use_dataloader,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         obs_preprocessor=obs_preprocessor,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, done, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """
        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0
        d0 = [(first_data_idx + i) % self.memory_size for i, _ in enumerate(buffer.memory)]  # indexes  # FIXME: check that this works
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # gear
        d4 = [b[1][2] for b in buffer.memory]  # rpm
        for bi, di in zip(buffer.memory, d0):
            cv2.imwrite(str(self.path / (str(di) + '.png')), cv2.imdecode(np.array(bi[1][3][1]), cv2.IMREAD_UNCHANGED))
        d5 = [b[3] for b in buffer.memory]  # dones
        d6 = [b[2] for b in buffer.memory]  # rewards
        d7 = [b[4] for b in buffer.memory]  # infos

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
        return self

    def __len__(self):
        if len(self.data) < self.min_samples + 1:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        imgs = self.load_imgs(item)
        acts = self.load_acts(item)

        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        last_obs = (self.data[2][idx_last], self.data[3][idx_last], self.data[4][idx_last], imgs[:-1], *last_act_buf)
        rew = np.float32(self.data[6][idx_now])
        new_act = np.array(self.data[1][idx_now], dtype=np.float32)
        new_obs = (self.data[2][idx_now], self.data[3][idx_now], self.data[4][idx_now], imgs[1:], *new_act_buf)
        done = self.data[5][idx_now]
        info = self.data[7][idx_now]
        return last_obs, new_act, rew, new_obs, done, info

    def load_imgs(self, item):
        res = []
        for i in range(item + self.start_imgs_offset, item + self.start_imgs_offset + self.imgs_obs + 1):
            img_path = str(self.path / (str(self.data[0][i]) + ".png"))
            img = cv2.imread(img_path)
            res.append(np.moveaxis(img, -1, 0))
        return np.array(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res


class MemoryTM2020RAM(MemoryTM2020):
    """
    Same as MemoryTM2020 but the full buffer is in RAM to avoid dataloading latencies
    """
    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, done, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """
        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0
        d0 = [(first_data_idx + i) % self.memory_size for i, _ in enumerate(buffer.memory)]  # indexes  # FIXME: check that this works
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # gear
        d4 = [b[1][2] for b in buffer.memory]  # rpm
        d5 = [b[3] for b in buffer.memory]  # dones
        d6 = [b[2] for b in buffer.memory]  # rewards
        d7 = [b[4] for b in buffer.memory]  # infos
        d8 = [np.moveaxis(cv2.imdecode(np.array(b[1][3][1]), cv2.IMREAD_UNCHANGED), -1, 0) for b in buffer.memory]

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
        return self

    def load_imgs(self, item):
        res = self.data[8][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.array(res)
