from abc import ABC, abstractmethod
from random import randint, random
import pickle
from pathlib import Path
import os
import zlib
import numpy as np
from tmrl.util import collate, data_to_cuda
from torch.utils.data import Dataset, Sampler, DataLoader
import torch


def check_samples_crc(original_po, original_a, original_o, original_r, original_d, rebuilt_po, rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, device):
    try:
        original_po_t = tuple(torch.tensor(x) for x in original_po) if original_po is not None else None
        original_a_t = torch.tensor(original_a)
        original_o_t = tuple(torch.tensor(x) for x in original_o)
        original_r_t = torch.tensor(np.float32(original_r))
        original_d_t = torch.tensor(np.float32(original_d))
        assert original_po_t is None or str(original_po_t) == str(rebuilt_po), f"previous observations don't match:\noriginal:\n{original_po_t}\n!= rebuilt:\n{rebuilt_po}"
        assert str(original_a_t) == str(rebuilt_a), f"actions don't match:\noriginal:\n{original_a_t}\n!= rebuilt:\n{rebuilt_a}"
        assert str(original_o_t) == str(rebuilt_o), f"observations don't match:\noriginal:\n{original_o_t}\n!= rebuilt:\n{rebuilt_o}"
        assert str(original_r_t) == str(rebuilt_r), f"rewards don't match:\noriginal:\n{original_r_t}\n!= rebuilt:\n{rebuilt_r}"
        assert str(original_d_t) == str(rebuilt_d), f"dones don't match:\noriginal:\n{original_d_t}\n!= rebuilt:\n{rebuilt_d}"
    except Exception as e:
        print(f"Caught exception: {e}")
        print(f"previous observations:\noriginal:\n{original_po}\n?= rebuilt:\n{rebuilt_po}")
        print(f"actions:\noriginal:\n{original_a}\n?= rebuilt:\n{rebuilt_a}")
        print(f"observations:\noriginal:\n{original_o}\n?= rebuilt:\n{rebuilt_o}")
        print(f"rewards:\noriginal:\n{original_r}\n?= rebuilt:\n{rebuilt_r}")
        print(f"dones:\noriginal:\n{original_d}\n?= rebuilt:\n{rebuilt_d}")
        print(f"Device: {device}")
        exit()
    print("DEBUG: CRC check passed.")


def check_samples_crc_traj(original_o, original_r, original_d, rebuilt_o, rebuilt_r, rebuilt_d):
    assert str(original_o) == str(rebuilt_o), f"observations don't match:\noriginal:\n{original_o}\n!= rebuilt:\n{rebuilt_o}"
    assert str(original_r) == str(rebuilt_r), f"rewards don't match:\noriginal:\n{original_r}\n!= rebuilt:\n{rebuilt_r}"
    assert str(original_d) == str(rebuilt_d), f"dones don't match:\noriginal:\n{original_d}\n!= rebuilt:\n{rebuilt_d}"
    original_crc = zlib.crc32(str.encode(str((original_o, original_r, original_d))))
    crc = zlib.crc32(str.encode(str((rebuilt_o, rebuilt_r, rebuilt_d))))
    assert crc == original_crc, f"CRC failed: new crc:{crc} != old crc:{original_crc}.\nEither the custom pipeline is corrupted, or crc_debug is False in the rollout worker.\noriginal sample:\n{(original_o, original_r, original_d)}\n!= rebuilt sample:\n{(rebuilt_o, rebuilt_r, rebuilt_d)}"
    print("DEBUG: CRC check passed.")


class MemoryBatchSampler(Sampler):
    """
    Iterator over nb_steps randomly sampled batches of size batchsize
    """
    def __init__(self, data_source, nb_steps, batchsize):
        super().__init__(data_source)
        self._dataset = data_source
        self._nb_steps = nb_steps
        self._batchsize = batchsize

    def __len__(self):
        return self._nb_steps

    def __iter__(self):
        for _ in range(self._nb_steps):
            yield (int(len(self._dataset) * random()) - 1 for _ in range(self._batchsize))  # faster than randint


class MemoryDataloading(ABC):  # FIXME: should be an instance of Dataset but partial doesn't work with Dataset
    """
    To sample from a MemoryDataloading, use either the iterator OR the get_dataloader() method
    e.g. either:
        for batch in myMemoryDataloading:  # this uses single worker dataloading (directly collates on device, unlike pytorch)
            operations on batch ...
    OR:
        for batch in myMemoryDataloading.get_dataloader():  # this uses pytorch dataloading (does NOT collate on device)
            operations on batch ...
    When sequences is True, the Memory expects tensors with the sequence length as first dimension
    """
    def __init__(self,
                 memory_size,
                 batchsize,
                 path_loc="",
                 nb_steps=1,
                 use_dataloader=False,
                 num_workers=0,
                 pin_memory=False,
                 remove_size=100,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu",
                 sequences=True,
                 collate_fn=None):

        print(f"DEBUG: MemoryDataloading use_dataloader:{use_dataloader}")
        print(f"DEBUG: MemoryDataloading pin_memory:{pin_memory}")

        self.nb_steps = nb_steps
        self.use_dataloader = use_dataloader
        self.device = device
        self.batchsize = batchsize
        self.memory_size = memory_size
        self.remove_size = remove_size
        self.sample_preprocessor = sample_preprocessor
        self.crc_debug = crc_debug
        self.sequences = sequences
        self.collate_fn = collate_fn if collate_fn is not None else collate

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0
        self.stat_test_steps = 0
        self.stat_train_steps = 0

        # init memory
        self.path = Path(path_loc)
        print(f"DEBUG: MemoryDataloading self.path:{self.path}")
        if os.path.isfile(self.path / 'data.pkl'):
            with open(self.path / 'data.pkl', 'rb') as f:
                self.data = pickle.load(f)
                print(f"DEBUG: data found, loaded in self.data")
        else:
            print("INFO: no data found, initializing self.data to None")
            self.data = None

        # init dataloader
        self._batch_sampler = MemoryBatchSampler(data_source=self, nb_steps=nb_steps, batchsize=batchsize)
        self._dataloader = DataLoader(dataset=self, batch_sampler=self._batch_sampler, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate)

    def __iter__(self):
        if not self.use_dataloader:
            for _ in range(self.nb_steps):
                yield self.sample()
        else:
            for batch in self._dataloader:
                if self.device == 'cuda':
                    batch = data_to_cuda(batch)  # FIXME: probably doesn't work with multiprocessing
                yield batch

    @abstractmethod
    def append_buffer(self, buffer):
        """
        CAUTION: don't forget to append the info dictionary if you want to use CRC debugging.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def get_transition(self, item):
        """
        Returns:
            if not sequence: tuple (prev_obs, prev_act(prev_obs), rew(prev_obs, prev_act), obs, done, info)
            else: sequence (tuple) of tuple (prev_obs, prev_act(prev_obs), rew(prev_obs, prev_act), obs, done, info)
        info is required in each sample for CRC debugging. The 'crc' key is what is important when using this feature
        Do NOT apply observation preprocessing here, as it will be applied automatically after this
        If the sequence option is set, this expects the first dimension of all tensors contained in both observations to have the first dimension as the sequence length.
        """
        raise NotImplementedError

    def append(self, buffer):
        if len(buffer) > 0:
            self.stat_train_return = buffer.stat_train_return
            self.stat_test_return = buffer.stat_test_return
            self.stat_train_steps = buffer.stat_train_steps
            self.stat_test_steps = buffer.stat_test_steps
            self.append_buffer(buffer)

    def getitem_no_sequences(self, item):
        prev_obs, new_act, rew, new_obs, done, info = self.get_transition(item)
        if self.crc_debug:
            po, a, o, r, d = info['crc_sample']
            check_samples_crc(po, a, o, r, d, prev_obs, new_act, new_obs, rew, done, self.device)
        # if self.obs_preprocessor is not None:
        #     prev_obs = self.obs_preprocessor(prev_obs)
        #     new_obs = self.obs_preprocessor(new_obs)
        if self.sample_preprocessor is not None:
            prev_obs, new_act, rew, new_obs, done = self.sample_preprocessor(prev_obs, new_act, rew, new_obs, done)
        # done = np.float32(done)  # we don't want bool tensors
        return prev_obs, new_act, rew, new_obs, done

    def getitem_sequences(self, item):
        """
        Here, a dimension exists for sequences on obs
        """
        prev_obs, new_act, rew, new_obs, done, info = self.get_transition(item)

        if self.crc_debug:
            po, a, o, r, d = info['crc_sample']
            last_prev_obs = tuple(x[-1] for x in prev_obs)
            last_new_obs = tuple(x[-1] for x in new_obs)
            check_samples_crc(po, a, o, r, d, last_prev_obs, new_act, last_new_obs, rew, done, self.device)
        # if self.obs_preprocessor is not None:
        #     prev_obs = self.obs_preprocessor(prev_obs)
        #     new_obs = self.obs_preprocessor(new_obs)
        if self.sample_preprocessor is not None:
            prev_obs, new_act, rew, new_obs, done = self.sample_preprocessor(prev_obs, new_act, rew, new_obs, done)
        # done = np.float32(done)  # we don't want bool tensors
        return prev_obs, new_act, rew, new_obs, done

    def __getitem__(self, item):
        if item < 0:
            item = self.__len__() + item
        return self.getitem_no_sequences(item) if not self.sequences else self.getitem_sequences(item)

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batchsize))

    def sample(self, indices=None):
        # print("DEBUG: called sample()")
        indices = self.sample_indices() if indices is None else indices
        batch = [self[idx] for idx in indices]
        batch = self.collate_fn(batch, self.device)  # collate batch dimension
        # if self.sequences:
        #     batch = self.collate_fn(batch, self.device)  # collate sequence dimension
        #     po, a, r, o, d = batch
        #     batch = po, a[-1], r[-1], o, d[-1]  # drop useless sequences (keep only for observations so they can be processed by RNNs)
        return batch


class TrajMemoryDataloading(MemoryDataloading, ABC):
    """
    This is specifically for the DC/AC algorithm
    """
    def __init__(self,
                 memory_size,
                 batchsize,
                 path_loc,
                 traj_len=1,
                 nb_steps=1,
                 use_dataloader=False,
                 num_workers: callable = 0,
                 pin_memory=False,
                 remove_size=100,
                 obs_preprocessor: callable = None,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        super().__init__(memory_size=memory_size,
                         batchsize=batchsize,
                         path_loc=path_loc,
                         nb_steps=nb_steps,
                         use_dataloader=use_dataloader,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         remove_size=remove_size,
                         obs_preprocessor=obs_preprocessor,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)
        self.traj_len = traj_len  # this should be used in __len__() and in get_trajectory()

    def get_transition(self, item):
        assert False, f"Invalid method for this class, implement get_trajectory instead."

    @abstractmethod
    def get_trajectory(self, item):
        """
        Returns: tuple (augm_obs_traj:list, rew_traj:list, done_traj:list, info_traj:list)
        each trajectory must be of length self.traj_len
        info_traj is required for CRC debugging. The 'crc' key is what is important when using this feature.
        Do NOT apply observation preprocessing here, as it will be applied automatically after this
        """
        raise NotImplementedError

    def __getitem__(self, item):
        augm_obs_traj, rew_traj, done_traj, info_traj = self.get_trajectory(item)
        assert len(augm_obs_traj) == len(rew_traj) == len(done_traj) == self.traj_len, f"all trajectories must be of length self.traj_len:{self.traj_len}."
        if self.crc_debug:
            for i in range(len(augm_obs_traj)):
                _, _, o, r, d = info_traj[i]['crc_sample']
                new_obs, rew, done = augm_obs_traj[i], rew_traj[i], done_traj[i]
                check_samples_crc_traj(o, r, d, new_obs, rew, done)
        if self.obs_preprocessor is not None:
            augm_obs_traj = [self.obs_preprocessor(obs) for obs in augm_obs_traj]
        if self.sample_preprocessor is not None:
            raise NotImplementedError("Sample preprocessing is not supported for trajectories.")
        done_traj = [np.float32(done) for done in done_traj]  # we don't want bool tensors
        return augm_obs_traj, rew_traj, done_traj


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
