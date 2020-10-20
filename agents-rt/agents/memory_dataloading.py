from random import randint
import pickle
from pathlib import Path
import os

from agents.util import collate


class MemoryDataloading:
    def __init__(self, memory_size, batchsize, device, path_loc, remove_size=100, obs_preprocessor: callable = None, sample_preprocessor: callable = None):
        self.device = device
        self.batchsize = batchsize
        self.memory_size = memory_size
        self.remove_size = remove_size
        self.obs_preprocessor = obs_preprocessor
        self.sample_preprocessor = sample_preprocessor

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0

        # init memory
        self.path = Path(path_loc)
        if os.path.isfile(self.path / 'data.pkl'):
            with open(self.path / 'data.pkl', 'rb') as f:
                self.data = list(pickle.load(f))
                print(f"DEBUG: len data:{len(self.data)}")
                print(f"DEBUG: len data[0]:{len(self.data[0])}")
        else:
            print("INFO: no data found, initializing empty replay memory")
            self.data = []

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            print(f"WARNING: the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")

    def append(self, buffer):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_transition(self, item):
        """
        Returns: tuple (prev_obs, prev_act(prev_obs), rew(prev_obs, prev_act), obs, done)
        Do NOT apply observation preprocessing here, as it will be applied automatically after this
        """
        raise NotImplementedError

    def __getitem__(self, item):
        last_obs, new_act, rew, new_obs, done = self.get_transition(item)
        if self.obs_preprocessor is not None:
            last_obs = self.obs_preprocessor(last_obs)
            new_obs = self.obs_preprocessor(new_obs)
        if self.sample_preprocessor is not None:
            last_obs, new_act, rew, new_obs, done = self.sample_preprocessor(last_obs, new_act, rew, new_obs, done)
        return last_obs, new_act, rew, new_obs, done

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batchsize))

    def sample(self, indices=None):
        indices = self.sample_indices() if indices is None else indices
        batch = [self[idx] for idx in indices]
        batch = collate(batch, self.device)
        return batch


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
