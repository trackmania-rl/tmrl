# standard library imports
import os
import pickle
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from random import randint
import time
import logging

# third-party imports
import numpy as np


__docformat__ = "google"


def check_samples_crc(original_po, original_a, original_o, original_r, original_d, original_t, rebuilt_po, rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t, debug_ts, debug_ts_res, epsilon=0.0):
    if epsilon == 0.0:  # strict check
        assert original_po is None or str(original_po) == str(rebuilt_po), f"previous observations don't match:\noriginal:\n{str(original_po)}\n!= rebuilt:\n{str(rebuilt_po)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        assert str(original_a) == str(rebuilt_a), f"actions don't match:\noriginal:\n{str(original_a)}\n!= rebuilt:\n{str(rebuilt_a)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        assert str(original_o) == str(rebuilt_o), f"observations don't match:\noriginal:\n{str(original_o)}\n!= rebuilt:\n{str(rebuilt_o)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        assert str(original_r) == str(rebuilt_r), f"rewards don't match:\noriginal:\n{str(original_r)}\n!= rebuilt:\n{str(rebuilt_r)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        assert str(original_d) == str(rebuilt_d), f"terminated don't match:\noriginal:\n{str(original_d)}\n!= rebuilt:\n{str(rebuilt_d)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        assert str(original_t) == str(rebuilt_t), f"truncated don't match:\noriginal:\n{str(original_t)}\n!= rebuilt:\n{str(rebuilt_t)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        original_crc = zlib.crc32(str.encode(str((original_a, original_o, original_r, original_d, original_t))))
        crc = zlib.crc32(str.encode(str((rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t))))
        assert crc == original_crc, f"CRC failed: new crc:{crc} != old crc:{original_crc}.\nEither the custom pipeline is corrupted, or crc_debug is False in the rollout worker.\noriginal sample:\n{(original_a, original_o, original_r, original_d)}\n!= rebuilt sample:\n{(rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
    else:  # check if values are close
        if isinstance(original_o, np.ndarray):
            assert original_po is None or np.allclose(original_po, rebuilt_po, atol=epsilon), f"previous observations don't match:\noriginal:\n{str(original_po)}\n!= rebuilt:\n{str(rebuilt_po)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
            assert np.allclose(original_o, rebuilt_o, atol=epsilon), f"observations don't match:\noriginal:\n{str(original_o)}\n!= rebuilt:\n{str(rebuilt_o)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        else:
            assert original_po is None or str(original_po) == str(rebuilt_po), f"previous observations don't match:\noriginal:\n{str(original_po)}\n!= rebuilt:\n{str(rebuilt_po)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
            assert str(original_o) == str(rebuilt_o), f"observations don't match:\noriginal:\n{str(original_o)}\n!= rebuilt:\n{str(rebuilt_o)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        if isinstance(original_a, np.ndarray):
            assert np.allclose(original_a, rebuilt_a, atol=epsilon), f"actions don't match:\noriginal:\n{str(original_a)}\n!= rebuilt:\n{str(rebuilt_a)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        else:
            assert str(original_a) == str(rebuilt_a), f"actions don't match:\noriginal:\n{str(original_a)}\n!= rebuilt:\n{str(rebuilt_a)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        assert np.allclose(original_r, rebuilt_r, atol=epsilon), f"rewards don't match:\noriginal:\n{str(original_r)}\n!= rebuilt:\n{str(rebuilt_r)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        assert original_d == rebuilt_d, f"terminated don't match:\noriginal:\n{str(original_d)}\n!= rebuilt:\n{str(rebuilt_d)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
        assert original_t == rebuilt_t, f"truncated don't match:\noriginal:\n{str(original_t)}\n!= rebuilt:\n{str(rebuilt_t)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"
    print(f"DEBUG: CRC check passed. Time step: {debug_ts}, since reset: {debug_ts_res}")


class BaseMemory(ABC):
    """
    Directly implement this instead of Memory if you want to optimize sampling (e.g., no collating...).

    Note: sample preprocessing (i.e., data augmentation), offline dataset and CRC debugging are optional and left to your discretion.
    """
    def __init__(self,
                 device,
                 memory_size=1000000,
                 batch_size=256,
                 sample_preprocessor:callable = None,
                 dataset_path="",
                 crc_debug=False):
        """
        Args:
            device (str): output tensors will be collated to this device
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            sample_preprocessor (callable): optional, can be used for data augmentation
            dataset_path (str): optional, can be used to initialize the memory with an offline dataset
            crc_debug (bool): optional, False usually, True when CRC debugging is activated
        """
        # Base memory attributes:
        self.device = device
        self.batch_size = batch_size
        self.memory_size = memory_size

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0
        self.stat_test_steps = 0
        self.stat_train_steps = 0

        # Optional attributes:
        self.sample_preprocessor = sample_preprocessor
        self.dataset_path = dataset_path
        self.crc_debug = crc_debug

    @abstractmethod
    def append_buffer(self, buffer):
        """
        Must append a Buffer object to the memory.

        Args:
            buffer (tmrl.networking.Buffer): the buffer of samples to append.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """
        Must sample a minibatch collated to self.device.

        Returns:
            minibatch (Tuple of tensors): possibly nested structure of tensors collated to self.device, and of batch dimension self.batch_size;
                minibatches are tuples of tensors of the form (last_obs, new_act, rew, new_obs, terminated, truncated)
        """
        raise NotImplementedError

    def get_benchmarks(self):
        """
        Can be overridden to log a tuple of floats for benchmarking the performance of your Memory class.

        Returns: benchmarks (Tuple of floats) or None
        """
        return None

    def get_benchmarks_names(self):
        """
        Use this method to name the elements of the tuple returned by get_benchmarks.

        Returns: benchmarks names (Tuple of strings) or None
        """
        return None

    def append(self, buffer):
        if len(buffer) > 0:
            self.stat_train_return = buffer.stat_train_return
            self.stat_test_return = buffer.stat_test_return
            self.stat_train_steps = buffer.stat_train_steps
            self.stat_test_steps = buffer.stat_test_steps
            self.append_buffer(buffer)


class Memory(BaseMemory, ABC):
    """
    Interface implementing the replay buffer.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """
    def __init__(self,
                 device,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=256,
                 dataset_path="",
                 crc_debug=False):
        """
        Args:
            device (str): output tensors will be collated to this device
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
        """
        super().__init__(device=device,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         sample_preprocessor=sample_preprocessor,
                         dataset_path=dataset_path,
                         crc_debug=crc_debug)

        # Benchmarking stats
        self.sampling_time = 0
        self.collating_time = 0
        self.nb_iterations = 0

        # init data
        self.path = Path(dataset_path)
        logging.debug(f"Memory self.path:{self.path}")
        if os.path.isfile(self.path / 'data.pkl'):
            with open(self.path / 'data.pkl', 'rb') as f:
                self.data = list(pickle.load(f))
        else:
            logging.info("no data found, initializing empty replay memory")
            self.data = []

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            logging.warning(f"the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")

    # def __iter__(self):
    #     for _ in range(self.nb_steps):
    #         yield self.sample()

    def get_benchmarks(self):
        if self.nb_iterations == 0:
            return 0.0, 0.0
        else:
            sampling_time = self.sampling_time / self.nb_iterations
            collating_time = self.collating_time / self.nb_iterations
            self.nb_iterations = 0
            self.sampling_time = 0.0
            self.collating_time = 0.0
            return sampling_time, collating_time

    def get_benchmarks_names(self):
        return "get_transitions", "collate"

    @abstractmethod
    def __len__(self):
        """
        Must return the length of the memory.

        Returns:
            int: the maximum `item` argument of `get_transition`

        """
        raise NotImplementedError

    @abstractmethod
    def get_transition(self, item):
        """
        Must return a transition.

        `info` is required in each sample for CRC debugging. The 'crc' key is what is important when using this feature.

        Args:
            item (int): the index where to sample

        Returns:
            Tuple: (prev_obs, prev_act, rew, obs, terminated, truncated, info)
        """
        raise NotImplementedError

    @abstractmethod
    def collate(self, batch, device):
        """
        Must collate `batch` onto `device`.

        `batch` is a list of training samples.
        The length of `batch` is `batch_size`.
        Each training sample in the list is of the form `(prev_obs, new_act, rew, new_obs, terminated, truncated)`.
        These samples must be collated into 6 tensors of batch dimension `batch_size`.
        These tensors should be collated onto the device indicated by the `device` argument.
        Then, your implementation must return a single tuple containing these 6 tensors.

        Args:
            batch (list): list of `(prev_obs, new_act, rew, new_obs, terminated, truncated)` tuples
            device: device onto which the list needs to be collated into batches `batch_size`

        Returns:
            Tuple of tensors:
            (prev_obs_tens, new_act_tens, rew_tens, new_obs_tens, terminated_tens, truncated_tens)
            collated on device `device`, each of batch dimension `batch_size`
        """
        raise NotImplementedError

    def __getitem__(self, item):
        prev_obs, new_act, rew, new_obs, terminated, truncated, info = self.get_transition(item)
        if self.crc_debug:
            po, a, o, r, d, t = info['crc_sample']
            debug_ts, debug_ts_res = info['crc_sample_ts']
            check_samples_crc(po, a, o, r, d, t, prev_obs, new_act, new_obs, rew, terminated, truncated, debug_ts, debug_ts_res)
        if self.sample_preprocessor is not None:
            prev_obs, new_act, rew, new_obs, terminated, truncated = self.sample_preprocessor(prev_obs, new_act, rew, new_obs, terminated, truncated)
        terminated = np.float32(terminated)  # we don't want bool tensors
        truncated = np.float32(truncated)  # we don't want bool tensors
        return prev_obs, new_act, rew, new_obs, terminated, truncated

    def sample(self):
        indices = self.sample_indices()
        t1 = time.perf_counter()
        batch = [self[idx] for idx in indices]
        t2 = time.perf_counter()
        batch = self.collate(batch, self.device)
        t3 = time.perf_counter()
        self.nb_iterations += 1
        self.sampling_time += t2 - t1
        self.collating_time += t3 - t2
        return batch

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batch_size))
