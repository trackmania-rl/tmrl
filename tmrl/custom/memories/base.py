"""Base memory classes for TrackMania reinforcement learning."""

from collections.abc import Callable
from typing import Any

import numpy as np

from tmrl.custom.memories.enums import BufferField, GenericField
from tmrl.memory import TorchMemory


def last_true_in_list(li: list[bool]) -> int | None:
    """Find the index of the last True value in a list."""
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


def replace_hist_before_eoe(hist: list, eoe_idx_in_hist: int) -> None:
    """Pad history before the End Of Episode (EOE) index."""
    last_idx = len(hist) - 1
    if eoe_idx_in_hist > last_idx:
        raise ValueError(
            f"replace_hist_before_eoe: eoe_idx_in_hist ({eoe_idx_in_hist}) > last_idx ({last_idx})"
        )
    if 0 <= eoe_idx_in_hist < last_idx:
        for i in reversed(range(len(hist))):
            if i <= eoe_idx_in_hist:
                hist[i] = hist[i + 1]


class GenericTorchMemory(TorchMemory):
    """Generic torch-based memory for simple replay buffer scenarios."""

    def __init__(
        self,
        memory_size: int = 1_000_000,
        batch_size: int = 1,
        dataset_path: str = "",
        nb_steps: int = 1,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
        )

    def append_buffer(self, buffer: Any) -> None:
        """Append a buffer of samples to the memory."""
        bf = BufferField
        data_fields = [
            [b[bf.ACTION] for b in buffer.memory],
            [b[bf.OBSERVATION] for b in buffer.memory],
            [b[bf.REWARD] for b in buffer.memory],
            [b[bf.TERMINATED] for b in buffer.memory],
            [b[bf.TRUNCATED] for b in buffer.memory],
            [b[bf.INFO] for b in buffer.memory],
            [b[bf.TERMINATED] or b[bf.TRUNCATED] for b in buffer.memory],
        ]

        if self.__len__() > 0:
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            self.data = list(data_fields)

        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            for i in range(len(data_fields)):
                self.data[i] = self.data[i][to_trim:]

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        return max(0, res)

    def clear(self) -> None:
        """Remove all transitions from the memory."""
        self.data = []

    def get_transition(self, item: int) -> tuple:
        """Get a single transition from the memory."""
        field = GenericField

        max_retries = self.__len__()
        for _ in range(max_retries):
            if not self.data[field.DONE][item]:
                break
            item = np.random.randint(0, self.__len__() - 1)
        else:
            raise RuntimeError("All transitions in memory are terminal (done=True).")

        idx_last = item
        idx_now = item + 1

        return (
            self.data[field.OBSERVATIONS][idx_last],
            self.data[field.ACTIONS][idx_now],
            self.data[field.REWARDS][idx_now],
            self.data[field.OBSERVATIONS][idx_now],
            self.data[field.TERMINATED][idx_now],
            self.data[field.TRUNCATED][idx_now],
            self.data[field.INFO][idx_now],
        )


class MemoryTM(TorchMemory):
    """Base class for TrackMania replay memories with temporal structure."""

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 1,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
    ):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
        )

    def append_buffer(self, buffer):
        """Append a buffer of samples - must be implemented by subclasses."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        return max(0, res)

    def get_transition(self, item: int):
        """Get a single transition - must be implemented by subclasses."""
        raise NotImplementedError
