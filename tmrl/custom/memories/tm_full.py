"""Full TrackMania memory with images and all telemetry data."""

import random

import numpy as np

from tmrl.custom.memories.base import MemoryTM, last_true_in_list, replace_hist_before_eoe
from tmrl.custom.memories.enums import BufferField, TMFullField, TMFullObsField


class MemoryTMFull(MemoryTM):
    """Full-featured TrackMania replay memory with images."""

    def get_transition(self, item: int):
        """Get a single transition with proper episode boundary handling."""
        f = TMFullField

        if self.data[f.EOES][item + self.min_samples - 1]:
            if item == 0:
                item += 1
            elif item == self.__len__() - 1:
                item -= 1
            elif random.random() < 0.5:
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        last_eoes = self.data[f.EOES][idx_now - self.min_samples : idx_now]
        last_eoe_idx = last_true_in_list(last_eoes)

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(
                hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1
            )
            replace_hist_before_eoe(
                hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset
            )
            replace_hist_before_eoe(
                hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1
            )
            replace_hist_before_eoe(
                hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset
            )

        return (
            (
                self.data[f.SPEEDS][idx_last],
                self.data[f.GEARS][idx_last],
                self.data[f.RPMS][idx_last],
                imgs_last_obs,
                *last_act_buf,
            ),
            self.data[f.ACTIONS][idx_now],
            np.float32(self.data[f.REWARDS][idx_now]),
            (
                self.data[f.SPEEDS][idx_now],
                self.data[f.GEARS][idx_now],
                self.data[f.RPMS][idx_now],
                imgs_new_obs,
                *new_act_buf,
            ),
            self.data[f.TERMINATED][idx_now],
            self.data[f.TRUNCATED][idx_now],
            self.data[f.INFOS][idx_now],
        )

    def load_imgs(self, item: int):
        """Load image sequence for a transition."""
        res = self.data[TMFullField.IMAGES][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item: int):
        """Load action sequence for a transition."""
        res = self.data[TMFullField.ACTIONS][
            (item + self.start_acts_offset) : (item + self.start_acts_offset + self.act_buf_len + 1)
        ]
        return res

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory."""
        f = TMFullField
        first_data_idx = self.data[f.INDEXES][-1] + 1 if self.__len__() > 0 else 0
        bf = BufferField

        o = TMFullObsField

        data_fields = [
            [first_data_idx + i for i, _ in enumerate(buffer.memory)],
            [b[bf.ACTION] for b in buffer.memory],
            [b[bf.OBSERVATION][o.SPEEDS] for b in buffer.memory],
            [b[bf.OBSERVATION][o.IMAGES] for b in buffer.memory],
            [b[bf.TERMINATED] or b[bf.TRUNCATED] for b in buffer.memory],
            [b[bf.REWARD] for b in buffer.memory],
            [b[bf.INFO] for b in buffer.memory],
            [b[bf.OBSERVATION][o.GEARS] for b in buffer.memory],
            [b[bf.OBSERVATION][o.RPMS] for b in buffer.memory],
            [b[bf.TERMINATED] for b in buffer.memory],
            [b[bf.TRUNCATED] for b in buffer.memory],
        ]

        if self.__len__() > 0:
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            self.data = list(data_fields)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            for i in range(len(data_fields)):
                self.data[i] = self.data[i][to_trim:]

        return self
