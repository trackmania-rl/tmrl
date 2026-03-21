"""Best TrackMania memory implementation with all telemetry fields."""

import numpy as np

from tmrl.custom.memories.base import MemoryTM, last_true_in_list, replace_hist_before_eoe
from tmrl.custom.memories.enums import BufferField, TMBestField, TMBestObsField


class MemoryTMBest(MemoryTM):
    """Comprehensive TrackMania memory with full telemetry data."""

    def get_transition(self, item: int):
        """Get a single transition with proper episode boundary handling."""
        f = TMBestField

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        last_eoes = self.data[f.EOES][idx_now - self.min_samples : idx_now]
        last_eoe_idx = last_true_in_list(last_eoes)

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(
                hist=acts[1:], eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1
            )
            replace_hist_before_eoe(
                hist=acts[:-1], eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset
            )
            replace_hist_before_eoe(
                hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1
            )
            replace_hist_before_eoe(
                hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset
            )

        last_obs = (
            self.data[f.POSITION][idx_last],
            self.data[f.SPEED][idx_last],
            self.data[f.ACCELERATION][idx_last],
            self.data[f.JERK][idx_last],
            self.data[f.RACE_PROGRESS][idx_last],
            self.data[f.INPUT_STEER][idx_last],
            self.data[f.INPUT_GAS_PEDAL][idx_last],
            self.data[f.INPUT_BRAKE][idx_last],
            self.data[f.GEAR][idx_last],
            self.data[f.AIM_YAW][idx_last],
            self.data[f.AIM_PITCH][idx_last],
            self.data[f.SURFACE_ID][idx_last],
            self.data[f.STEER_ANGLE][idx_last],
            self.data[f.WHEEL_ROT][idx_last],
            self.data[f.WHEEL_ROT_SPEED][idx_last],
            self.data[f.DAMPER_LEN][idx_last],
            self.data[f.SLIP_COEF][idx_last],
            self.data[f.REACTOR_GROUND_MODE][idx_last],
            self.data[f.GROUND_CONTACT][idx_last],
            self.data[f.REACTOR_AIR_CONTROL][idx_last],
            self.data[f.GROUND_DIST][idx_last],
            self.data[f.CRASHED][idx_last],
            self.data[f.FAILURE_COUNTER][idx_last],
            self.data[f.IMGS][idx_last],
            *last_act_buf,
        )
        new_act = self.data[f.ACTIONS][idx_now]
        rew = np.float32(self.data[f.REWARDS][idx_now])
        new_obs = (
            self.data[f.POSITION][idx_now],
            self.data[f.SPEED][idx_now],
            self.data[f.ACCELERATION][idx_now],
            self.data[f.JERK][idx_now],
            self.data[f.RACE_PROGRESS][idx_now],
            self.data[f.INPUT_STEER][idx_now],
            self.data[f.INPUT_GAS_PEDAL][idx_now],
            self.data[f.INPUT_BRAKE][idx_now],
            self.data[f.GEAR][idx_now],
            self.data[f.AIM_YAW][idx_now],
            self.data[f.AIM_PITCH][idx_now],
            self.data[f.SURFACE_ID][idx_now],
            self.data[f.STEER_ANGLE][idx_now],
            self.data[f.WHEEL_ROT][idx_now],
            self.data[f.WHEEL_ROT_SPEED][idx_now],
            self.data[f.DAMPER_LEN][idx_now],
            self.data[f.SLIP_COEF][idx_now],
            self.data[f.REACTOR_GROUND_MODE][idx_now],
            self.data[f.GROUND_CONTACT][idx_now],
            self.data[f.REACTOR_AIR_CONTROL][idx_now],
            self.data[f.GROUND_DIST][idx_now],
            self.data[f.CRASHED][idx_now],
            self.data[f.FAILURE_COUNTER][idx_now],
            self.data[f.IMGS][idx_now],
            *acts[1:],
        )
        terminated = self.data[f.TERMINATED][idx_now]
        truncated = self.data[f.TRUNCATED][idx_now]
        info = self.data[f.INFOS][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item: int):
        """Load image sequence for a transition."""
        res = self.data[TMBestField.IMGS][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item: int):
        """Load action sequence for a transition."""
        res = self.data[TMBestField.ACTIONS][
            (item + self.start_acts_offset) : (item + self.start_acts_offset + self.act_buf_len + 1)
        ]
        return res

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory."""
        f = TMBestField
        first_data_idx = self.data[f.INDEXES][-1] + 1 if self.__len__() > 0 else 0
        bf = BufferField

        o = TMBestObsField
        crashed_list = [b[bf.OBSERVATION][o.CRASHED_LIST].tolist() for b in buffer.memory]

        data_fields = [
            [first_data_idx + i for i, _ in enumerate(buffer.memory)],
            [b[bf.ACTION] for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.POSITION]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.SPEED]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.ACCELERATION]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.JERK]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.RACE_PROGRESS]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.INPUT_STEER]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.INPUT_GAS_PEDAL]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.INPUT_BRAKE]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.GEAR]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.AIM_YAW]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.AIM_PITCH]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.STEER_ANGLE][0]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.WHEEL_ROT][0]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.WHEEL_ROT_SPEED][0]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.DAMPER_LEN][0]]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.SLIP_COEF]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.REACTOR_GROUND_MODE]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.GROUND_CONTACT]]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.REACTOR_AIR_CONTROL]]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.GROUND_DIST]) for b in buffer.memory],
            [np.array([b[bf.OBSERVATION][o.CRASHED]]) for b in buffer.memory],
            [np.array([el]) for el in crashed_list],
            [np.array([b[bf.OBSERVATION][o.FAILURE_COUNTER]]) for b in buffer.memory],
            [b[bf.OBSERVATION][o.IMGS] for b in buffer.memory],
            [b[bf.TERMINATED] or b[bf.TRUNCATED] for b in buffer.memory],
            [b[bf.REWARD] for b in buffer.memory],
            [b[bf.INFO] for b in buffer.memory],
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
