"""R2D2-based memory implementations for TrackMania."""

from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from tmrl.custom.memories.base import last_true_in_list, replace_hist_before_eoe
from tmrl.custom.memories.enums import (
    BufferField,
    R2D2Field,
    R2D2ObsField,
    R2D2SophyField,
    R2D2SophyObsField,
    R2D2woImagesTrailingField,
)
from tmrl.memory import R2D2Memory


class MemoryR2D2(R2D2Memory):
    """R2D2-style replay memory with full telemetry and images."""

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 2,
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

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        return max(0, res)

    def get_transition(self, item: int):
        """Get a single transition with proper episode boundary handling."""
        f = R2D2Field

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
                self.data[f.CHECKPOINTS][idx_last],
                self.data[f.SPEEDS][idx_last],
                self.data[f.ACCELERATIONS][idx_last],
                self.data[f.JERKS][idx_last],
                self.data[f.RACE_PROGRESS][idx_last],
                self.data[f.INPUT_STEER][idx_last],
                self.data[f.INPUT_GAS_PEDAL][idx_last],
                self.data[f.INPUT_BRAKE][idx_last],
                self.data[f.GEAR][idx_last],
                self.data[f.AIM_YAW][idx_last],
                self.data[f.AIM_PITCH][idx_last],
                self.data[f.STEER_ANGLE][idx_last],
                self.data[f.SLIP_COEF][idx_last],
                self.data[f.FAILURE_COUNTER][idx_last],
                *last_act_buf,
            ),
            self.data[f.ACTIONS][idx_now],
            np.float32(self.data[f.REWARDS][idx_now]),
            (
                self.data[f.CHECKPOINTS][idx_now],
                self.data[f.SPEEDS][idx_now],
                self.data[f.ACCELERATIONS][idx_now],
                self.data[f.JERKS][idx_now],
                self.data[f.RACE_PROGRESS][idx_now],
                self.data[f.INPUT_STEER][idx_now],
                self.data[f.INPUT_GAS_PEDAL][idx_now],
                self.data[f.INPUT_BRAKE][idx_now],
                self.data[f.GEAR][idx_now],
                self.data[f.AIM_YAW][idx_now],
                self.data[f.AIM_PITCH][idx_now],
                self.data[f.STEER_ANGLE][idx_now],
                self.data[f.SLIP_COEF][idx_now],
                self.data[f.FAILURE_COUNTER][idx_now],
                *new_act_buf,
            ),
            self.data[f.TERMINATED][idx_now],
            self.data[f.TRUNCATED][idx_now],
            self.data[f.INFOS][idx_now],
        )

    def load_imgs(self, item: int):
        """Load image sequence for a transition."""
        res = self.data[R2D2Field.IMGS][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item: int):
        """Load action sequence for a transition."""
        res = self.data[R2D2Field.ACTIONS][
            (item + self.start_acts_offset) : (item + self.start_acts_offset + self.act_buf_len + 1)
        ]
        return res

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory."""
        f = R2D2Field
        first_data_idx = self.data[f.INDEXES][-1] + 1 if self.__len__() > 0 else 0
        bf = BufferField
        o = R2D2ObsField

        data_fields = [
            [first_data_idx + i for i, _ in enumerate(buffer.memory)],
            [b[bf.ACTION] for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.CHECKPOINTS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.SPEEDS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.ACCELERATIONS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.JERKS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.RACE_PROGRESS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.INPUT_STEER]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.INPUT_GAS_PEDAL]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.INPUT_BRAKE]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.GEAR]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.AIM_YAW]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.AIM_PITCH]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.STEER_ANGLE]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.SLIP_COEF]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.FAILURE_COUNTER]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.IMGS]) for b in buffer.memory],
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


class MemoryR2D2woImages(R2D2Memory):
    """R2D2-style replay memory without images."""

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 2,
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

    def _obs_end(self) -> int:
        """Index (exclusive) of the last observation column in self.data."""
        return len(self.data) - 5

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        return max(0, res)

    def set_observation_space(self, space) -> None:
        """Set the authoritative observation space (from the env/interface).

        When set, alignment in ``get_transition`` and ``append_buffer`` uses this
        space instead of rebuilding one from ``cfg.POINTS_NUMBER``, which avoids
        a 1-dim mismatch when ``RewardFunction._points_number`` overrides the
        config-level constant.
        """
        self._cached_tqc_obs_space = space

    def _trainer_tqc_obs_space(self):
        if not hasattr(self, "_cached_tqc_obs_space"):
            from tmrl.custom.tm.tqc_observation_space import build_tqc_sophy_tuple_observation_space

            self._cached_tqc_obs_space = build_tqc_sophy_tuple_observation_space()
        return self._cached_tqc_obs_space

    def get_transition(self, item: int):
        """Get a single transition."""
        from tmrl.tools.player_runs import align_observation_to_space

        t = R2D2woImagesTrailingField
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        obs_end = self._obs_end()

        space = self._trainer_tqc_obs_space()
        prev_obs = tuple(self.data[i][idx_last] for i in range(2, obs_end))
        next_obs = tuple(self.data[i][idx_now] for i in range(2, obs_end))
        prev_obs = align_observation_to_space(prev_obs, space)
        next_obs = align_observation_to_space(next_obs, space)

        return (
            prev_obs,
            self.data[1][idx_now],
            np.float32(self.data[obs_end + t.REWARDS][idx_now]),
            next_obs,
            self.data[obs_end + t.TERMINATED][idx_now],
            self.data[obs_end + t.TRUNCATED][idx_now],
            self.data[obs_end + t.INFOS][idx_now],
        )

    def load_imgs(self, item: int):
        """Placeholder for API compatibility."""
        res = self.data[2][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res)

    def load_acts(self, item: int):
        """Load action sequence for a transition."""
        res = self.data[1][
            (item + self.start_acts_offset) : (item + self.start_acts_offset + self.act_buf_len + 1)
        ]
        return res

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory."""
        from tmrl.tools.player_runs import (
            align_observation_to_space,
            observation_matches_space,
        )

        bf = BufferField
        space = self._trainer_tqc_obs_space()
        kept: list[Any] = []
        n_drop = 0
        for b in buffer.memory:
            obs = align_observation_to_space(b[bf.OBSERVATION], space)
            if not observation_matches_space(obs, space):
                n_drop += 1
                continue
            kept.append(
                (b[bf.ACTION], obs, b[bf.REWARD], b[bf.TERMINATED], b[bf.TRUNCATED], b[bf.INFO])
            )
        if n_drop:
            logger.warning(
                "MemoryR2D2woImages: dropped {} sample(s) with observations incompatible "
                "with trainer TQC observation_space (after alignment).",
                n_drop,
            )
        if not kept:
            return self
        buffer.memory = kept

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0
        n_obs = len(buffer.memory[0][bf.OBSERVATION])

        data_fields = [
            [first_data_idx + i for i, _ in enumerate(buffer.memory)],
            [b[bf.ACTION] for b in buffer.memory],
        ]
        for j in range(n_obs):
            data_fields.append([np.array(b[bf.OBSERVATION][j]) for b in buffer.memory])
        data_fields.append([b[bf.TERMINATED] or b[bf.TRUNCATED] for b in buffer.memory])
        data_fields.append([b[bf.REWARD] for b in buffer.memory])
        data_fields.append([b[bf.INFO] for b in buffer.memory])
        data_fields.append([b[bf.TERMINATED] for b in buffer.memory])
        data_fields.append([b[bf.TRUNCATED] for b in buffer.memory])

        if self.__len__() > 0 and len(self.data) == len(data_fields):
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            if self.__len__() > 0 and len(self.data) != len(data_fields):
                logger.warning(
                    "Memory column count changed ({} -> {}); resetting buffer.",
                    len(self.data),
                    len(data_fields),
                )
            self.data = list(data_fields)

        self.rewards_index = len(self.data) - 3

        n_fields = len(self.data)
        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            for i in range(n_fields):
                self.data[i] = self.data[i][to_trim:]

        return self


class MemoryR2D2Sophy(R2D2Memory):
    """R2D2-style replay memory for Sophy interface."""

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 2,
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

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        return max(0, res)

    def get_transition(self, item: int):
        """Get a single transition."""
        f = R2D2SophyField

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)

        return (
            (
                self.data[f.TRACK_INFO][idx_last],
                self.data[f.SPEEDS][idx_last],
                self.data[f.ACCELERATIONS][idx_last],
                self.data[f.JERKS][idx_last],
                self.data[f.RACE_PROGRESS][idx_last],
                self.data[f.INPUT_STEER][idx_last],
                self.data[f.INPUT_GAS_PEDAL][idx_last],
                self.data[f.INPUT_BRAKE][idx_last],
                self.data[f.GEAR][idx_last],
                self.data[f.AIM_YAW][idx_last],
                self.data[f.AIM_PITCH][idx_last],
                self.data[f.STEER_ANGLE][idx_last],
                self.data[f.SLIP_COEF][idx_last],
                self.data[f.FAILURE_COUNTER][idx_last],
                *acts[:-1],
            ),
            self.data[f.ACTIONS][idx_now],
            np.float32(self.data[f.REWARDS][idx_now]),
            (
                self.data[f.TRACK_INFO][idx_now],
                self.data[f.SPEEDS][idx_now],
                self.data[f.ACCELERATIONS][idx_now],
                self.data[f.JERKS][idx_now],
                self.data[f.RACE_PROGRESS][idx_now],
                self.data[f.INPUT_STEER][idx_now],
                self.data[f.INPUT_GAS_PEDAL][idx_now],
                self.data[f.INPUT_BRAKE][idx_now],
                self.data[f.GEAR][idx_now],
                self.data[f.AIM_YAW][idx_now],
                self.data[f.AIM_PITCH][idx_now],
                self.data[f.STEER_ANGLE][idx_now],
                self.data[f.SLIP_COEF][idx_now],
                self.data[f.FAILURE_COUNTER][idx_now],
                *acts[1:],
            ),
            self.data[f.TERMINATED][idx_now],
            self.data[f.TRUNCATED][idx_now],
            self.data[f.INFOS][idx_now],
        )

    def load_imgs(self, item: int):
        """Placeholder for API compatibility."""
        res = self.data[2][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res)

    def load_acts(self, item: int):
        """Load action sequence for a transition."""
        res = self.data[R2D2SophyField.ACTIONS][
            (item + self.start_acts_offset) : (item + self.start_acts_offset + self.act_buf_len + 1)
        ]
        return res

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory."""
        f = R2D2SophyField
        first_data_idx = self.data[f.INDEXES][-1] + 1 if self.__len__() > 0 else 0
        bf = BufferField

        o = R2D2SophyObsField

        data_fields = [
            [first_data_idx + i for i, _ in enumerate(buffer.memory)],
            [b[bf.ACTION] for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.TRACK_INFO]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.SPEEDS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.ACCELERATIONS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.JERKS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.RACE_PROGRESS]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.INPUT_STEER]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.INPUT_GAS_PEDAL]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.INPUT_BRAKE]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.GEAR]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.AIM_YAW]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.AIM_PITCH]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.STEER_ANGLE]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.SLIP_COEF]) for b in buffer.memory],
            [np.array(b[bf.OBSERVATION][o.FAILURE_COUNTER]) for b in buffer.memory],
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
