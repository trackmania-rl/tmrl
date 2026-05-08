import logging

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from tmrl.core.memory import BaseMemory


class NNXVar(nnx.Variable): pass


class ArrayNNXMemory(nnx.Module, BaseMemory):
    """
    Memory optimized for vector obsevations and actions.
    Based on NNX, on-device, jit-compilable.

    This is highly optimized for the sampling-training loop, at the expense of append_buffer.
    append_buffer must be called parsimoniously, with large buffers.
    """

    def __init__(self,
                 observation_vector_dim: int,
                 action_vector_dim: int,
                 memory_size: int = 1e6,
                 batch_size: int = 1,
                 dataset_path: str = "",
                 sample_preprocessor: callable = None,
                 crc_debug: bool = False,
                 device: str = "cpu",
                 seed: int = 0):
        
        memory_size = int(memory_size)

        if sample_preprocessor is not None:
            raise ValueError("sample_preprocessor not supported")
        if crc_debug:
            raise ValueError("crc_debug not supported")

        BaseMemory.__init__(self,
                             device=device,
                             memory_size=memory_size,
                             batch_size=batch_size,
                             dataset_path=dataset_path,
                             sample_preprocessor=sample_preprocessor,
                             crc_debug=crc_debug)
        

        
        self.observation_vector_dim = observation_vector_dim
        self.action_vector_dim = action_vector_dim

        # NNX seed for random sampling
        self._rng_key = NNXVar(jax.random.key(seed))

        # pre-allocate empty data (TODO: could naturally implement dataset_path support here)
        self._data_obs = NNXVar(jnp.zeros((memory_size, observation_vector_dim), dtype=jnp.float32))
        self._data_act = NNXVar(jnp.zeros((memory_size, action_vector_dim), dtype=jnp.float32))
        self._data_rew = NNXVar(jnp.zeros(memory_size, dtype=jnp.float32))
        self._data_terminated = NNXVar(jnp.zeros(memory_size, dtype=bool))
        self._data_truncated = NNXVar(jnp.zeros(memory_size, dtype=bool))

        # done flag (terminated or truncated)
        # this is initialized to True everywhere as it also serves as a mask for samplable slots
        # (NB: slots where done is True cannot be sampled because they go from done to reset)
        self._data_done = NNXVar(jnp.ones(memory_size, dtype=bool))

        # valid mask from which we can sample
        self._valid_indices = NNXVar(jnp.zeros(memory_size, dtype=jnp.int32))  # samplable indices
        self._num_valid = NNXVar(jnp.array(0, dtype=jnp.int32))  # nb of samplable indices

        # pointer into the circular buffer for appending
        self._ptr = NNXVar(jnp.array(0, dtype=jnp.int32))  # next wite index
        self._size = NNXVar(jnp.array(0, dtype=jnp.int32))  # nb of written slots

        # info dicts (outside JAX workspace)
        # self._data_info: list = [None] * memory_size

    def __len__(self):
        return int(self._num_valid.value)

    def append_buffer(self, buffer):
        """
        Appends a `tmrl.networking.Buffer` of samples to the memory.
        This method is not jit-able.
        """

        elt = buffer.memory[0]
        assert isinstance(elt[0], np.ndarray), f"Actions must be numpy arrays. Found {type(elt[0])}"
        assert isinstance(elt[1], np.ndarray), f"Observations must be numpy arrays. Found {type(elt[1])}"

        # parse:
        d_act = np.stack([b[0] for b in buffer.memory], dtype=np.float32)  # actions
        d_obs = np.stack([b[1] for b in buffer.memory], dtype=np.float32)  # observations
        d_rew = np.stack([b[2] for b in buffer.memory], dtype=np.float32)  # rewards
        d_terminated = np.stack([b[3] for b in buffer.memory])  # terminated
        d_truncated = np.stack([b[4] for b in buffer.memory])  # truncated
        # d_info = [b[5] for b in buffer.memory]  # info dicts
        d_done = d_terminated | d_truncated  # done

        len_buf = len(buffer.memory)
        assert len(d_done) == len_buf, f"DEBUG: something is wrong"
        mem_size = self.memory_size
        ptr = int(self._ptr.value)

        # JAX arrays are immutable
        # Thus we need to copy the full memory to append the incoming buffer
        np_act = np.array(self._data_act.value)
        np_obs = np.array(self._data_obs.value)
        np_rew = np.array(self._data_rew.value)
        np_terminated = np.array(self._data_terminated.value)
        np_truncated = np.array(self._data_truncated.value)
        np_done = np.array(self._data_done.value)

        # vectorized indexing (circular memory)
        indices = (ptr + np.arange(len_buf)) % mem_size
        np_act[indices] = d_act
        np_obs[indices] = d_obs
        np_rew[indices] = d_rew
        np_terminated[indices] = d_terminated
        np_truncated[indices] = d_truncated
        np_done[indices] = d_done

        # info dictionaries
        # for k, idx in enumerate(indices):
        #     self._data_info[idx] = d_info[k]
        
        new_ptr = int((ptr + len_buf) % mem_size)
        new_size = min(int(self._size.value) + len_buf, mem_size)

        # compute list of samplable indices
        last_written_idx = new_ptr - 1 if new_ptr else mem_size - 1
        valid_mask = ~np_done
        valid_mask[last_written_idx] = False
        valid_indices = np.where(valid_mask)[0].astype(np.int32)
        num_valid = len(valid_indices)
        padded = np.zeros(mem_size, dtype=np.int32)
        padded[:num_valid] = valid_indices
        
        # update JAX memory
        self._data_act.value = jnp.array(np_act)
        self._data_obs.value = jnp.array(np_obs)
        self._data_rew.value = jnp.array(np_rew)
        self._data_terminated.value = jnp.array(np_terminated)
        self._data_truncated.value = jnp.array(np_truncated)
        self._data_done.value = jnp.array(np_done)
        self._ptr.value = jnp.int32(new_ptr)
        self._size.value = jnp.int32(new_size)
        self._valid_indices.value = jnp.array(padded)
        self._num_valid.value = jnp.int32(num_valid)
    
    @nnx.jit
    def sample(self):
        """
        JIT-compilable
        """

        mem_size = self.memory_size
        batch_size = self.batch_size
        num_valid = self._num_valid.value

        # draw indices from valid_indices directly (NB: circular buffer)
        key, subkey = jax.random.split(self._rng_key.value)
        self._rng_key.value = key
        idx = jax.random.randint(key=subkey, shape=(batch_size,), minval=0, maxval=num_valid)
        idx_last = self._valid_indices.value[idx]
        idx_now = (idx_last + 1) % mem_size

        # jnp batched tensors:
        last_obs_batch = self._data_obs.value[idx_last]
        new_act_batch = self._data_act.value[idx_now]
        rew_batch = self._data_rew.value[idx_now]
        new_obs_batch = self._data_obs.value[idx_now]
        terminated_batch = self._data_terminated.value[idx_now]
        truncated_batch = self._data_truncated.value[idx_now]

        # CRC-debug numpy batched tensors: # FIXME: CANNOT BE DONE HERE
        # if self.crc_debug:
        #     for i in range(len(idx_now)):
        #         prev_obs = last_obs_batch[i]
        #         new_act = new_act_batch[i]
        #         rew = rew_batch[i]
        #         new_obs = new_obs_batch[i]
        #         terminated = terminated_batch[i]
        #         truncated = truncated_batch[i]
        #         info = self.data[5][idx_now[i]]
        #         po, a, o, r, d, t = info['crc_sample']
        #         debug_ts, debug_ts_res = info['crc_sample_ts']
        #         check_samples_crc(po, a, o, r, d, t, prev_obs, new_act, new_obs, rew, terminated, truncated, debug_ts, debug_ts_res, epsilon=1e-4)

        return last_obs_batch, new_act_batch, rew_batch, new_obs_batch, terminated_batch, truncated_batch
