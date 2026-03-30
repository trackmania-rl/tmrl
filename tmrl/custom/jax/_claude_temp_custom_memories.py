# standard library imports
import numpy as np

# third-party imports
import jax
import jax.numpy as jnp
from flax import nnx

# local imports
from tmrl.core.memory import BaseMemory


# VARIABLE TYPE ========================================================================================================

class _BufferVar(nnx.Variable):
    """Non-differentiable NNX variable used to hold replay-buffer state."""
    pass


# MEMORY ===============================================================================================================

class ArrayFlaxMemory(BaseMemory):
    """
    JAX/Flax NNX replay buffer for flat numpy-array observations and actions.

    Unlike :class:`~tmrl.custom.torch.custom_memories.ArrayTorchMemory`, which
    grows its backing arrays dynamically, this class pre-allocates a
    fixed-size circular buffer so that **sample() has static array shapes and
    can be JIT-compiled** (e.g. via ``@nnx.jit`` or ``jax.jit``).

    Constraints (same as ArrayTorchMemory):
    - ``action`` and ``observation`` in each buffer sample must be
      **1-D homogeneous numpy arrays** (no nested structures).
    - ``sample_preprocessor`` is not supported.
    - CRC debug is not supported.

    Design overview
    ---------------
    *append_buffer* (host-side, not JIT-compiled)
        Writes incoming samples into the circular backing arrays using numpy
        slice assignment, then copies the updated arrays back to JAX device
        memory.  Info dicts are kept in a plain Python list.

    *sample* (JIT-compilable)
        1.  Builds an ``invalid`` boolean mask combining real episode-done
            flags (``buf_dones``) with a one-slot circular-boundary marker
            ``(ptr - 1) % mem`` — this prevents sampling a transition that
            would cross the write-pointer boundary.
        2.  Draws a batch of random indices with ``jax.random.randint``.
        3.  Uses ``jax.lax.while_loop`` (fixed-shape state → JIT-safe) to
            resample any index that falls on an invalid slot, exactly mirroring
            the rejection-sampling logic in ArrayTorchMemory.
        4.  Returns JAX arrays for ``(last_obs, new_act, rew, new_obs,
            terminated, truncated)``.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        memory_size: int = 1_000_000,
        batch_size: int = 256,
        dataset_path: str = "",
        sample_preprocessor: callable = None,
        crc_debug: bool = False,
        device: str = "cpu",
        seed: int = 0,
    ):
        """
        Args:
            obs_dim (int): flat observation dimension.
            act_dim (int): flat action dimension.
            memory_size (int): maximum number of stored transitions.
            batch_size (int): mini-batch size returned by sample().
            dataset_path (str): unused; kept for API compatibility.
            sample_preprocessor (callable): must be None (not supported).
            crc_debug (bool): must be False (not supported).
            device (str): unused; JAX selects the device automatically.
            seed (int): seed for the internal JAX PRNG.
        """
        if sample_preprocessor is not None:
            raise ValueError("ArrayFlaxMemory does not support sample_preprocessor.")
        if crc_debug:
            raise ValueError("ArrayFlaxMemory does not support crc_debug.")

        super().__init__(
            device=device,
            memory_size=memory_size,
            batch_size=batch_size,
            sample_preprocessor=sample_preprocessor,
            dataset_path=dataset_path,
            crc_debug=crc_debug,
        )

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        mem = memory_size

        # ------------------------------------------------------------------
        # Pre-allocated fixed-size circular buffer arrays.
        # Stored as _BufferVar so nnx.jit can lift/lower their values.
        # ------------------------------------------------------------------

        # Observations and actions — float32 arrays.
        self._buf_obs  = _BufferVar(jnp.zeros((mem, obs_dim), dtype=jnp.float32))
        self._buf_acts = _BufferVar(jnp.zeros((mem, act_dim), dtype=jnp.float32))
        self._buf_rews = _BufferVar(jnp.zeros(mem, dtype=jnp.float32))
        self._buf_ter  = _BufferVar(jnp.zeros(mem, dtype=jnp.bool_))
        self._buf_tru  = _BufferVar(jnp.zeros(mem, dtype=jnp.bool_))

        # Done flags (terminated OR truncated).
        # Initialised to True so that unwritten slots are never sampled as
        # the source index of a transition.  Written values replace these
        # True sentinels as real data arrives.
        self._buf_dones = _BufferVar(jnp.ones(mem, dtype=jnp.bool_))

        # Info dicts live outside JAX device memory.
        self._buf_infos: list = [None] * mem

        # Circular-buffer metadata (int32 scalars).
        self._ptr  = _BufferVar(jnp.array(0, dtype=jnp.int32))  # next write index
        self._size = _BufferVar(jnp.array(0, dtype=jnp.int32))  # valid entries written (≤ mem)

        # JAX PRNG key — split on every sample() call.
        self._rng_key = _BufferVar(jax.random.PRNGKey(seed))

    # ------------------------------------------------------------------
    # Host-side helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of valid sampleable transitions."""
        s = int(self._size.value)
        return max(0, min(s, self.memory_size) - 1)

    # ------------------------------------------------------------------
    # append_buffer  (host-side, not JIT-compiled)
    # ------------------------------------------------------------------

    def append_buffer(self, buffer) -> None:
        """
        Appends a :class:`~tmrl.networking.Buffer` of samples to the replay
        buffer.  Runs on the host (not JIT-compiled).

        Each sample in ``buffer.memory`` must be a tuple::

            (action: np.ndarray,
             obs: np.ndarray,
             rew: float,
             terminated: bool,
             truncated: bool,
             info: dict)
        """
        elt = buffer.memory[0]
        assert isinstance(elt[0], np.ndarray), (
            f"Actions must be numpy arrays. Got {type(elt[0])}"
        )
        assert isinstance(elt[1], np.ndarray), (
            f"Observations must be numpy arrays. Got {type(elt[1])}"
        )

        d_acts  = np.stack([b[0] for b in buffer.memory]).astype(np.float32)
        d_obs   = np.stack([b[1] for b in buffer.memory]).astype(np.float32)
        d_rews  = np.array([b[2] for b in buffer.memory], dtype=np.float32)
        d_ter   = np.array([b[3] for b in buffer.memory], dtype=bool)
        d_tru   = np.array([b[4] for b in buffer.memory], dtype=bool)
        d_dones = d_ter | d_tru
        d_infos = [b[5] for b in buffer.memory]

        n   = len(buffer.memory)
        mem = self.memory_size
        ptr = int(self._ptr.value)

        # ------------------------------------------------------------------
        # Work in host numpy for efficient vectorised slice-writes, then
        # transfer back to JAX arrays in one shot.
        # ------------------------------------------------------------------
        acts_np  = np.asarray(self._buf_acts.value)
        obs_np   = np.asarray(self._buf_obs.value)
        rews_np  = np.asarray(self._buf_rews.value)
        ter_np   = np.asarray(self._buf_ter.value)
        tru_np   = np.asarray(self._buf_tru.value)
        dones_np = np.asarray(self._buf_dones.value)

        # Vectorised write with wrap-around using modular indices.
        slot_indices = (ptr + np.arange(n)) % mem
        acts_np[slot_indices]  = d_acts
        obs_np[slot_indices]   = d_obs
        rews_np[slot_indices]  = d_rews
        ter_np[slot_indices]   = d_ter
        tru_np[slot_indices]   = d_tru
        dones_np[slot_indices] = d_dones
        for k, idx in enumerate(slot_indices):
            self._buf_infos[idx] = d_infos[k]

        new_ptr  = int((ptr + n) % mem)
        new_size = min(int(self._size.value) + n, mem)

        # Commit to JAX device memory.
        self._buf_acts.value  = jnp.array(acts_np)
        self._buf_obs.value   = jnp.array(obs_np)
        self._buf_rews.value  = jnp.array(rews_np)
        self._buf_ter.value   = jnp.array(ter_np)
        self._buf_tru.value   = jnp.array(tru_np)
        self._buf_dones.value = jnp.array(dones_np)
        self._ptr.value       = jnp.int32(new_ptr)
        self._size.value      = jnp.int32(new_size)

    # ------------------------------------------------------------------
    # sample  (JIT-compilable)
    # ------------------------------------------------------------------

    def sample(self):
        """
        Samples a mini-batch of valid transitions.

        This method is **JIT-compilable** via ``@nnx.jit`` or ``jax.jit``
        because:

        - All array shapes are static (pre-allocated circular buffer).
        - The rejection-sampling loop uses ``jax.lax.while_loop`` (fixed
          state pytree with known shapes).
        - The PRNG state is a JAX array advanced via ``jax.random.split``.

        Call only when ``len(self) >= self.batch_size``.

        Returns:
            Tuple of JAX float32/bool arrays, each of shape
            ``(batch_size, ...)``:
            ``(last_obs, new_act, rew, new_obs, terminated, truncated)``
        """
        mem        = self.memory_size
        batch_size = self.batch_size
        ptr        = self._ptr.value           # int32 scalar
        size       = self._size.value          # int32 scalar
        buf_dones  = self._buf_dones.value     # (mem,) bool

        valid_size = jnp.minimum(size, jnp.int32(mem))

        # ------------------------------------------------------------------
        # Build the invalid-source mask.
        # A slot i must NOT be used as idx_last when:
        #   (a) buf_dones[i] is True  — real episode terminal/truncation, OR
        #       unwritten sentinel (initialised to True), or
        #   (b) i == (ptr - 1) % mem  — the most-recently written slot,
        #       whose successor idx_now == ptr would either be unwritten
        #       (buffer not yet full) or the oldest overwritten slot
        #       (buffer full — cross-episode boundary).
        # ------------------------------------------------------------------
        boundary_idx = (ptr - jnp.int32(1) + jnp.int32(mem)) % jnp.int32(mem)
        range_mem    = jnp.arange(mem, dtype=jnp.int32)
        invalid      = buf_dones | (range_mem == boundary_idx)

        # ------------------------------------------------------------------
        # Sample idx_last from [0, valid_size - 1).
        # The upper bound excludes slot (valid_size - 1) == (ptr - 1) which
        # is already guarded by the boundary check above, but the range
        # restriction also keeps indices within the written portion when the
        # buffer is not yet full.
        # ------------------------------------------------------------------
        max_idx = valid_size - jnp.int32(1)

        rng_key, subkey = jax.random.split(self._rng_key.value)
        self._rng_key.value = rng_key

        subkey, sk0 = jax.random.split(subkey)
        indices = jax.random.randint(sk0, shape=(batch_size,), minval=0, maxval=max_idx)

        # ------------------------------------------------------------------
        # Rejection-sampling loop — mirrors the Python while loop in
        # ArrayTorchMemory but uses jax.lax.while_loop for JIT safety.
        # State pytree: (indices: int32[batch], key: PRNGKey)  — fixed shapes.
        # ------------------------------------------------------------------
        def cond_fn(state):
            idxs, _key = state
            return jnp.any(invalid[idxs])

        def body_fn(state):
            idxs, key = state
            key, new_key = jax.random.split(key)
            bad  = invalid[idxs]                                           # (batch,) bool
            new  = jax.random.randint(new_key, shape=(batch_size,), minval=0, maxval=max_idx)
            idxs = jnp.where(bad, new, idxs)
            return idxs, key

        indices, _ = jax.lax.while_loop(cond_fn, body_fn, (indices, subkey))

        idx_last = indices
        idx_now  = (indices + jnp.int32(1)) % jnp.int32(mem)

        last_obs_batch   = self._buf_obs.value[idx_last]
        new_act_batch    = self._buf_acts.value[idx_now]
        rew_batch        = self._buf_rews.value[idx_now]
        new_obs_batch    = self._buf_obs.value[idx_now]
        terminated_batch = self._buf_ter.value[idx_now]
        truncated_batch  = self._buf_tru.value[idx_now]

        return (
            last_obs_batch,
            new_act_batch,
            rew_batch,
            new_obs_batch,
            terminated_batch,
            truncated_batch,
        )
