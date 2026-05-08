import jax.numpy as jnp
import jax
from flax import nnx
import numpy as np


__docformat__ = "google"


MAX_SEED = 2**32


def collate_jax(batch, device=None):
    """Turns a batch of nested structures with numpy arrays as leaves into a single element of the same nested structure with batched jax arrays as leaves"""
    collated = jax.tree.map(lambda *xs: jnp.stack(xs), *batch)
    if device is not None:
        collated = jax.device_put(collated, device)
    return collated


# === randomness =======================================================================================================

# This function is NOT jit-compilable
def get_rngs(
        params_seed=None,
        dropout_seed=None,
        noise_seed=None
        ):
    """
    Return an RNG with default seeds for the following streams:
    "params", "dropout", "noise"
    """
    if params_seed is None:
        params_seed = int(np.random.randint(0, MAX_SEED, dtype=np.uint32))
    if dropout_seed is None:
        dropout_seed = int(np.random.randint(0, MAX_SEED, dtype=np.uint32))
    if noise_seed is None:
        noise_seed = int(np.random.randint(0, MAX_SEED, dtype=np.uint32))

    return nnx.Rngs(params=jax.random.key(params_seed),
                    dropout=jax.random.key(dropout_seed),
                    noise=jax.random.key(noise_seed))
