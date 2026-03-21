import jax.numpy as jnp
import jax
from flax import nnx


__docformat__ = "google"


def collate_jax(batch, device=None):
    """Turns a batch of nested structures with numpy arrays as leaves into a single element of the same nested structure with batched jax arrays as leaves"""
    collated = jax.tree.map(lambda *xs: jnp.stack(xs), *batch)
    if device is not None:
        collated = jax.device_put(collated, device)
    return collated


# === randomness =======================================================================================================

def default_rngs():
    """
    Return an RNG with default seeds for the following streams:
    "params", "dropout", "noise"
    """
    return nnx.Rngs(params=jax.random.key(0),
                    dropout=jax.random.key(1),
                    noise=jax.random.key(3))
