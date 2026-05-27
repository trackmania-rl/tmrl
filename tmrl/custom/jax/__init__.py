# This code block sets TLS-Python-Object processes to be spawned rather than forked.
# Otherwise, JAX outputs a warning saying that it is not compatible with the fork method.
import multiprocessing as mp
try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass
