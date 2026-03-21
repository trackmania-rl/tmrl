# standard library imports
import contextlib
import functools
import inspect
import json
import operator
import os
import pickle
import signal
import subprocess
import threading
import weakref
from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import torch
from loguru import logger

T = TypeVar("T")  # helps with type inference in some editors


def pandas_dict(*args, **kwargs) -> pd.Series:
    return pd.Series(dict(*args, **kwargs), dtype=object)


def wandb_monotonic_step(proposed: int, run: Any | None = None) -> int:
    """Return a step value safe for ``wandb.log(..., step=...)``.

    After resuming a run, ``wandb.run.step`` can be ahead of local batch counters
    (e.g. ``total_updates`` / ``_training_step`` restored from an older checkpoint).
    Using ``max(proposed, run.step)`` keeps steps monotonically increasing.
    """
    p = int(proposed)
    if run is None:
        try:
            import wandb as _wandb

            run = _wandb.run
        except ImportError:
            return p
    if run is None:
        return p
    current = getattr(run, "step", None)
    if current is None:
        return p
    try:
        c = int(current)
    except (TypeError, ValueError):
        return p
    return max(p, c)


def shallow_copy[T](obj: T) -> T:
    x = type(obj).__new__(type(obj))
    vars(x).update(vars(obj))
    return x


# === collate, partition, etc =====================================================================


def collate_torch(batch: Sequence[Any], device: Any = None) -> Any:
    """Collate a batch of nested structures (numpy/tensor leaves) to batched torch tensors.

    Args:
        batch: Sequence of samples with the same nested structure.
        device: Optional device to move the collated tensors to.

    Returns:
        Single nested structure with leaves as batched torch tensors on device.
    """
    non_blocking = device is not None and (
        (isinstance(device, torch.device) and device.type == "cuda")
        or (isinstance(device, str) and device.startswith("cuda"))
    )
    elem = batch[0]
    pin_memory_threshold = 20_000
    if isinstance(elem, torch.Tensor):
        if elem.numel() < pin_memory_threshold:
            stacked = torch.stack(cast(list[torch.Tensor], batch))
            # Use pinned memory for async GPU transfers when non_blocking is enabled
            if non_blocking and device is not None:
                stacked = stacked.pin_memory()
            return stacked.to(device, non_blocking=non_blocking)
        else:
            # For large tensors, pin memory before transfer for async GPU copy
            if non_blocking and device is not None:
                batch = [b.contiguous().pin_memory() for b in batch]
            return torch.stack([b.to(device, non_blocking=non_blocking) for b in batch], 0)
    elif isinstance(elem, np.ndarray):
        if elem.dtype == np.object_:
            return collate_torch(tuple(torch.from_numpy(b) for b in batch), device)
        try:
            arr = np.stack(batch, axis=0)
        except ValueError:
            shapes = [b.shape for b in batch]
            unique_shapes = set(shapes)
            raise ValueError(
                "collate_torch: mismatched ndarray shapes in batch; "
                f"refusing to auto-pad to avoid silent data corruption. "
                f"unique_shapes={sorted(unique_shapes)}"
            ) from None
        tensor = torch.as_tensor(arr)
        # Use pinned memory for async GPU transfers when non_blocking is enabled
        if non_blocking and device is not None:
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=non_blocking)
    elif hasattr(elem, "__torch_tensor__"):
        tensors = [b.__torch_tensor__() for b in batch]
        if non_blocking and device is not None:
            tensors = [t.pin_memory() for t in tensors]
        return torch.stack([t.to(device, non_blocking=non_blocking) for t in tensors], 0)
    elif isinstance(elem, (list, tuple)):
        transposed = zip(*batch, strict=True)
        return type(elem)(collate_torch(samples, device) for samples in transposed)
    elif isinstance(elem, dict):
        return {key: collate_torch(tuple(d[key] for d in batch), device) for key in elem}
    else:
        tensor = torch.from_numpy(np.array(batch))
        # Use pinned memory for async GPU transfers when non_blocking is enabled
        if non_blocking and device is not None:
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=non_blocking)


class cached_property:  # noqa: N801
    """Property-like descriptor that caches the result of the getter.

    The value is keyed by instance id; when the instance is garbage collected
    the cache entry is removed. Useful for attributes that should not be pickled.
    """

    def __init__(self, init=None):
        self.cache = {}
        self.init = init

    def __get__(self, instance, owner):
        if id(instance) not in self.cache:
            if self.init is None:
                raise AttributeError()
            self.__set__(instance, self.init(instance))
        return self.cache[id(instance)][0]

    def __set__(self, instance, value):
        self.cache[id(instance)] = (
            value,
            weakref.ref(instance, functools.partial(self.cache.pop, id(instance))),
        )


def _partial_default():
    """Sentinel for tmrl.util.partial; not meant to be called directly."""
    raise ValueError("This is a dummy function and not meant to be called.")


def partial[T](
    func: type[T] | Callable[..., Any] = _partial_default, *args: Any, **kwargs: Any
) -> functools.partial[Any]:
    """
    Like `functools.partial`, except if used as a keyword argument for another `partial`
    and no function is supplied. Then, the outer `partial` will insert the appropriate
    default value as the function.
    """
    if func is not _partial_default:
        for k, v in kwargs.items():
            if isinstance(v, functools.partial) and v.func is _partial_default:
                kwargs[k] = partial(
                    inspect.signature(func).parameters[k].default, *v.args, **v.keywords
                )
    return functools.partial(func, *args, **kwargs)


FKEY = "+"


def partial_to_dict(partial_obj: functools.partial, version: str = "3") -> dict:
    """Serialize a partial (e.g. for wandb config). Supports nested partials and Gymnasium."""
    assert not partial_obj.args, "So far only keyword arguments are supported"
    fields = {k: v.default for k, v in inspect.signature(partial_obj.func).parameters.items()}
    fields = {k: v for k, v in fields.items() if v is not inspect.Parameter.empty}
    fields.update(partial_obj.keywords)
    nested = {k: partial_to_dict(partial(v), version="") for k, v in fields.items() if callable(v)}
    simple = {k: v for k, v in fields.items() if k not in nested}
    output = {
        FKEY: partial_obj.func.__module__ + ":" + partial_obj.func.__qualname__,
        **simple,
        **nested,
    }
    return dict(output, __format_version__=version) if version else output


def get_class_or_function(func):
    module, name = func.split(":")
    return getattr(import_module(module), name)


def partial_from_args(func: str | Callable[..., Any], kwargs: dict[str, str]):
    resolved: Callable[..., Any] = get_class_or_function(func) if isinstance(func, str) else func
    func = cast(Callable[..., Any], resolved)
    keys = {k.split(".")[0] for k in kwargs}
    keywords = {}
    for key in keys:
        params = inspect.signature(func).parameters
        assert key in params, (
            f"'{key}' is not a valid parameter of {func}. "
            f"Valid parameters are {tuple(params.keys())}."
        )
        param = params[key]
        value = kwargs.get(key, param.default)
        if param.annotation is type:
            sub_keywords = {
                k.split(".", 1)[1]: v for k, v in kwargs.items() if k.startswith(key + ".")
            }
            keywords[key] = partial_from_args(value, sub_keywords)
        elif param.annotation is bool:
            keywords[key] = value.lower() in ("true", "1", "yes")
        else:
            keywords[key] = param.annotation(value)
    return partial(func, **keywords)


def get_output(*args, default="", **kwargs):
    try:
        output = subprocess.check_output(*args, text=True, **kwargs)
        return output.rstrip("\n")  # skip trailing newlines as done in bash
    except subprocess.CalledProcessError:
        return default


def git_info(path=None):
    """
    Returns a dict with information about the git repo at path
    (path can be a sub-directory of the git repo).

    Args:
        path: The path to the git repo (default: the directory of the main module).

    Returns:
        A dict with information about the git repo.
    """
    import __main__

    path = path or os.path.dirname(__main__.__file__)
    rev = get_output(["git", "rev-parse", "HEAD"], cwd=path)
    count = int(get_output(["git", "rev-list", "HEAD", "--count"], default="-1", cwd=path))
    status = get_output(["git", "status", "--short"], cwd=path)  # shows un-committed modified files
    commit_date = get_output(
        [
            "git",
            "show",
            "--quiet",
            "--date=format-local:%Y-%m-%dT%H:%M:%SZ",
            "--format=%cd",
        ],
        cwd=path,
        env={"TZ": "UTC"},
    )
    desc = get_output(
        [
            "git",
            "describe",
            "--long",
            "--tags",
            "--dirty",
            "--always",
            "--match",
            r"v[0-9]*\.[0-9]*",
        ],
        cwd=path,
    )
    message = (
        desc
        + " "
        + " ".join(
            get_output(
                ["git", "log", "--oneline", "--format=%B", "-n", "1", "HEAD"],
                cwd=path,
            ).splitlines()
        )
    )

    url = get_output(["git", "config", "--get", "remote.origin.url"], cwd=path).strip()
    # if on github, change remote to a meaningful https url
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:") : -len(".git")] + "/commit/" + rev
    elif url.startswith("https://github.com"):
        url = url[: len(".git")] + "/commit/" + rev

    return {
        "url": url,
        "rev": rev,
        "count": count,
        "status": status,
        "desc": desc,
        "date": commit_date,
        "message": message,
    }


def dump(obj, path):
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")
    # DelayInterrupt uses signal.signal(), which only works in the main thread.
    ctx = (
        DelayInterrupt()
        if threading.current_thread() is threading.main_thread()
        else contextlib.nullcontext()
    )
    with ctx:
        with open(tmp_path, "wb") as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(d, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DelayInterrupt:
    """Catches SIGINT and SIGTERM and re-raises them after the context manager exits.

    Can be used in a context, e.g., `with DelayInterrupt():`
    """

    signal_received = False
    signals = (signal.SIGINT, signal.SIGTERM)

    def __enter__(self):
        self.default_handlers = [signal.getsignal(s) for s in self.signals]
        [signal.signal(s, self.on_signal) for s in self.signals]

    def on_signal(self, *args):
        logger.info("tmrl.util:DelayInterrupt -- Signal received! {}", args)
        self.signal_received = True

    def __exit__(self, *args):
        [signal.signal(s, d) for s, d in zip(self.signals, self.default_handlers, strict=True)]
        if self.signal_received:
            raise KeyboardInterrupt()


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)
