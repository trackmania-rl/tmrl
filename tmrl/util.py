# standard library imports
import functools
import operator
import inspect
import io
import json
import os
import pickle
import signal
import subprocess
import weakref
from pathlib import Path
# from contextlib import contextmanager
# from dataclasses import Field, dataclass, fields, is_dataclass, make_dataclass
from importlib import import_module
# from itertools import chain
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Type, TypeVar, Union
# from weakref import WeakKeyDictionary

# third-party imports
import numpy as np
import pandas as pd
import logging
import torch

T = TypeVar('T')  # helps with type inference in some editors


def pandas_dict(*args, **kwargs) -> pd.Series:
    return pd.Series(dict(*args, **kwargs), dtype=object)


def shallow_copy(obj: T) -> T:
    x = type(obj).__new__(type(obj))
    vars(x).update(vars(obj))
    return x


# === collate, partition, etc ==========================================================================================

def collate_torch(batch, device=None):
    """Turns a batch of nested structures with numpy arrays as leaves into into a single element of the same nested structure with batched torch tensors as leaves"""
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        # return torch.stack(batch, 0).to(device, non_blocking=non_blocking)
        if elem.numel() < 20000:  # TODO: link to the relevant profiling that lead to this threshold
            return torch.stack(batch).to(device)
        else:
            return torch.stack([b.contiguous().to(device) for b in batch], 0)
    elif isinstance(elem, np.ndarray):
        return collate_torch(tuple(torch.from_numpy(b) for b in batch), device)
    elif hasattr(elem, '__torch_tensor__'):
        return torch.stack([b.__torch_tensor__().to(device) for b in batch], 0)
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return type(elem)(collate_torch(samples, device) for samples in transposed)
    elif isinstance(elem, Mapping):
        return type(elem)((key, collate_torch(tuple(d[key] for d in batch), device)) for key in elem)
    else:
        return torch.from_numpy(np.array(batch)).to(device)  # we create a numpy array first to work around https://github.com/pytorch/pytorch/issues/24200


# === catched property =================================================================================================


# noinspection PyPep8Naming
class cached_property:
    """Similar to `property` but after calling the getter/init function the result is cached.
    It can be used to create object attributes that aren't stored in the object's __dict__.
    This is useful if we want to exclude certain attributes from being pickled."""
    def __init__(self, init=None):
        self.cache = {}
        self.init = init

    def __get__(self, instance, owner):
        if id(instance) not in self.cache:
            if self.init is None: raise AttributeError()
            self.__set__(instance, self.init(instance))
        return self.cache[id(instance)][0]

    def __set__(self, instance, value):
        # Cache the attribute value based on the instance id. If instance is garbage collected its cached value is removed.
        self.cache[id(instance)] = (value, weakref.ref(instance, functools.partial(self.cache.pop, id(instance))))


# === partial ==========================================================================================================
def default():
    raise ValueError("This is a dummy function and not meant to be called.")


def partial(func: Type[T] = default, *args, **kwargs) -> Union[T, Type[T]]:
    """Like `functools.partial`, except if used as a keyword argument for another `partial` and no function is supplied.
     Then, the outer `partial` will insert the appropriate default value as the function. """

    if func is not default:
        for k, v in kwargs.items():
            if isinstance(v, functools.partial) and v.func is default:
                kwargs[k] = partial(inspect.signature(func).parameters[k].default, *v.args, **v.keywords)
    return functools.partial(func, *args, **kwargs)


FKEY = '+'


def partial_to_dict(p: functools.partial, version="3"):
    """
    Only for wandb.

    This function has become lenient to work with Gymnasium.
    """
    assert not p.args, "So far only keyword arguments are supported, here"
    fields = {k: v.default for k, v in inspect.signature(p.func).parameters.items()}
    fields = {k: v for k, v in fields.items() if v is not inspect.Parameter.empty}
    # diff = p.keywords.keys() - fields.keys()
    # assert not diff, f"{p} cannot be converted to dict. There are invalid keywords present: {diff}"
    fields.update(p.keywords)
    nested = {k: partial_to_dict(partial(v), version="") for k, v in fields.items() if callable(v)}
    simple = {k: v for k, v in fields.items() if k not in nested}
    output = {FKEY: p.func.__module__ + ":" + p.func.__qualname__, **simple, **nested}
    return dict(output, __format_version__=version) if version else output


# def partial_from_dict(d: dict):
#     d = d.copy()
#     assert d.pop("__format_version__", "3") == "3"
#     d = {k: partial_from_dict(v) if isinstance(v, dict) and FKEY in v else v for k, v in d.items()}
#     func = get_class_or_function(d.pop(FKEY) or "tmrl.util:default")
#     return partial(func, **d)


def get_class_or_function(func):
    module, name = func.split(":")
    return getattr(import_module(module), name)


def partial_from_args(func: Union[str, callable], kwargs: Dict[str, str]):
    # logging.info(func, kwargs)  # useful to visualize the parsing process
    func = get_class_or_function(func) if isinstance(func, str) else func
    keys = {k.split('.')[0] for k in kwargs}
    keywords = {}
    for key in keys:
        params = inspect.signature(func).parameters
        assert key in params, f"'{key}' is not a valid parameter of {func}. Valid parameters are {tuple(params.keys())}."
        param = params[key]
        value = kwargs.get(key, param.default)
        if param.annotation is type:
            sub_keywords = {k.split('.', 1)[1]: v for k, v in kwargs.items() if k.startswith(key + '.')}
            keywords[key] = partial_from_args(value, sub_keywords)
        elif param.annotation is bool:
            keywords[key] = bool(eval(value))  # because bool('False') will evaluate to True (it's a non-empty string).
        else:
            keywords[key] = param.annotation(value)
    return partial(func, **keywords)


# === git ==============================================================================================================


def get_output(*args, default='', **kwargs):
    try:
        output = subprocess.check_output(*args, universal_newlines=True, **kwargs)
        return output.rstrip("\n")  # skip trailing newlines as done in bash
    except subprocess.CalledProcessError:
        return default


def git_info(path=None):
    """returns a dict with information about the git repo at path (path can be a sub-directory of the git repo)
    """
    # third-party imports
    import __main__
    path = path or os.path.dirname(__main__.__file__)
    rev = get_output('git rev-parse HEAD'.split(), cwd=path)
    count = int(get_output('git rev-list HEAD --count'.split(), default='-1', cwd=path))
    status = get_output('git status --short'.split(), cwd=path)  # shows un-committed modified files
    commit_date = get_output("git show --quiet --date=format-local:%Y-%m-%dT%H:%M:%SZ --format=%cd".split(), cwd=path, env=dict(TZ='UTC'))
    desc = get_output(['git', 'describe', '--long', '--tags', '--dirty', '--always', '--match', r'v[0-9]*\.[0-9]*'], cwd=path)
    message = desc + " " + ' '.join(get_output(['git', 'log', '--oneline', '--format=%B', '-n', '1', "HEAD"], cwd=path).splitlines())

    url = get_output('git config --get remote.origin.url'.split(), cwd=path).strip()
    # if on github, change remote to a meaningful https url
    if url.startswith('git@github.com:'):
        url = 'https://github.com/' + url[len('git@github.com:'):-len('.git')] + '/commit/' + rev
    elif url.startswith('https://github.com'):
        url = url[:len('.git')] + '/commit/' + rev

    return dict(url=url, rev=rev, count=count, status=status, desc=desc, date=commit_date, message=message)


# === serialization ====================================================================================================


def dump(obj, path):
    path = Path(path)
    tmp_path = path.with_suffix('.tmp')
    with DelayInterrupt():  # Continue to save even if SIGINT or SIGTERM is sent and raise KeyboardInterrupt afterwards.
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)  # dump temporary file (can fail)
        os.replace(tmp_path, path)  # replace with definitive name (this is supposedly atomic)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(d, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# === signal handling ==================================================================================================


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
        logging.info(f"tmrl.util:DelayInterrupt -- Signal received!", *args)
        self.signal_received = True

    def __exit__(self, *args):
        [signal.signal(s, d) for s, d in zip(self.signals, self.default_handlers)]
        if self.signal_received:
            raise KeyboardInterrupt()


# === operations =======================================================================================================


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)
