from typing import Sequence, Mapping

import numpy as np
import torch


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


def concat_collated(batches):
    """Concatenates a list of already-collated batches (with torch tensors as leaves) along the batch dimension."""
    elem = batches[0]
    if isinstance(elem, torch.Tensor):
        return torch.cat(batches, dim=0)
    elif hasattr(elem, '__torch_tensor__'):
        return torch.cat([b.__torch_tensor__() for b in batches], dim=0)
    elif isinstance(elem, Sequence):
        transposed = zip(*batches)
        return type(elem)(concat_collated(samples) for samples in transposed)
    elif isinstance(elem, Mapping):
        return type(elem)((key, concat_collated([d[key] for d in batches])) for key in elem)
    else:
        raise TypeError(f"Unsupported type: {type(elem)}")
