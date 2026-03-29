import numpy as np
import torch
from tmrl.util import collate_torch, partial, shallow_copy


def test_shallow_copy_copies_instance_attrs():
    class Box:
        def __init__(self):
            self.x = 1

    b = Box()
    c = shallow_copy(b)
    assert c is not b
    assert c.x == 1
    c.x = 2
    assert b.x == 1


def test_collate_torch_numpy_stacks_to_device():
    batch = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    out = collate_torch(batch, device="cpu")
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_collate_torch_nested_tuple():
    a = (torch.tensor([1.0]), torch.tensor([2.0]))
    b = (torch.tensor([3.0]), torch.tensor([4.0]))
    out = collate_torch([a, b], device="cpu")
    assert len(out) == 2
    assert out[0].shape == (2, 1)
    assert out[1].shape == (2, 1)


def test_partial_wraps_like_functools():
    def f(a, b=0):
        return a + b

    g = partial(f, b=3)
    assert g(2) == 5
