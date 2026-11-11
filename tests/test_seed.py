import random

import numpy as np
import torch

from train.seed import seed_everything


def test_seed_everything_is_repeatable() -> None:
    seed_everything(123)
    py_a = [random.random() for _ in range(4)]
    np_a = np.random.rand(4)
    th_a = torch.rand(4)

    seed_everything(123)
    py_b = [random.random() for _ in range(4)]
    np_b = np.random.rand(4)
    th_b = torch.rand(4)

    assert py_a == py_b
    assert np.allclose(np_a, np_b)
    assert torch.equal(th_a, th_b)
