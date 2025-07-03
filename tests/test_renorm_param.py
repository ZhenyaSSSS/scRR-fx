"""Дополнительные параметризованные тесты для renormalize.
Покрывают большие значения k (до 1024) и проверяют сохранение суммы.
"""

import random

import torch
from mpmath import mp
import pytest

from scrr_fx import renormalize
from tests.helpers import _mp_sum

mp.dps = 100


def _mp_sum(t: torch.Tensor):
    return mp.fsum(mp.mpf(x.item()) for x in t.flatten())


@pytest.mark.parametrize("k", [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_renormalize_preserves_sum_various_k(k):
    # Случайная длина dirty между k и 2k
    m = random.randint(k, k * 2)
    dirty = torch.randn(3, m, dtype=torch.float64)

    original_sum = _mp_sum(dirty)
    clean = renormalize(dirty, k)
    new_sum = _mp_sum(clean)

    assert clean.shape[-1] == k
    
    # Проверяем, что разница находится в разумных пределах
    # Для k=1 допуск чуть мягче, т.к. renormalize почти ничего не делает
    tolerance = 1e-14 if k == 1 else 1e-15
    if original_sum != 0:
        assert mp.fabs((original_sum - new_sum) / original_sum) < tolerance
    else:
        assert mp.fabs(new_sum) < tolerance

    # Проверяем нули в хвосте, если m < k (pad case)
    if m < k:
        assert torch.count_nonzero(clean[..., m:]) == 0

    # Проверяем форму
    assert clean.shape[-1] == k

    # Проверяем нули в хвосте, если m < k (pad case)
    if m < k:
        assert torch.count_nonzero(clean[..., m:]) == 0
    # residual length should be max(0, m - k) -> ЭТОТ ТЕСТ НЕВЕРНЫЙ. renormalize всегда возвращает k.
    # assert clean.shape[-1] == max(0, m - k) 