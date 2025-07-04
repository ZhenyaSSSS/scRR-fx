"""Тесты для алгоритма ренормализации (фаза 2)."""

import torch
from mpmath import mp
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np

from scrr_fx._renorm import renormalize, fast_two_sum_reduction
from scrr_fx._tensor import SCRR_Tensor
from tests.helpers import _mp_sum, to_scrr

mp.dps = 200  # Устанавливаем высокую точность для mpmath


def test_fast_two_sum_reduction_exactness():
    """
    Проверяет, что fast_two_sum_reduction точно сохраняет сумму.
    sum(original) == head + sum(tail)
    """
    original_tensor = torch.randn(10, 20, dtype=torch.float64)
    original_sum = _mp_sum(original_tensor)

    head, tail = fast_two_sum_reduction(original_tensor, dim=-1)

    # Сумма после редукции должна быть абсолютно такой же
    reduced_sum = _mp_sum(head) + _mp_sum(tail)

    assert mp.almosteq(original_sum, reduced_sum)


@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("m", [1, 3, 15, 16]) # m - число "грязных" компонент
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
def test_renormalize_shape_and_sum_preservation(k, m, batch_shape):
    """
    Проверяет, что renormalize:
    1. Возвращает тензор правильной формы (..., k).
    2. Сохраняет сумму исходных компонентов.
    """
    dirty_shape = batch_shape + (m,)
    dirty = torch.randn(dirty_shape, dtype=torch.float64)
    
    original_sum_mp = _mp_sum(dirty)

    clean = renormalize(dirty, k, dim=-1)
    
    # 1. Проверка формы
    expected_shape = batch_shape + (k,)
    assert clean.shape == expected_shape

    # 2. Проверка сохранения суммы
    new_sum_mp = _mp_sum(clean)
    
    # Суммы должны быть равны с очень высокой точностью
    # Допускаем небольшую относительную погрешность из-за единственной неточной операции sum() в конце
    assert mp.almosteq(original_sum_mp, new_sum_mp, rel_eps=1e-15, abs_eps=1e-300)


@pytest.mark.parametrize(
    "data_generator",
    [
        # Большой разброс порядков
        lambda: torch.randn(5, 20, dtype=torch.float64) * 1e100 + torch.randn(5, 20, dtype=torch.float64) * 1e-100,
        # Случай, приводящий к сокращению разрядов
        lambda: torch.tensor([1.0, 1e20, 1e-20, -1.0, -1e20, -1e-20] * 3, dtype=torch.float64),
        # Все нули
        lambda: torch.zeros(5, 10, dtype=torch.float64),
    ],
    ids=["wide_range", "cancellation", "zeros"]
)
def test_renormalize_edge_cases(data_generator):
    """Проверяет renormalize на крайних случаях."""
    dirty = data_generator()
    k = 4
    
    original_sum = _mp_sum(dirty)
    clean = renormalize(dirty, k)
    
    assert not torch.isnan(clean).any()
    assert not torch.isinf(clean).any()
    
    new_sum = _mp_sum(clean)
    
    assert mp.almosteq(original_sum, new_sum, rel_eps=1e-15)


def test_scrr_tensor_from_dirty():
    dirty = torch.randn(5, 15, dtype=torch.float64)
    k = 3
    scrr = SCRR_Tensor.from_dirty(dirty, k)

    assert scrr.precision_k == k
    # Проверяем, что сумма компонентов совпадает с исходной суммой
    assert torch.allclose(scrr.value(), dirty.sum(dim=-1))

# Этот тест больше не нужен, т.к. renormalize теперь всегда возвращает один тензор
# def test_renormalize_edge_zero_padding():
#     dirty = torch.randn(2, 1, dtype=torch.float64)  # лишь один компонент
#     k = 4
#     clean, residual = renormalize(dirty, k)
#     assert clean.shape[-1] == k
#     # Последние k-1 компонентов должны быть нули
#     assert torch.count_nonzero(clean[..., 1:]) == 0
#     assert residual.shape[-1] == 0 

@given(
    batch_shape=st.lists(st.integers(1, 4), min_size=0, max_size=2),
    m=st.integers(2, 20),
    k=st.integers(2, 8),
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=64), min_size=2, max_size=100)
)
@settings(max_examples=50)
def test_renormalize_property(batch_shape, m, k, values):
    shape = tuple(batch_shape) + (m,)
    if np.prod(shape) > len(values):
        return  # skip oversize
    arr = np.array(values[:int(np.prod(shape))], dtype=np.float64).reshape(shape)
    dirty = torch.tensor(arr)
    original_sum = _mp_sum(dirty)
    clean = renormalize(dirty, k, dim=-1)
    new_sum = _mp_sum(clean)
    assert mp.almosteq(original_sum, new_sum, rel_eps=1e-13) 