import math
import itertools

import torch
import pytest
from mpmath import mp
import hypothesis.strategies as st
from hypothesis import given, settings

from scrr_fx import two_prod, two_sum

mp.dps = 100  # высокая точность для ground-truth


def _to_mp(t: torch.Tensor):
    """Преобразует тензор в плоский список mpmath-чисел."""
    return [mp.mpf(x.item()) for x in t.flatten()]


@pytest.mark.parametrize(
    "a, b",
    [
        # Базовый случай: случайные тензоры
        (torch.randn(10, dtype=torch.float64), torch.randn(10, dtype=torch.float64)),
        # Нули
        (torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64)),
        # Катастрофическое сокращение (сумма близка к нулю)
        (torch.tensor([1.0], dtype=torch.float64), torch.tensor([1e-16], dtype=torch.float64)),
        (torch.tensor([1e20], dtype=torch.float64), torch.tensor([-1e20 + 1], dtype=torch.float64)),
        # Неконечные числа
        (torch.tensor([float("inf")], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64)),
        (torch.tensor([float("inf")], dtype=torch.float64), torch.tensor([float("-inf")], dtype=torch.float64)),
        # Broadcasting
        (torch.randn(5, 1, dtype=torch.float64), torch.randn(1, 5, dtype=torch.float64)),
    ],
    ids=["random", "zeros", "cancellation_small", "cancellation_large", "inf", "inf-inf_is_nan", "broadcast"]
)
def test_two_sum(a, b):
    """Проверяет точность two_sum на различных входных данных."""
    s, e = two_sum(a, b)
    
    expected_sum = a + b
    
    # Отдельно обрабатываем NaN, т.к. a+b может дать nan, где s,e - нет
    nan_mask = torch.isnan(expected_sum)
    if nan_mask.any():
        assert torch.isnan(s[nan_mask]).all()
        # Ошибка `e` в случае inf-inf может быть 0 или nan, оба варианта корректны
        assert (torch.isnan(e[nan_mask]) | (e[nan_mask] == 0)).all()

    # Проверяем неконечные случаи: ошибка должна быть 0
    inf_mask = torch.isinf(expected_sum)
    if inf_mask.any():
        assert torch.all(e[inf_mask] == 0)

    # Проверяем конечные случаи с высокой точностью
    finite_mask = torch.isfinite(expected_sum)
    if not finite_mask.any():
        return

    a_bc, b_bc = torch.broadcast_tensors(a, b)
    s_bc, e_bc = torch.broadcast_tensors(s, e)
    
    # Проверяем только конечные элементы
    a_mp = _to_mp(a_bc[finite_mask])
    b_mp = _to_mp(b_bc[finite_mask])
    s_mp = _to_mp(s_bc[finite_mask])
    e_mp = _to_mp(e_bc[finite_mask])
    
    for ga, gb, gs, ge in zip(a_mp, b_mp, s_mp, e_mp):
        assert mp.almosteq(ga + gb, gs + ge)


@pytest.mark.parametrize(
    "a, b",
    [
        # Базовый случай: случайные тензоры
        (torch.randn(10, dtype=torch.float64), torch.randn(10, dtype=torch.float64)),
        # Нули
        (torch.tensor([0.0], dtype=torch.float64), torch.tensor([123.456], dtype=torch.float64)),
        # Очень большие/маленькие числа
        (torch.tensor([1e154], dtype=torch.float64), torch.tensor([1e154], dtype=torch.float64)),
        (torch.tensor([1e-154], dtype=torch.float64), torch.tensor([1e-154], dtype=torch.float64)),
        # Неконечные числа
        (torch.tensor([float("-inf")], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64)),
        (torch.tensor([float("inf")], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64)), # inf * 0 = nan
        # Broadcasting
        (torch.randn(1, 5, dtype=torch.float64), torch.randn(5, 1, dtype=torch.float64)),
    ],
    ids=["random", "zeros", "large_values", "small_values", "inf", "inf*0_is_nan", "broadcast"]
)
def test_two_prod(a, b):
    """Проверяет точность two_prod на различных входных данных."""
    p, e = two_prod(a, b)

    expected_prod = a * b
    
    # Обработка NaN
    nan_mask = torch.isnan(expected_prod)
    if nan_mask.any():
        assert torch.isnan(p[nan_mask]).all()
        assert torch.isnan(e[nan_mask]).all()

    # Проверяем неконечные случаи
    inf_mask = torch.isinf(expected_prod)
    if inf_mask.any():
        assert torch.all(e[inf_mask] == 0)

    # Проверяем конечные случаи с высокой точностью
    finite_mask = torch.isfinite(expected_prod)
    if not finite_mask.any():
        return
        
    a_bc, b_bc = torch.broadcast_tensors(a, b)
    p_bc, e_bc = torch.broadcast_tensors(p, e)

    a_mp = _to_mp(a_bc[finite_mask])
    b_mp = _to_mp(b_bc[finite_mask])
    p_mp = _to_mp(p_bc[finite_mask])
    e_mp = _to_mp(e_bc[finite_mask])

    for ga, gb, gp, ge in zip(a_mp, b_mp, p_mp, e_mp):
        assert mp.almosteq(ga * gb, gp + ge)


@given(
    arrs=st.lists(st.tuples(st.floats(allow_nan=False, allow_infinity=False, width=64),
                             st.floats(allow_nan=False, allow_infinity=False, width=64)), min_size=1, max_size=20)
)
@settings(max_examples=100)
def test_two_sum_property(arrs):
    import numpy as np
    a = torch.tensor(np.array([x for x, _ in arrs], dtype=np.float64))
    b = torch.tensor(np.array([y for _, y in arrs], dtype=np.float64))
    s, e = two_sum(a, b)
    # skip if any result is not finite
    if not (torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(s).all() and torch.isfinite(e).all()):
        return
    for ga, gb, gs, ge in zip(_to_mp(a), _to_mp(b), _to_mp(s), _to_mp(e)):
        if not (mp.isfinite(ga) and mp.isfinite(gb) and mp.isfinite(gs) and mp.isfinite(ge)):
            continue
        assert mp.almosteq(ga + gb, gs + ge)


@given(
    arrs=st.lists(st.tuples(st.floats(allow_nan=False, allow_infinity=False, width=64),
                             st.floats(allow_nan=False, allow_infinity=False, width=64)), min_size=1, max_size=20)
)
@settings(max_examples=100)
def test_two_prod_property(arrs):
    import numpy as np
    a = torch.tensor(np.array([x for x, _ in arrs], dtype=np.float64))
    b = torch.tensor(np.array([y for _, y in arrs], dtype=np.float64))
    p, e = two_prod(a, b)
    # skip if any result is not finite
    if not (torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(p).all() and torch.isfinite(e).all()):
        return
    for ga, gb, gp, ge in zip(_to_mp(a), _to_mp(b), _to_mp(p), _to_mp(e)):
        if not (mp.isfinite(ga) and mp.isfinite(gb) and mp.isfinite(gp) and mp.isfinite(ge)):
            continue
        # Для EFT требуем абсолютную точность
        assert ga * gb == gp + ge 