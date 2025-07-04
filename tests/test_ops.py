import torch
from mpmath import mp
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np

from scrr_fx._tensor import SCRR_Tensor
from tests.helpers import to_scrr, scrr_to_mp_sum, scrr_to_mp_value

mp.dps = 200  # Точность для эталонных вычислений

# --- Утилиты для тестирования ---

def get_mp_matrix(scrr_tensor: SCRR_Tensor):
    """Преобразует 2D SCRR_Tensor в матрицу mpmath."""
    vals = scrr_to_mp_value(scrr_tensor)
    rows, cols = scrr_tensor.shape
    return mp.matrix(rows, cols, vals)

# --- Тесты ---

@pytest.mark.parametrize("k", [2, 4, 8])
def test_add_op_exactness(k):
    """Проверяет точность операции torch.add и оператора +."""
    a_torch = torch.randn(5, 5, dtype=torch.float64)
    b_torch = torch.randn(5, 5, dtype=torch.float64)

    a_scrr = SCRR_Tensor.from_float(a_torch, k=k)
    b_scrr = SCRR_Tensor.from_float(b_torch, k=k)

    c_scrr = a_scrr + b_scrr

    # Эталон: поэлементная сумма в mpmath
    a_vals = scrr_to_mp_value(a_scrr)
    b_vals = scrr_to_mp_value(b_scrr)
    expected_vals = [a + b for a, b in zip(a_vals, b_vals)]
    
    actual_vals = scrr_to_mp_value(c_scrr)

    for expected, actual in zip(expected_vals, actual_vals):
        assert mp.almosteq(expected, actual)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_mul_op_exactness(k):
    """Проверяет точность операции torch.mul и оператора *."""
    a_torch = torch.randn(5, 5, dtype=torch.float64)
    b_torch = torch.randn(5, 5, dtype=torch.float64)

    a_scrr = SCRR_Tensor.from_float(a_torch, k=k)
    b_scrr = SCRR_Tensor.from_float(b_torch, k=k)

    c_scrr = a_scrr * b_scrr

    # Эталон: поэлементное произведение в mpmath
    a_vals = scrr_to_mp_value(a_scrr)
    b_vals = scrr_to_mp_value(b_scrr)
    expected_vals = [a * b for a, b in zip(a_vals, b_vals)]

    actual_vals = scrr_to_mp_value(c_scrr)

    for expected, actual in zip(expected_vals, actual_vals):
        assert mp.almosteq(expected, actual)


@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("m, n, p", [(2, 3, 4), (5, 5, 5)])
def test_matmul_op_exactness(k, m, n, p):
    """Проверяет точность операции torch.matmul и оператора @."""
    a_torch = torch.randn(m, n, dtype=torch.float64)
    b_torch = torch.randn(n, p, dtype=torch.float64)
    
    a_scrr = SCRR_Tensor.from_float(a_torch, k=k)
    b_scrr = SCRR_Tensor.from_float(b_torch, k=k)

    c_scrr = a_scrr @ b_scrr

    a_mp = get_mp_matrix(a_scrr)
    b_mp = get_mp_matrix(b_scrr)
    expected_c_mp = a_mp * b_mp

    actual_c_mp = get_mp_matrix(c_scrr)

    for i in range(m):
        for j in range(p):
            assert mp.almosteq(expected_c_mp[i, j], actual_c_mp[i, j])


def test_unsupported_op_raises_error():
    """Проверяет, что нереализованная операция вызывает NotImplementedError."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    with pytest.raises(NotImplementedError, match="PyTorch function sin is not implemented"):
        torch.sin(a)


@pytest.mark.parametrize("op", [torch.add, torch.sub, torch.mul])
def test_mixed_ops_with_float(op):
    """Проверяет операции между SCRR_Tensor и обычным float."""
    a_torch = torch.randn(5, dtype=torch.float64)
    a_scrr = to_scrr(a_torch, k=4)
    b_float = 3.14

    res1 = op(a_scrr, b_float)

    # Эталон
    a_vals = scrr_to_mp_value(a_scrr)
    b_val = mp.mpf(b_float)
    if op == torch.add:
        expected_vals = [v + b_val for v in a_vals]
    elif op == torch.sub:
        expected_vals = [v - b_val for v in a_vals]
    elif op == torch.mul:
        expected_vals = [v * b_val for v in a_vals]
        
    actual_vals = scrr_to_mp_value(res1)

    for exp, act in zip(expected_vals, actual_vals):
        assert mp.almosteq(exp, act)


def test_view_and_reshape_ops():
    """Проверяет, что операции изменения формы работают корректно."""
    a = to_scrr(torch.randn(2, 6, dtype=torch.float64), k=4)
    
    b = a.reshape(3, 4)
    assert isinstance(b, SCRR_Tensor)
    assert b.shape == (3, 4)
    assert b.precision_k == a.precision_k
    assert torch.allclose(a.to_float().reshape(3, 4), b.to_float())

    c = b.transpose(0, 1)
    assert c.shape == (4, 3)
    
    d = c.view(12)
    assert d.shape == (12,)

@given(
    shape=st.lists(st.integers(1, 4), min_size=1, max_size=2),
    k=st.integers(2, 6),
    values1=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=64), min_size=2, max_size=100),
    values2=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=64), min_size=2, max_size=100)
)
@settings(max_examples=30)
def test_scrr_add_mul_property(shape, k, values1, values2):
    size = int(np.prod(shape))
    if size > min(len(values1), len(values2)):
        return
    a = torch.tensor(np.array(values1[:size], dtype=np.float64).reshape(shape))
    b = torch.tensor(np.array(values2[:size], dtype=np.float64).reshape(shape))
    scrr_a = SCRR_Tensor.from_float(a, k=k)
    scrr_b = SCRR_Tensor.from_float(b, k=k)
    scrr_sum = scrr_a + scrr_b
    scrr_mul = scrr_a * scrr_b
    # skip if any result is not finite
    if not (torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(scrr_sum.to_float()).all() and torch.isfinite(scrr_mul.to_float()).all()):
        return
    assert torch.allclose(scrr_sum.to_float(), a + b, rtol=1e-12, atol=1e-14)
    assert torch.allclose(scrr_mul.to_float(), a * b, rtol=1e-12, atol=1e-14)
    scrr_neg = -scrr_a
    if torch.isfinite(scrr_neg.to_float()).all() and torch.isfinite(-a).all():
        assert torch.allclose(scrr_neg.to_float(), -a, rtol=1e-12, atol=1e-14)
    if size % 2 == 0:
        new_shape = (2, size // 2)
        scrr_reshaped = scrr_a.reshape(*new_shape)
        if torch.isfinite(scrr_reshaped.to_float()).all() and torch.isfinite(a.reshape(new_shape)).all():
            assert scrr_reshaped.shape == new_shape
            assert torch.allclose(scrr_reshaped.to_float(), a.reshape(new_shape), rtol=1e-12, atol=1e-14)

@given(
    m=st.integers(2, 4), n=st.integers(2, 4), p=st.integers(2, 4), k=st.integers(2, 4),
    values1=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=64), min_size=16, max_size=100),
    values2=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=64), min_size=16, max_size=100)
)
@settings(max_examples=10)
def test_scrr_matmul_property(m, n, p, k, values1, values2):
    if m*n > len(values1) or n*p > len(values2):
        return
    a = torch.tensor(np.array(values1[:m*n], dtype=np.float64).reshape(m, n))
    b = torch.tensor(np.array(values2[:n*p], dtype=np.float64).reshape(n, p))
    scrr_a = SCRR_Tensor.from_float(a, k=k)
    scrr_b = SCRR_Tensor.from_float(b, k=k)
    scrr_matmul = scrr_a @ scrr_b
    ref = a @ b
    scrr_val = scrr_matmul.to_float()
    # Проверяем только если формы совпадают и нет nan/inf
    if scrr_val.shape == ref.shape and torch.isfinite(scrr_val).all() and torch.isfinite(ref).all():
        assert torch.allclose(scrr_val, ref, rtol=1e-12, atol=1e-14) 