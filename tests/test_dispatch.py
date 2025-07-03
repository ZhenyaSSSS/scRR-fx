"""Тесты для __torch_dispatch__ механизма (фаза 3)."""

import torch
import pytest
import mpmath as mp

from scrr_fx._tensor import SCRR_Tensor, HANDLED_FUNCTIONS
from tests.helpers import to_scrr, scrr_to_mp_value

mp.dps = 200


def test_dispatch_registration():
    """Проверяет, что все операции зарегистрированы в HANDLED_FUNCTIONS."""
    required_ops = [torch.add, torch.sub, torch.mul, torch.matmul, torch.neg]
    
    for op in required_ops:
        assert op in HANDLED_FUNCTIONS, f"Operation {op.__name__} not registered"


def test_dispatch_interception():
    """Проверяет, что __torch_dispatch__ перехватывает операции."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    b = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    
    # Эти операции должны перехватываться __torch_dispatch__
    c1 = torch.add(a, b)
    c2 = a + b  # Должно использовать __add__
    
    assert isinstance(c1, SCRR_Tensor)
    assert isinstance(c2, SCRR_Tensor)
    assert torch.allclose(c1.to_float(), c2.to_float())


def test_dispatch_unsupported_operation():
    """Проверяет, что неподдерживаемые операции вызывают NotImplementedError."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    
    # torch.sin не реализован
    with pytest.raises(NotImplementedError, match="PyTorch function sin is not implemented"):
        torch.sin(a)
    
    # torch.cos не реализован
    with pytest.raises(NotImplementedError, match="PyTorch function cos is not implemented"):
        torch.cos(a)


def test_dispatch_mixed_types():
    """Проверяет диспатчинг с смешанными типами (SCRR_Tensor + torch.Tensor)."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    b = torch.randn(5, dtype=torch.float64)
    
    # SCRR_Tensor + torch.Tensor
    c1 = torch.add(a, b)
    assert isinstance(c1, SCRR_Tensor)
    
    # torch.Tensor + SCRR_Tensor
    c2 = torch.add(b, a)
    assert isinstance(c2, SCRR_Tensor)
    
    # Результаты должны быть одинаковыми
    assert torch.allclose(c1.to_float(), c2.to_float())


def test_dispatch_scalar_operations():
    """Проверяет диспатчинг со скалярными значениями."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    scalar = 3.14
    
    # SCRR_Tensor + scalar
    c1 = torch.add(a, scalar)
    assert isinstance(c1, SCRR_Tensor)
    
    # scalar + SCRR_Tensor
    c2 = torch.add(scalar, a)
    assert isinstance(c2, SCRR_Tensor)
    
    # Результаты должны быть одинаковыми
    assert torch.allclose(c1.to_float(), c2.to_float())


def test_dispatch_matmul():
    """Проверяет диспатчинг матричного умножения."""
    a = to_scrr(torch.randn(3, 4, dtype=torch.float64), k=4)
    b = to_scrr(torch.randn(4, 2, dtype=torch.float64), k=4)
    
    # torch.matmul
    c1 = torch.matmul(a, b)
    assert isinstance(c1, SCRR_Tensor)
    
    # Оператор @
    c2 = a @ b
    assert isinstance(c2, SCRR_Tensor)
    
    # Результаты должны быть одинаковыми
    assert torch.allclose(c1.to_float(), c2.to_float())


def test_dispatch_neg():
    """Проверяет диспатчинг унарного минуса."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    
    # torch.neg
    c1 = torch.neg(a)
    assert isinstance(c1, SCRR_Tensor)
    
    # Оператор -
    c2 = -a
    assert isinstance(c2, SCRR_Tensor)
    
    # Результаты должны быть одинаковыми
    assert torch.allclose(c1.to_float(), c2.to_float())
    
    # Проверяем, что результат действительно отрицательный
    assert torch.allclose(c1.to_float(), -a.to_float())


def test_dispatch_sub():
    """Проверяет диспатчинг вычитания."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    b = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    
    # torch.sub
    c1 = torch.sub(a, b)
    assert isinstance(c1, SCRR_Tensor)
    
    # Оператор -
    c2 = a - b
    assert isinstance(c2, SCRR_Tensor)
    
    # Результаты должны быть одинаковыми
    assert torch.allclose(c1.to_float(), c2.to_float())


def test_dispatch_mul():
    """Проверяет диспатчинг умножения."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    b = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    
    # torch.mul
    c1 = torch.mul(a, b)
    assert isinstance(c1, SCRR_Tensor)
    
    # Оператор *
    c2 = a * b
    assert isinstance(c2, SCRR_Tensor)
    
    # Результаты должны быть одинаковыми
    assert torch.allclose(c1.to_float(), c2.to_float())


def test_dispatch_preserves_precision():
    """Проверяет, что диспатчинг сохраняет точность k."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    b = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    
    c = a + b
    assert c.precision_k == 4
    
    d = a * b
    assert d.precision_k == 4
    
    # Для matmul делаем b вектором подходящей формы
    b_mat = to_scrr(torch.randn(5, 3, dtype=torch.float64), k=4)  # [5, 3] для умножения на [5]
    e = a @ b_mat  # [5] @ [5, 3] -> [3]
    assert e.precision_k == 4


def test_dispatch_error_handling():
    """Проверяет обработку ошибок в диспатчинге."""
    a = to_scrr(torch.randn(5, dtype=torch.float64), k=4)
    b = to_scrr(torch.randn(3, dtype=torch.float64), k=4)  # Разные размеры
    
    # Должно вызывать ошибку при попытке сложения тензоров разных размеров
    with pytest.raises(RuntimeError):
        torch.add(a, b)
    
    # Должно вызывать ошибку при попытке матричного умножения несовместимых матриц
    a_mat = to_scrr(torch.randn(3, 4, dtype=torch.float64), k=4)
    b_mat = to_scrr(torch.randn(5, 2, dtype=torch.float64), k=4)  # 4 != 5
    
    with pytest.raises(ValueError, match="Matrix dimensions don't match"):
        torch.matmul(a_mat, b_mat) 