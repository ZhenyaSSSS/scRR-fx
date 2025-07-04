from __future__ import annotations

"""scrr_fx._core
=================
Числовые примитивы «error-free transformation» (EFT),
используемые во всём фреймворке SCRR-FX.

* two_sum — точное представление суммы
* two_prod — точное представление произведения

Все операции работают покомпонентно на `torch.float64` тензорах
и возвращают пару (value, error) такой же формы.
"""

import torch

__all__ = ["two_sum", "two_prod"]


def two_sum(a: torch.Tensor, b: torch.Tensor):
    """Двойная сумма (Knuth + Kahan).

    Возвращает `(s, e)` такие, что `a + b == s + e` *точно* в
    вещественной арифметике, где `s` — округлённая сумма, `e` — ошибка
    округления. Работает для broadcast-совместимых тензоров `float64`.
    """
    if a.dtype != torch.float64 or b.dtype != torch.float64:
        raise TypeError("two_sum expects float64 tensors")
        
    # Явное распространение NaN
    if torch.isnan(a).any() or torch.isnan(b).any():
        nan_tensor = torch.full_like(a, float('nan'))
        return nan_tensor, nan_tensor

    # Если есть инф/нан, точная ошибка не имеет смысла — возвращаем e=0
    if not torch.isfinite(a + b).all():
        s = a + b
        e = torch.zeros_like(s)
        return s, e

    # Основной EFT-алгоритм (a, b конечны)
    s = a + b
    bp = s - a
    ap = s - bp
    delta_b = b - bp
    delta_a = a - ap
    e = delta_a + delta_b
    return s, e


def _two_prod_fma(a: torch.Tensor, b: torch.Tensor):
    """Произведение с использованием FMA, если доступно."""
    p = a * b
    e = torch.fma(a, b, -p)  # Точный остаток, если поддерживается устройством
    return p, e


def _two_prod_dekker(a: torch.Tensor, b: torch.Tensor):
    """Dekker split-algorithm без FMA (Hida, Li, Bailey 2001)."""
    split = 134217729.0  # 2^27 + 1

    c = split * a
    a_high = c - (c - a)
    a_low = a - a_high

    c = split * b
    b_high = c - (c - b)
    b_low = b - b_high

    p = a * b
    e = ((a_high * b_high - p) + a_high * b_low + a_low * b_high) + a_low * b_low
    # Явное создание e той же формы, что и p, заполненное нулями,
    # и добавление e. Это гарантирует правильную форму.
    err = torch.zeros_like(p)
    err.add_(e)
    return p, err


def two_prod(a: torch.Tensor, b: torch.Tensor):
    """Двойное произведение (TwoProd).

    Возвращает `(p, e)` такие, что `a * b == p + e` *точно*.
    Предпочитает использовать FMA для максимальной точности/скорости,
    но при его отсутствии автоматически переключается на Dekker-вариант.
    """
    if a.dtype != torch.float64 or b.dtype != torch.float64:
        raise TypeError("two_prod expects float64 tensors")
        
    # Явное распространение NaN
    prod_result = a * b
    # Если где-то есть NaN (включая inf*0), возвращаем (nan, nan)
    if torch.isnan(prod_result).any():
        nan_tensor = torch.full_like(a, float('nan'))
        return nan_tensor, nan_tensor

    # Если произведение Inf, но не NaN, то ошибка 0
    if not torch.isfinite(prod_result).all():
        return prod_result, torch.zeros_like(a)

    if hasattr(torch, "fma"):
        try:
            p, e = _two_prod_fma(a, b)
            # Убедимся, что e имеет ту же форму, что и p
            if e.shape != p.shape:
                e = torch.broadcast_to(e, p.shape)
            return p, e
        except RuntimeError:
            # Если FMA недоступен (например, на CPU без AVX),
            # используем фоллбэк на Dekker
            pass
    return _two_prod_dekker(a, b) 