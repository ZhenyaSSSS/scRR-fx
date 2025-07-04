from __future__ import annotations

"""scrr_fx._ops
================
Реализации SCRR-операций, перехватываемых через __torch_dispatch__.

Каждая функция принимает один или несколько SCRR_Tensor-ов и возвращает
новый SCRR_Tensor с результатом.
"""

from typing import Tuple, Union

import torch

from ._core import two_prod
from ._renorm import renormalize
from ._tensor import SCRR_Tensor, implements, HANDLED_FUNCTIONS

__all__ = [
    "scrr_add",
    "scrr_sub",
    "scrr_mul",
    "scrr_matmul",
    "scrr_neg",
]


# -----------------------------------------------------------------------------
# Вспомогательные утилиты
# -----------------------------------------------------------------------------

def _unify_args(
    x: Union[SCRR_Tensor, torch.Tensor], y: Union[SCRR_Tensor, torch.Tensor]
) -> Tuple[SCRR_Tensor, SCRR_Tensor]:
    """Приводит оба аргумента к SCRR_Tensor с одинаковой точностью k."""
    is_x_scrr = isinstance(x, SCRR_Tensor)
    is_y_scrr = isinstance(y, SCRR_Tensor)

    if is_x_scrr and is_y_scrr:
        if x.precision_k != y.precision_k:
            raise ValueError("Mixed precision SCRR ops not supported yet.")
        k = x.precision_k
    elif is_x_scrr:
        k = x.precision_k
    elif is_y_scrr:
        k = y.precision_k
    else:
        raise TypeError("At least one argument must be an SCRR_Tensor.")

    # Оборачиваем не-SCRR аргументы
    if not is_x_scrr:
        x_tensor = torch.as_tensor(x, dtype=torch.float64, device=y.device)
        x = SCRR_Tensor.from_float(x_tensor, k=k)
    if not is_y_scrr:
        y_tensor = torch.as_tensor(y, dtype=torch.float64, device=x.device)
        y = SCRR_Tensor.from_float(y_tensor, k=k)

    # Broadcasting
    final_shape = torch.broadcast_shapes(x.shape, y.shape)
    
    x_comps = x.components.expand(final_shape + (k,))
    y_comps = y.components.expand(final_shape + (k,))
    
    x_final = SCRR_Tensor(x_comps)
    y_final = SCRR_Tensor(y_comps)

    return x_final, y_final


# -----------------------------------------------------------------------------
# Реализации операций
# -----------------------------------------------------------------------------

@implements(torch.neg)
def scrr_neg(x: SCRR_Tensor) -> SCRR_Tensor:
    """SCRR-реализация `torch.neg` (унарный минус)."""
    return SCRR_Tensor(-x.components)


@implements(torch.add)
def scrr_add(x: Union[SCRR_Tensor, torch.Tensor], y: Union[SCRR_Tensor, torch.Tensor]) -> SCRR_Tensor:
    """SCRR-реализация `torch.add`."""
    x_scrr, y_scrr = _unify_args(x, y)
    k = x_scrr.precision_k
    
    # Шаг 1: Expansion (просто конкатенация)
    dirty = torch.cat([x_scrr.components, y_scrr.components], dim=-1)
    
    # Шаг 2: Renormalization
    new_components = renormalize(dirty, k=k)
    result = SCRR_Tensor(new_components, requires_grad=x_scrr.requires_grad or y_scrr.requires_grad)
    
    # Сохраняем граф вычислений для backward
    if result.requires_grad:
        def backward_hook(grad):
            # Градиент по x: grad
            if x_scrr.requires_grad:
                x_scrr.backward(grad)
            # Градиент по y: grad
            if y_scrr.requires_grad:
                y_scrr.backward(grad)
        
        result.register_backward_hook(backward_hook)
    
    return result


@implements(torch.sub)
def scrr_sub(x: Union[SCRR_Tensor, torch.Tensor], y: Union[SCRR_Tensor, torch.Tensor]) -> SCRR_Tensor:
    """SCRR-реализация `torch.sub`."""
    # `y` может быть не SCRR_Tensor, поэтому нельзя просто вызвать scrr_neg
    if not isinstance(y, SCRR_Tensor):
        y_scrr = _unify_args(x, y)[1]
        return scrr_add(x, scrr_neg(y_scrr))
    return scrr_add(x, scrr_neg(y))


@implements(torch.mul)
def scrr_mul(x: Union[SCRR_Tensor, torch.Tensor], y: Union[SCRR_Tensor, torch.Tensor]) -> SCRR_Tensor:
    """SCRR-реализация `torch.mul` (поэлементное умножение)."""
    x_scrr, y_scrr = _unify_args(x, y)
    k = x_scrr.precision_k
    
    x_comps = x_scrr.components.unsqueeze(-1)
    y_comps = y_scrr.components.unsqueeze(-2)

    p, e = two_prod(x_comps, y_comps)
    dirty = torch.cat([p.flatten(start_dim=-2), e.flatten(start_dim=-2)], dim=-1)
    
    new_components = renormalize(dirty, k=k)
    return SCRR_Tensor(new_components)


@implements(torch.matmul)
def scrr_matmul(a: SCRR_Tensor, b: SCRR_Tensor) -> SCRR_Tensor:
    """
    SCRR-реализация `torch.matmul` с использованием иерархического блочного алгоритма.
    
    Алгоритм (векторизованный):
    1. Expansion: все перекрестные произведения компонентов a[i,n,l] * b[n,j,m]
       вычисляются через two_prod с помощью broadcasting.
    2. Aggregation: все продукты (p) и ошибки (e) собираются в один
       "грязный" тензор размерности [M, P, N*k*k*2].
    3. Renormalization: Renormalize применяется параллельно к каждому
       элементу [i, j] "грязного" тензора.
    """
    # Ручная проверка и унификация типов, т.к. broadcasting здесь не нужен
    if not isinstance(a, SCRR_Tensor) and isinstance(b, SCRR_Tensor):
        a = SCRR_Tensor.from_float(torch.as_tensor(a, dtype=torch.float64, device=b.device), k=b.precision_k)
    elif isinstance(a, SCRR_Tensor) and not isinstance(b, SCRR_Tensor):
        b = SCRR_Tensor.from_float(torch.as_tensor(b, dtype=torch.float64, device=a.device), k=a.precision_k)
    elif not isinstance(a, SCRR_Tensor) and not isinstance(b, SCRR_Tensor):
        raise TypeError("At least one argument to matmul must be an SCRR_Tensor")

    if a.precision_k != b.precision_k:
        raise ValueError("Mixed precision matmul is not supported.")
        
    a_scrr, b_scrr = a, b
    k = a_scrr.precision_k

    # Сохраняем исходные размерности для восстановления формы результата
    orig_a_ndim = a_scrr.ndim
    orig_b_ndim = b_scrr.ndim
    
    # Приводим к 2D матрицам для matmul, если это векторы
    if orig_a_ndim == 1:
        a_scrr = SCRR_Tensor(a_scrr.components.unsqueeze(0)) # [N, k] -> [1, N, k]
    if orig_b_ndim == 1:
        b_scrr = SCRR_Tensor(b_scrr.components.unsqueeze(-2)) # [N, k] -> [N, 1, k]

    # Поддерживаем только 2D x 2D после приведения
    if a_scrr.ndim != 2 or b_scrr.ndim != 2:
        # TODO: Добавить поддержку batch matmul
        raise NotImplementedError(f"SCRR matmul supports 2D@2D, 2D@1D, 1D@2D, 1D@1D. Got {orig_a_ndim}D@{orig_b_ndim}D.")
    
    M, N = a_scrr.shape
    N_check, P = b_scrr.shape
    
    if N != N_check:
        raise ValueError(f"Matrix dimensions don't match for matmul: {a_scrr.shape} vs {b_scrr.shape}")
        
    # --- Векторизованный алгоритм ---
    
    # 1. Expansion
    # a_comps: [M, N, k] -> [M, 1, N, k, 1]
    # b_comps: [N, P, k] -> [1, P, N, 1, k]
    a_exp = a_scrr.components.unsqueeze(1).unsqueeze(-1)
    b_exp = b_scrr.components.permute(1, 0, 2).unsqueeze(0).unsqueeze(-2)

    # Broadcasting-умножение: [M, 1, N, k, 1] * [1, P, N, 1, k] -> [M, P, N, k, k]
    p, e = two_prod(a_exp, b_exp) # p и e имеют форму [M, P, N, k, k]

    # 2. Aggregation
    # Собираем *ВСЕ* промежуточные продукты и ошибки в один большой "грязный" тензор.
    # Это ключевой шаг, который избегает неточной промежуточной суммы.
    # Форма p и e: [M, P, N, k, k]
    # Мы хотим получить грязный тензор формы [M, P, N * k * k * 2]
    
    # Явное приведение `e` к форме `p` для надежности
    e = e.broadcast_to(p.shape)
    
    # Сначала объединяем p и e
    all_prods = torch.cat([p, e], dim=-1) # Форма [M, P, N, k, 2k]
    
    # Затем "сплющиваем" измерения N, k и 2k в одно
    dirty = all_prods.flatten(start_dim=2) # Форма [M, P, N*k*2k]
    
    # 3. Renormalization
    # Применяем renormalize параллельно ко всем M*P элементам
    # `renormalize` суммирует по последнему измерению, что нам и нужно.
    new_components = renormalize(dirty, k=k) # [M, P, k]
    
    result = SCRR_Tensor(new_components)
    
    # Восстанавливаем исходную форму результата в соответствии с правилами matmul
    if orig_a_ndim == 1 and orig_b_ndim == 1: # dot product
        result = SCRR_Tensor(result.components.squeeze(0).squeeze(0))
    elif orig_a_ndim == 1: # vector @ matrix
        result = SCRR_Tensor(result.components.squeeze(0))
    elif orig_b_ndim == 1: # matrix @ vector
        result = SCRR_Tensor(result.components.squeeze(1))
        
    return result


@implements(torch.div)
def scrr_div(x: Union[SCRR_Tensor, torch.Tensor], y: Union[SCRR_Tensor, torch.Tensor]) -> SCRR_Tensor:
    """
    SCRR-реализация `torch.div` (деление) через итерации Ньютона-Рафсона.
    Вычисляет x / y = x * (1/y).
    """
    x_scrr, y_scrr = _unify_args(x, y)
    k = x_scrr.precision_k

    # Начальное приближение для 1/y
    y_float = y_scrr.to_float()
    inv_y_float = 1.0 / y_float
    # Проверка на деление на ноль
    if torch.isinf(inv_y_float).any() or torch.isnan(inv_y_float).any():
         # Если результат inf/nan, просто используем стандартную операцию
         res_float = x_scrr.to_float() / y_float
         return SCRR_Tensor.from_float(res_float, k=k)

    z = SCRR_Tensor.from_float(inv_y_float, k=k)
    
    # Константа 2.0 как SCRR_Tensor
    two = SCRR_Tensor.from_float(torch.tensor(2.0, dtype=torch.float64, device=x_scrr.device), k=k)

    # Итерации Ньютона-Рафсона для уточнения 1/y
    # Количество итераций зависит от k. log2(k) должно быть достаточно.
    # k=2 -> 1 итерация, k=4 -> 2, k=8 -> 3
    num_iterations = max(2, int(torch.ceil(torch.log2(torch.tensor(float(k))))) + 1)
    
    for _ in range(num_iterations):
        # z_new = z * (2 - y * z)
        y_mul_z = scrr_mul(y_scrr, z)
        term = scrr_sub(two, y_mul_z)
        z = scrr_mul(z, term)

    # Финальное умножение x * (1/y)
    return scrr_mul(x_scrr, z)


@implements(torch.true_divide)
def scrr_true_divide(x: Union[SCRR_Tensor, torch.Tensor], y: Union[SCRR_Tensor, torch.Tensor]) -> SCRR_Tensor:
    """SCRR-реализация `torch.true_divide` (аналогично div)."""
    return scrr_div(x, y)


@implements(torch.pow)
def scrr_pow(base: SCRR_Tensor, exponent: Union[int, float, torch.Tensor]) -> SCRR_Tensor:
    """SCRR-реализация `torch.pow`."""
    if not isinstance(exponent, int) or exponent < 0:
        raise NotImplementedError("SCRR pow currently only supports non-negative integer exponents.")

    k = base.precision_k

    if exponent == 0:
        ones = torch.ones_like(base.to_float())
        return SCRR_Tensor.from_float(ones, k=k)
    
    if exponent == 1:
        return base

    # Алгоритм возведения в степень через двоичное разложение (exponentiation by squaring)
    result = SCRR_Tensor.from_float(torch.ones_like(base.to_float()), k=k)
    
    current_power = base
    exp = exponent
    
    while exp > 0:
        if exp % 2 == 1:
            result = scrr_mul(result, current_power)
        
        current_power = scrr_mul(current_power, current_power)
        exp //= 2
        
    return result


# Переопределяем, так как импортировали HANDLED_FUNCTIONS
# и @implements уже заполнил его
# Этот словарь теперь не нужен, так как декоратор делает всё сам.
# Оставим его для ясности, какие функции мы переопределяем.
SUPPORTED_OPS = [
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.true_divide,
    torch.matmul,
    torch.neg,
]
for op in SUPPORTED_OPS:
    if op not in HANDLED_FUNCTIONS:
        # Этого не должно случиться, если все обернуто в @implements
        print(f"Warning: op {op} not in HANDLED_FUNCTIONS") 