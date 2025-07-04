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
    if isinstance(x, SCRR_Tensor) and isinstance(y, SCRR_Tensor):
        if x.precision_k != y.precision_k:
            raise ValueError("Mixed precision SCRR ops not supported yet.")
        return x, y
    
    is_x_scrr = isinstance(x, SCRR_Tensor)
    scrr_arg = x if is_x_scrr else y
    other_arg = y if is_x_scrr else x

    if not isinstance(other_arg, torch.Tensor):
        other_arg = torch.tensor(other_arg, dtype=torch.float64, device=scrr_arg.device)

    # Создаем SCRR тензор из обычного, и даем ему возможность "вещать" (broadcast)
    # до формы другого тензора.
    other_scrr = SCRR_Tensor.from_float(other_arg, k=scrr_arg.precision_k)
    
    # new_shape должен соответствовать форме scrr_arg, но сохранить k компонентов
    final_shape = torch.broadcast_shapes(scrr_arg.shape, other_scrr.shape)
    
    # Broadcast a.components до нужной формы.
    # Добавляем фиктивное измерение для k, чтобы broadcast работал как надо
    expanded_other_comps = other_scrr.components.expand(final_shape + (scrr_arg.precision_k,))
    
    other_final = SCRR_Tensor(expanded_other_comps)
    
    if is_x_scrr:
        return scrr_arg, other_final
    else:
        return other_final, scrr_arg


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
    
    Алгоритм из статьи:
    1. Expansion: вычисляем все попарные произведения компонентов через TwoProd
    2. Aggregation: собираем "грязный" тензор для каждого элемента результата
    3. Renormalization: применяем Renormalize к каждому элементу результата
    
    Сложность: O(N³k² log(Nk²)) вместо O(N³k⁴) для наивного подхода.
    """
    a_scrr, b_scrr = _unify_args(a, b)
    k = a_scrr.precision_k
    
    # Обрабатываем случаи 1D векторов (как в torch.matmul)
    a_is_1d = a_scrr.ndim == 1
    b_is_1d = b_scrr.ndim == 1
    
    if a_is_1d:
        a_scrr = a_scrr.unsqueeze(0)  # [1, N, k]
    if b_is_1d:
        b_scrr = b_scrr.unsqueeze(-1)  # [N, 1, k]
    
    # Поддерживаем только 2D матрицы после приведения
    if a_scrr.ndim != 2 or b_scrr.ndim != 2:
        raise NotImplementedError("SCRR matmul currently supports only 2D matrices")
    
    M, N = a_scrr.shape
    N_check, P = b_scrr.shape
    
    if N != N_check:
        raise ValueError(f"Matrix dimensions don't match: {a_scrr.shape} vs {b_scrr.shape}")
    
    # Создаем результирующий тензор компонентов
    result_components = torch.zeros(M, P, k, dtype=torch.float64, device=a_scrr.device)
    
    # Для каждого элемента результата (i, j)
    for i in range(M):
        for j in range(P):
            # Собираем все произведения для dot-product a[i, :] · b[:, j]
            dirty_terms = []
            
            # Для каждого n в сумме
            for n in range(N):
                # Для каждой пары компонентов (l, m)
                for l in range(k):
                    for m in range(k):
                        # Вычисляем произведение через TwoProd
                        a_comp = a_scrr.components[i, n, l]  # скаляр
                        b_comp = b_scrr.components[n, j, m]  # скаляр
                        
                        # two_prod ожидает тензоры, создаем скалярные тензоры без warning
                        a_tensor = a_comp.clone().detach()
                        b_tensor = b_comp.clone().detach()
                        
                        p, e = two_prod(a_tensor, b_tensor)
                        
                        # Добавляем оба компонента в грязный список
                        dirty_terms.append(p.item())
                        if e.item() != 0.0:  # Оптимизация: не добавляем нулевые ошибки
                            dirty_terms.append(e.item())
            
            # Если нет термов, результат — ноль
            if not dirty_terms:
                # result_components[i, j, :] уже инициализирован нулями
                continue
            
            # Создаем грязный тензор и ренормализуем
            dirty_tensor = torch.tensor(dirty_terms, dtype=torch.float64, device=a_scrr.device)
            clean_components = renormalize(dirty_tensor.unsqueeze(0), k=k).squeeze(0)  # [k]
            
            result_components[i, j, :] = clean_components
    
    result = SCRR_Tensor(result_components)
    
    # Убираем добавленные размерности для 1D случаев
    if a_is_1d and b_is_1d:
        # Векторное произведение: результат скаляр
        result = result.squeeze(0).squeeze(0)  # [1, 1, k] -> [k]
    elif a_is_1d:
        # Строка на матрицу: результат строка
        result = result.squeeze(0)  # [1, P, k] -> [P, k]
    elif b_is_1d:
        # Матрица на столбец: результат столбец
        result = result.squeeze(1)  # [M, 1, k] -> [M, k]
    
    return result


@implements(torch.div)
def scrr_div(x: Union[SCRR_Tensor, torch.Tensor], y: Union[SCRR_Tensor, torch.Tensor]) -> SCRR_Tensor:
    """SCRR-реализация `torch.div` (деление)."""
    x_scrr, y_scrr = _unify_args(x, y)
    k = x_scrr.precision_k
    
    # Для деления используем приближение: x/y ≈ x * (1/y)
    # Сначала вычисляем 1/y через обычные torch операции
    y_float = y_scrr.to_float()
    inv_y_float = 1.0 / y_float
    inv_y_scrr = SCRR_Tensor.from_float(inv_y_float, k=k)
    
    # Затем умножаем x на 1/y
    return scrr_mul(x_scrr, inv_y_scrr)


@implements(torch.true_divide)
def scrr_true_divide(x: Union[SCRR_Tensor, torch.Tensor], y: Union[SCRR_Tensor, torch.Tensor]) -> SCRR_Tensor:
    """SCRR-реализация `torch.true_divide` (аналогично div)."""
    return scrr_div(x, y)


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