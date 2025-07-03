"""scrr_fx._renorm
===================
Реализация алгоритмов ренормализации на базе EFT.

* fast_two_sum_reduction — извлекает самый значимый компонент (head)
  из «грязного» тензора и возвращает поток точных ошибок.
* renormalize — строит k-компонентное представление SCRR,
  повторяя fast_two_sum_reduction пока не получит нужное число
  чистых компонентов.

Все функции принимают и возвращают `torch.float64` тензоры.
"""

from __future__ import annotations

from typing import Tuple, List

import torch
import math

from ._core import two_sum

__all__ = [
    "fast_two_sum_reduction",
    "renormalize",
]


def _assert_float64(t: torch.Tensor, name: str = "tensor") -> None:
    if t.dtype != torch.float64:
        raise TypeError(f"{name} must be torch.float64, got {t.dtype}")


def fast_two_sum_reduction(tensor: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduces a tensor along a dimension into a head and a tail component using a cascade of TwoSum operations.
    This is Algorithm 1 from the paper. It is an error-free transformation.
    The sum of the original tensor is exactly equal to `head + sum(tail)`.
    """
    if dim < 0:
        dim += tensor.ndim
        
    s_stream = tensor
    e_stream_parts = []

    while s_stream.shape[dim] > 1:
        m = s_stream.shape[dim]
        n_pairs = m // 2
        
        # Индексы для пар и для непарного элемента, если он есть
        indices_pairs_0 = torch.arange(0, n_pairs * 2, 2, device=tensor.device)
        indices_pairs_1 = torch.arange(1, n_pairs * 2, 2, device=tensor.device)

        s_even = s_stream.index_select(dim, indices_pairs_0)
        s_odd = s_stream.index_select(dim, indices_pairs_1)
        
        s_i, e_i = two_sum(s_even, s_odd)
        
        # Собираем новый поток сумм
        if m % 2 == 1:
            # Если было нечетное число, последний элемент остается
            s_last = s_stream.index_select(dim, torch.tensor([m - 1], device=tensor.device))
            s_stream = torch.cat([s_i, s_last], dim=dim)
        else:
            s_stream = s_i
            
        # Добавляем ошибки в общий поток ошибок
        e_stream_parts.append(e_i)

    # После цикла в s_stream остается один элемент - head
    head = s_stream.squeeze(dim)
    
    if not e_stream_parts:
        # Это может произойти, если на вход пришел тензор с одним элементом
        e_stream = torch.tensor([], dtype=tensor.dtype, device=tensor.device)
        # Нужно добавить размерности, чтобы cat не сломался
        reshape_shape = list(tensor.shape)
        reshape_shape[dim] = 0
        e_stream = e_stream.reshape(reshape_shape)
    else:
        e_stream = torch.cat(e_stream_parts, dim=dim)

    return head, e_stream


def renormalize(tensor: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Performs renormalization on a "dirty" tensor of components to produce a "clean"
    tensor with k components. This is the main renormalization routine.

    It iteratively extracts the most significant component using `fast_two_sum_reduction`
    and repeats the process on the stream of error terms.
    """
    if dim < 0:
        dim += tensor.ndim

    # Если тензор уже "чистый", просто дополняем нулями если нужно.
    if tensor.shape[dim] <= k:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = k - tensor.shape[dim]
        if pad_shape[dim] > 0:
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=dim)
        return tensor

    dirty = tensor
    clean_components = []

    # Извлекаем k-1 наиболее значимых компонентов
    for _ in range(k - 1):
        # Если в "грязном" потоке не осталось компонентов, выходим.
        # Это может случиться, если ошибок было меньше, чем k-1.
        if dirty.shape[dim] == 0:
            break
        head, tail = fast_two_sum_reduction(dirty, dim=dim)
        clean_components.append(head)
        dirty = tail

    # Последний (k-й) компонент — это неточная сумма всех оставшихся ошибок.
    # Это единственное место, где вводится ошибка представления (truncation).
    if dirty.shape[dim] > 0:
        # unsqueeze, чтобы размерность dim не пропала после sum
        last_component = torch.sum(dirty, dim=dim)
        clean_components.append(last_component)

    # Дополняем нулями, если получили меньше k компонентов
    num_собранных = len(clean_components)
    if num_собранных < k:
        # Формируем правильную форму для тензора нулей
        pad_shape = list(clean_components[0].shape) if clean_components else []
        if not pad_shape: # если dirty был пуст
             pad_shape = list(tensor.shape)
             pad_shape.pop(dim)

        for _ in range(k - num_собранных):
            clean_components.append(torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device))

    # Объединяем чистые компоненты в один тензор
    return torch.stack(clean_components, dim=dim)


def _exact_pairwise_sum(vec: torch.Tensor) -> torch.Tensor:
    """Возвращает сумму всех компонентов vec *точно* через цепочку two_sum.

    Работает по последнему измерению. Возвращает тензор [..., 1].
    """
    s = vec[..., 0]
    for idx in range(1, vec.shape[-1]):
        s_tmp, e = two_sum(s, vec[..., idx])
        s = s_tmp + e  # сумма без потери точности
    return s.unsqueeze(-1) 