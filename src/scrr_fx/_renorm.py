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


def fast_two_sum_reduction(tensor: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Точно суммирует тензор по измерению `dim` в голову (head) и хвост (tail) из точных ошибок.
    Реализует каскадное попарное суммирование соседних элементов.
    """
    if dim < 0:
        dim += tensor.ndim
        
    s_stream = tensor
    e_stream_parts = []

    while s_stream.shape[dim] > 1:
        m = s_stream.shape[dim]
        n_pairs = m // 2
        
        # Создаем пары соседних элементов: (0,1), (2,3), (4,5), ...
        # Берем четные индексы для первой половины пары
        even_indices = torch.arange(0, 2 * n_pairs, 2, device=tensor.device)
        odd_indices = torch.arange(1, 2 * n_pairs, 2, device=tensor.device)
        
        s_even = s_stream.index_select(dim, even_indices)
        s_odd = s_stream.index_select(dim, odd_indices)
        
        s_i, e_i = two_sum(s_even, s_odd)
        
        # Собираем новый поток сумм
        new_s_parts = [s_i]
        if m % 2 == 1:
            # Если был нечетный элемент, он "проходит" на следующий уровень без изменений
            s_last = s_stream.narrow(dim, m - 1, 1)
            new_s_parts.append(s_last)
        
        s_stream = torch.cat(new_s_parts, dim=dim)
            
        if e_i.numel() > 0:
            e_stream_parts.append(e_i)

    # После цикла в s_stream остается один элемент - это наша "голова"
    head = s_stream.squeeze(dim)
    
    if not e_stream_parts:
        tail_shape = list(tensor.shape)
        tail_shape[dim] = 0
        tail = torch.empty(tail_shape, dtype=tensor.dtype, device=tensor.device)
    else:
        tail = torch.cat(e_stream_parts, dim=dim)

    return head, tail


def renormalize(tensor: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Выполняет ренормализацию "грязного" тензора, итеративно извлекая k чистых компонентов.
    Эта функция точно следует логике, описанной в статье SCRR-FX.
    """
    if dim < 0:
        dim += tensor.ndim

    # Если тензор уже "чистый", просто дополняем нулями если нужно.
    if tensor.shape[dim] <= k:
        if tensor.shape[dim] < k:
            pad_shape = list(tensor.shape)
            pad_shape[dim] = k - tensor.shape[dim]
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=dim)
        return tensor

    dirty = tensor
    clean_components_list = []

    # Итеративно извлекаем k-1 наиболее значимых компонентов
    for _ in range(k - 1):
        # Если в "грязном" потоке не осталось компонентов, выходим.
        if dirty.shape[dim] == 0:
            break
            
        head, tail = fast_two_sum_reduction(dirty, dim=dim)
        
        # Добавляем извлеченную "голову" в список чистых компонентов
        clean_components_list.append(head)
        
        # Оставшийся "хвост" из ошибок становится новым "грязным" тензором для следующей итерации
        dirty = tail

    # Последний (k-й) компонент — это неточная сумма всех оставшихся ошибок.
    # Это единственное место, где вводится ошибка усечения (truncation).
    if dirty.shape[dim] > 0:
        last_component = torch.sum(dirty, dim=dim)
        clean_components_list.append(last_component)

    # Дополняем нулями, если получили меньше k компонентов
    num_собранных = len(clean_components_list)
    if num_собранных < k:
        # Формируем правильную форму для тензора нулей
        pad_shape = list(clean_components_list[0].shape) if clean_components_list else []
        if not pad_shape: # если dirty был пуст с самого начала
             pad_shape = list(tensor.shape)
             pad_shape.pop(dim)

        for _ in range(k - num_собранных):
            clean_components_list.append(torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device))

    # Объединяем чистые компоненты в один тензор
    return torch.stack(clean_components_list, dim=dim)


def _exact_pairwise_sum(vec: torch.Tensor) -> torch.Tensor:
    """Возвращает сумму всех компонентов vec *точно* через цепочку two_sum.

    Работает по последнему измерению. Возвращает тензор [..., 1].
    """
    s = vec[..., 0]
    for idx in range(1, vec.shape[-1]):
        s_tmp, e = two_sum(s, vec[..., idx])
        s = s_tmp + e  # сумма без потери точности
    return s.unsqueeze(-1) 