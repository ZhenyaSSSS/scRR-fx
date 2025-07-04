import torch
from mpmath import mp
from typing import List

from scrr_fx import SCRR_Tensor

mp.dps = 200 # Повысим точность для надежности


def to_scrr(t: torch.Tensor, k: int) -> SCRR_Tensor:
    """Вспомогательная функция для создания SCRR_Tensor из torch.Tensor."""
    return SCRR_Tensor.from_float(t, k=k)


def _mp_sum(tensor: torch.Tensor) -> mp.mpf:
    """Вычисляет точную сумму всех элементов обычного тензора с помощью mpmath."""
    return mp.fsum(mp.mpf(x.item()) for x in tensor.flatten())


def scrr_to_mp_value(scrr: SCRR_Tensor) -> List[mp.mpf]:
    """
    Конвертирует КАЖДЫЙ элемент в SCRR тензоре в его mpmath значение.
    Возвращает плоский список значений.
    """
    flat_components = scrr.components.reshape(-1, scrr.precision_k)
    values = [mp.fsum(mp.mpf(c.item()) for c in row) for row in flat_components]
    return values

def scrr_to_mp_sum(scrr: SCRR_Tensor) -> mp.mpf:
    """Вычисляет общую точную сумму всех элементов в SCRR тензоре."""
    return mp.fsum(scrr_to_mp_value(scrr))


def torch_to_mp(tensor: torch.Tensor) -> mp.matrix:
    """Конвертирует torch.Tensor (SCRR или обычный) в mpmath.matrix для точных вычислений."""
    # Суммируем по последней размерности (компоненты)
    t_sum = torch.sum(tensor, dim=-1)
    
    if t_sum.ndim == 0:
        return mp.mpf(str(t_sum.item()))
    if t_sum.ndim == 1:
        return mp.matrix([mp.mpf(str(x.item())) for x in t_sum])
    if t_sum.ndim == 2:
        rows, cols = t_sum.shape
        mat = mp.matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                mat[i, j] = mp.mpf(str(t_sum[i, j].item()))
        return mat
    raise ValueError(f"torch_to_mp: unsupported tensor ndim={t_sum.ndim}") 