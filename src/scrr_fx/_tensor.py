"""scrr_fx._tensor
==================
Реализация класса SCRR_Tensor и его интеграции с PyTorch
через протокол __torch_dispatch__.
"""

from __future__ import annotations

from typing import Any, Sequence, Dict, Tuple, List, Optional

import torch

from ._renorm import renormalize

# Глобальный диспатчер для SCRR-операций.
# Заполняется декоратором @implements в _ops.py
HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Декоратор для регистрации SCRR-реализаций функций torch."""
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


class SCRR_Tensor:
    """
    Основной класс для вычислений с высокой точностью.

    Это класс-обертка вокруг torch.Tensor, а не его подкласс.
    Он хранит число как тензор компонентов [..., k] и использует
    __torch_dispatch__ для перехвата и перенаправления операций
    PyTorch на кастомные, численно стабильные реализации.
    """

    # Указываем PyTorch, что у нас есть кастомный __torch_dispatch__
    __torch_function__ = torch._C._disabled_torch_function_impl

    def __init__(self, components: torch.Tensor):
        if not isinstance(components, torch.Tensor):
            raise TypeError("components must be a torch.Tensor")
        if components.dtype != torch.float64:
            raise TypeError("components tensor must be of type torch.float64")
        self.components = components

    @property
    def shape(self) -> Tuple[int, ...]:
        """Возвращает форму тензора без измерения компонентов."""
        return self.components.shape[:-1]

    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    @property
    def dtype(self):
        return self.components.dtype

    @property
    def device(self):
        return self.components.device

    @property
    def precision_k(self) -> int:
        """Возвращает количество компонентов (точность)."""
        return self.components.shape[-1]

    @classmethod
    def from_dirty(cls, tensor: torch.Tensor, k: int) -> SCRR_Tensor:
        """Создает SCRR_Tensor из "грязного" тензора с ренормализацией."""
        return cls(renormalize(tensor, k=k))

    @classmethod
    def from_float(cls, tensor: torch.Tensor, k: int = 2) -> SCRR_Tensor:
        """
        Создает SCRR_Tensor из обычного torch.Tensor.
        Первый компонент - это сам тензор, остальные - нули.
        """
        if tensor.dtype != torch.float64:
            tensor = tensor.to(torch.float64)
            
        padding_shape = tensor.shape + (k - 1,)
        padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
        components = torch.cat([tensor.unsqueeze(-1), padding], dim=-1)
        return cls(components)

    def to_float(self) -> torch.Tensor:
        """
        Конвертирует SCRR_Tensor обратно в стандартный torch.Tensor.
        Это операция с потерей точности.
        """
        return torch.sum(self.components, dim=-1)

    def __repr__(self) -> str:
        val_str = repr(self.to_float()).replace("tensor", "SCRR_Tensor")
        return f"{val_str[:-1]}, k={self.precision_k})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

        raise NotImplementedError(f"SCRR_Tensor: PyTorch function {func.__name__} is not implemented.")

    def value(self) -> torch.Tensor:
        """Возвращает сумму компонентов (числовое значение)."""
        return torch.sum(self.components, dim=-1)

    # --- Магические методы для операторов ---
    def __add__(self, other):
        return torch.add(self, other)

    def __radd__(self, other):
        return torch.add(other, self)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __rmul__(self, other):
        return torch.mul(other, self)

    def __matmul__(self, other):
        return torch.matmul(self, other)
    
    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def __neg__(self):
        return torch.neg(self)

    def __pow__(self, exponent):
        if isinstance(exponent, int):
            if exponent == 0:
                return SCRR_Tensor.from_float(torch.ones(self.shape, dtype=self.dtype, device=self.device), k=self.precision_k)
            elif exponent < 0:
                raise NotImplementedError("Negative powers not supported for SCRR_Tensor")
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        elif isinstance(exponent, float):
            # Используем torch.pow поэлементно
            return SCRR_Tensor.from_float(torch.pow(self.to_float(), exponent), k=self.precision_k)
        else:
            raise TypeError(f"Unsupported exponent type: {type(exponent)}")

    # --- Методы для изменения формы ---
    def reshape(self, *shape) -> SCRR_Tensor:
        new_components = self.components.reshape(shape + (self.precision_k,))
        return SCRR_Tensor(new_components)

    def view(self, *shape) -> SCRR_Tensor:
        # Используем reshape вместо view, так как view может не работать после transpose
        return self.reshape(*shape)
    
    def unsqueeze(self, dim: int) -> SCRR_Tensor:
        new_components = self.components.unsqueeze(dim)
        return SCRR_Tensor(new_components)

    def squeeze(self, dim: int) -> SCRR_Tensor:
        if dim == self.components.ndim - 1:
            raise ValueError("Cannot squeeze the component dimension of an SCRR_Tensor.")
        new_components = self.components.squeeze(dim)
        return SCRR_Tensor(new_components)
    
    def transpose(self, dim0, dim1) -> SCRR_Tensor:
        new_components = self.components.transpose(dim0, dim1)
        return SCRR_Tensor(new_components)

    # -------- поддержка PyTorch dispatch --------
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Старый протокол диспатчинга. Мы используем его как fallback
        для __torch_dispatch__, чтобы гарантировать перехват.
        """
        if kwargs is None:
            kwargs = {}
        
        return SCRR_Tensor.__torch_dispatch__(func, types, args, kwargs)

 