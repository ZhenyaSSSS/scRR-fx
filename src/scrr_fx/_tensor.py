"""scrr_fx._tensor
==================
Реализация класса SCRR_Tensor и его интеграции с PyTorch
через протокол __torch_dispatch__.
"""

from __future__ import annotations

from typing import Any, Sequence, Dict, Tuple, List, Optional

import torch
from mpmath import mp
import numpy as np

from ._renorm import renormalize
from ._core import two_sum

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

    def __init__(self, components: torch.Tensor, requires_grad: bool = False):
        if not isinstance(components, torch.Tensor):
            raise TypeError("components must be a torch.Tensor")
        if components.dtype != torch.float64:
            raise TypeError("components tensor must be of type torch.float64")
        self.components = components
        self.requires_grad = requires_grad
        self.grad = None  # Градиент тоже SCRR_Tensor
        self._backward_hooks = []  # Для цепочки backward

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
        Реализует честное разложение float64 в expansion (QD-style, Dekker/Hida):
        [x0, x1, ..., x_{k-1}], где x0+x1+...+x_{k-1} = x точно.
        
        Для больших чисел использует более стабильный подход.
        """
        if tensor.dtype != torch.float64:
            tensor = tensor.to(torch.float64)
        
        orig_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        # Создаем тензор для компонентов напрямую
        # Форма: [N, k], где N - количество элементов в исходном тензоре
        components = torch.zeros(flat_tensor.numel(), k, dtype=torch.float64, device=tensor.device)
        
        # Обрабатываем каждый элемент отдельно
        for i, x_val in enumerate(flat_tensor):
            cur = x_val.item()
            
            # Для очень больших чисел используем более стабильный подход
            if abs(cur) > 1e15:
                # Для больших чисел просто кладем число в первый компонент
                # и нули в остальные - это лучше, чем огромные ошибки разложения
                components[i, 0] = cur
                # Остальные компоненты уже инициализированы нулями
            else:
                # Для обычных чисел используем правильный Dekker split
                # Итеративно извлекаем компоненты, начиная с самого значимого
                remainder = cur
                
                for j in range(k - 1):
                    if abs(remainder) < 1e-300:
                        # Остаток слишком мал, остальные компоненты будут нулями
                        break
                    
                    # Dekker split для извлечения старшей части
                    split = 134217729.0  # 2^27 + 1
                    c = split * remainder
                    high = c - (c - remainder)
                    
                    components[i, j] = high
                    remainder = remainder - high
                
                # Последний компонент - это весь оставшийся остаток
                components[i, -1] = remainder
        
        # Возвращаем компонентам исходную форму
        final_shape = orig_shape + (k,)
        return cls(components.reshape(final_shape))

    @classmethod
    def from_mpmath(cls, mp_ctx, mpf_list, k: int = 2, device=None) -> 'SCRR_Tensor':
        """
        Создает SCRR_Tensor из списка mpmath.mpf, используя переданный контекст mp_ctx.
        Этот метод корректно извлекает k float64 компонентов из высокоточного числа.
        """
        if not isinstance(mpf_list, (list, tuple, np.ndarray)):
            mpf_list = [mpf_list]
            
        all_comps = []
        for x in mpf_list:
            local_comps = []
            residue = mp_ctx.mpf(x)  # Начинаем с полного высокоточного числа
            
            for _ in range(k):
                if abs(residue) == 0:
                    local_comps.append(0.0)
                    continue
                
                # Извлекаем наиболее значимую часть, представимую в float64
                c_float = float(residue)
                local_comps.append(c_float)
                
                # Вычитаем извлеченную часть с высокой точностью, чтобы получить остаток
                residue -= mp_ctx.mpf(repr(c_float))
            
            all_comps.append(local_comps)
            
        target_device = device if device is not None else torch.device("cpu")
        components_tensor = torch.tensor(all_comps, dtype=torch.float64, device=target_device)
        
        # Если на входе был один скаляр, возвращаем тензор правильной формы [k]
        if len(mpf_list) == 1:
            components_tensor = components_tensor.squeeze(0)

        return cls(components_tensor)

    def to_float(self, exact_sum: bool = False, mp_ctx=None) -> torch.Tensor:
        """
        Конвертирует SCRR_Tensor обратно в стандартный torch.Tensor.
        
        Args:
            exact_sum: Если True, возвращает точную сумму через mpmath
                      (медленно, но без потери точности)
            mp_ctx: Контекст mpmath для точных вычислений
        """
        if exact_sum:
            if mp_ctx is None:
                raise ValueError("mp_ctx must be provided when exact_sum=True")
            
            # Точная сумма через mpmath (медленно, но точно)
            mp_ctx.dps = 1000  # Высокая точность
            
            # Обрабатываем разные размерности
            if self.components.ndim == 1:
                # Скаляр: [k]
                sum_mp = mp_ctx.mpf(0)
                for j in range(self.precision_k):
                    sum_mp += mp_ctx.mpf(self.components[j].item())
                return torch.tensor(float(sum_mp), dtype=torch.float64, device=self.device)
            else:
                # Многомерный: [..., k]
                result = []
                for i in range(self.components.shape[0]):
                    sum_mp = mp_ctx.mpf(0)
                    for j in range(self.precision_k):
                        sum_mp += mp_ctx.mpf(self.components[i, j].item())
                    result.append(float(sum_mp))
                
                return torch.tensor(result, dtype=torch.float64, device=self.device)
        else:
            # Быстрая сумма в float64 (может потерять точность)
            return torch.sum(self.components, dim=-1)

    def to_mpmath(self, mp_ctx) -> 'mpf':
        """
        Конвертирует SCRR_Tensor обратно в mpmath.mpf, используя переданный контекст mp_ctx.
        Использует строковое представление для максимальной точности.
        """
        # Конвертируем каждый компонент в строку, чтобы избежать ошибок округления float -> mpf
        components_as_strings = [repr(c) for c in self.components.flatten().tolist()]
        
        # Создаем mpf из строковых представлений
        components_mp = [mp_ctx.mpf(s) for s in components_as_strings]
        
        # fsum - это точная сумма для чисел с плавающей точкой
        return mp_ctx.fsum(components_mp)

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
        if not isinstance(exponent, int) or exponent < 0:
            # Пока поддерживаем только целые неотрицательные степени
            raise NotImplementedError("Only non-negative integer powers are supported for SCRR_Tensor.")

        if exponent == 0:
            # Любое число в степени 0 равно 1
            ones_tensor = torch.ones(self.shape, dtype=self.dtype, device=self.device)
            return SCRR_Tensor.from_float(ones_tensor, k=self.precision_k)
        
        if exponent == 1:
            return self
        
        # Стандартный алгоритм exponentiation by squaring
        result = SCRR_Tensor.from_float(torch.ones(self.shape, dtype=self.dtype, device=self.device), k=self.precision_k)
        base = self
        exp = exponent

        while exp > 0:
            # Если степень нечетная, умножаем result на base
            if exp % 2 == 1:
                result = result * base
            # Возводим base в квадрат и уменьшаем степень вдвое
            base = base * base
            exp //= 2
            
        return result

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

    def register_backward_hook(self, hook):
        """Регистрирует хук для backward pass."""
        self._backward_hooks.append(hook)
        return hook

    def backward(self, gradient: Optional[SCRR_Tensor] = None):
        """
        Вычисляет градиенты для всех тензоров с requires_grad=True.
        
        Args:
            gradient: Градиент от следующего слоя (тоже SCRR_Tensor)
        """
        if not self.requires_grad:
            return
        
        # Если градиент не передан, используем единичный
        if gradient is None:
            gradient = SCRR_Tensor.from_float(
                torch.ones_like(self.components[..., 0]), 
                k=self.precision_k
            )
        
        # Устанавливаем градиент
        if self.grad is None:
            self.grad = gradient
        else:
            # Градиенты накапливаются
            self.grad = self.grad + gradient
        
        # Вызываем хуки
        for hook in self._backward_hooks:
            hook(gradient)
    
    def detach(self) -> SCRR_Tensor:
        """Отсоединяет тензор от графа вычислений."""
        return SCRR_Tensor(self.components.clone(), requires_grad=False)
    
    def retain_grad(self):
        """Сохраняет градиент даже после backward."""
        self.requires_grad = True

 