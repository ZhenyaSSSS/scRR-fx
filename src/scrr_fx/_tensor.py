"""scrr_fx._tensor
==================
Реализация класса SCRR_Tensor и его интеграции с PyTorch
через протокол __torch_dispatch__.
"""

from __future__ import annotations

from typing import Any, Sequence, Dict, Tuple, List, Optional, Union

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

        Представление простое и точное: исходный тензор становится
        первым компонентом, а остальные k-1 компонентов заполняются нулями.
        Это гарантирует, что `value(SCRR_Tensor) == tensor`.
        """
        if tensor.dtype != torch.float64:
            tensor = tensor.to(torch.float64)

        # Добавляем новое измерение для компонентов
        components = torch.zeros(tensor.shape + (k,), dtype=torch.float64, device=tensor.device)
        
        # Помещаем исходный тензор в первый компонент
        components[..., 0] = tensor
        
        return cls(components)

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

    def to_float(self) -> torch.Tensor:
        """
        Конвертирует SCRR_Tensor обратно в стандартный torch.Tensor.
        Это быстрая, но неточная операция. Для точного значения
        используйте `to_mpmath`.
        """
        return torch.sum(self.components, dim=-1)

    def to_mpmath(self, mp_ctx) -> Union[mp.mpf, List[mp.mpf]]:
        """
        Конвертирует SCRR_Tensor обратно в mpmath.mpf (или список),
        используя переданный контекст mp_ctx.
        Это медленный, но численно точный метод.
        """
        # Векторизованный вариант будет сложен из-за природы mpmath,
        # поэтому итерируем.
        if self.ndim == 0: # Скаляр
            return self.to_mpmath_scalar(mp_ctx)
        
        # Работаем с плоским представлением и восстанавливаем форму в конце
        flat_components = self.components.reshape(-1, self.precision_k)
        mp_values = []
        for i in range(flat_components.shape[0]):
            # Используем строковое представление для максимальной точности
            comp_strings = [repr(c.item()) for c in flat_components[i]]
            comp_mp = [mp_ctx.mpf(s) for s in comp_strings]
            mp_values.append(mp_ctx.fsum(comp_mp))
            
        # Восстанавливаем исходную форму
        if self.ndim > 1:
            return np.array(mp_values, dtype=object).reshape(self.shape).tolist()
        else:
            return mp_values
            
    def to_mpmath_scalar(self, mp_ctx) -> mp.mpf:
        """Вспомогательная функция для конвертации скалярного SCRR_Tensor в mpmath.mpf."""
        if self.ndim > 0:
            raise ValueError("This method is for scalar SCRR_Tensors only")
        
        components_as_strings = [repr(c) for c in self.components.flatten().tolist()]
        components_mp = [mp_ctx.mpf(s) for s in components_as_strings]
        return mp_ctx.fsum(components_mp)

    def __repr__(self) -> str:
        # Для repr используем быструю, неточную конвертацию
        val_str = repr(self.to_float()).replace("tensor", "SCRR_Tensor")
        # Удаляем лишние пробелы и переносы строк
        val_str = ' '.join(val_str.split())
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
        # Реализация через torch.pow, будет перехвачена диспатчером
        return torch.pow(self, exponent)

    def __truediv__(self, other):
        return torch.div(self, other)

    def __rtruediv__(self, other):
        return torch.div(other, self)

    def __getitem__(self, key):
        """Позволяет индексировать SCRR_Tensor как обычный тензор."""
        if not isinstance(key, tuple):
            key = (key,)
        
        # Добавляем полный срез для измерения компонентов, чтобы оно не затрагивалось
        key_for_components = key + (slice(None),)
        
        sub_components = self.components[key_for_components]
        
        # Если в результате индексации мы все еще имеем измерение для k,
        # возвращаем новый SCRR_Tensor. Иначе, это скаляр, и мы должны
        # вернуть его "значение".
        if sub_components.ndim > 0 and sub_components.shape[-1] == self.precision_k:
             return SCRR_Tensor(sub_components)
        else:
             # Если мы выбрали один компонент или срез, который не является полным
             # возвращаем обычный torch.Tensor
             return sub_components

    # --- Методы для изменения формы ---
    def reshape(self, *shape) -> SCRR_Tensor:
        """Возвращает SCRR_Tensor с измененной формой."""
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

 