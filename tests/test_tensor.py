"""Тесты для класса SCRR_Tensor (фаза 2)."""

import torch
import pytest
import mpmath as mp

from scrr_fx._tensor import SCRR_Tensor
from tests.helpers import to_scrr, scrr_to_mp_value

mp.dps = 200


def test_scrr_tensor_constructor():
    """Проверяет конструктор SCRR_Tensor."""
    # Создание из компонентов
    components = torch.randn(5, 4, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    assert scrr.shape == (5,)
    assert scrr.precision_k == 4
    assert scrr.dtype == torch.float64
    assert torch.allclose(scrr.components, components)


def test_scrr_tensor_constructor_errors():
    """Проверяет обработку ошибок в конструкторе."""
    # Неправильный тип
    with pytest.raises(TypeError, match="components must be a torch.Tensor"):
        SCRR_Tensor([1, 2, 3])
    
    # Неправильный dtype
    wrong_dtype = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(TypeError, match="components tensor must be of type torch.float64"):
        SCRR_Tensor(wrong_dtype)


def test_from_float():
    """Проверяет создание SCRR_Tensor из обычного тензора."""
    tensor = torch.randn(5, 3, dtype=torch.float64)
    k = 4
    
    scrr = SCRR_Tensor.from_float(tensor, k=k)
    
    assert scrr.shape == (5, 3)
    assert scrr.precision_k == k
    assert torch.allclose(scrr.components[..., 0], tensor)
    assert torch.all(scrr.components[..., 1:] == 0)


def test_from_float_auto_conversion():
    """Проверяет автоматическое преобразование dtype в from_float."""
    tensor = torch.randn(5, dtype=torch.float32)
    k = 4
    
    scrr = SCRR_Tensor.from_float(tensor, k=k)
    
    assert scrr.dtype == torch.float64
    assert scrr.precision_k == k


def test_from_dirty():
    """Проверяет создание SCRR_Tensor из "грязного" тензора."""
    dirty = torch.randn(5, 10, dtype=torch.float64)
    k = 4
    
    scrr = SCRR_Tensor.from_dirty(dirty, k=k)
    
    assert scrr.shape == (5,)
    assert scrr.precision_k == k
    
    # Сумма должна сохраниться
    original_sum = torch.sum(dirty, dim=-1)
    scrr_sum = scrr.to_float()
    assert torch.allclose(original_sum, scrr_sum)


def test_to_float():
    """Проверяет конвертацию обратно в обычный тензор."""
    components = torch.randn(5, 4, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    result = scrr.to_float()
    
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float64
    assert result.shape == (5,)
    assert torch.allclose(result, torch.sum(components, dim=-1))


def test_value():
    """Проверяет метод value()."""
    components = torch.randn(5, 4, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    value = scrr.value()
    
    assert torch.allclose(value, scrr.to_float())
    assert torch.allclose(value, torch.sum(components, dim=-1))


def test_properties():
    """Проверяет свойства SCRR_Tensor."""
    components = torch.randn(3, 4, 5, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    assert scrr.shape == (3, 4)
    assert scrr.ndim == 2
    assert scrr.precision_k == 5
    assert scrr.dtype == torch.float64
    assert scrr.device == components.device


def test_reshape():
    """Проверяет операцию reshape."""
    components = torch.randn(6, 4, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    reshaped = scrr.reshape(2, 3)
    
    assert isinstance(reshaped, SCRR_Tensor)
    assert reshaped.shape == (2, 3)
    assert reshaped.precision_k == 4
    assert torch.allclose(reshaped.to_float(), scrr.to_float().reshape(2, 3))


def test_view():
    """Проверяет операцию view."""
    components = torch.randn(6, 4, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    viewed = scrr.view(2, 3)
    
    assert isinstance(viewed, SCRR_Tensor)
    assert viewed.shape == (2, 3)
    assert viewed.precision_k == 4


def test_unsqueeze():
    """Проверяет операцию unsqueeze."""
    components = torch.randn(5, 4, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    unsqueezed = scrr.unsqueeze(0)
    
    assert isinstance(unsqueezed, SCRR_Tensor)
    assert unsqueezed.shape == (1, 5)
    assert unsqueezed.precision_k == 4
    assert torch.allclose(unsqueezed.to_float(), scrr.to_float().unsqueeze(0))


def test_squeeze():
    """Проверяет операцию squeeze."""
    components = torch.randn(1, 5, 4, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    squeezed = scrr.squeeze(0)
    
    assert isinstance(squeezed, SCRR_Tensor)
    assert squeezed.shape == (5,)
    assert squeezed.precision_k == 4
    assert torch.allclose(squeezed.to_float(), scrr.to_float().squeeze(0))


def test_squeeze_component_dimension_error():
    """Проверяет, что squeeze не может сжать размерность компонентов."""
    components = torch.randn(5, 1, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    with pytest.raises(ValueError, match="Cannot squeeze the component dimension"):
        scrr.squeeze(1)


def test_transpose():
    """Проверяет операцию transpose."""
    components = torch.randn(3, 4, 5, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    transposed = scrr.transpose(0, 1)
    
    assert isinstance(transposed, SCRR_Tensor)
    assert transposed.shape == (4, 3)
    assert transposed.precision_k == 5
    assert torch.allclose(transposed.to_float(), scrr.to_float().transpose(0, 1))


def test_repr():
    """Проверяет строковое представление."""
    components = torch.randn(5, 4, dtype=torch.float64)
    scrr = SCRR_Tensor(components)
    
    repr_str = repr(scrr)
    
    assert "SCRR_Tensor" in repr_str
    assert "k=4" in repr_str
    assert "tensor" in repr_str.lower()


def test_mixed_precision_error():
    """Проверяет ошибку при смешивании разных точностей."""
    a = SCRR_Tensor.from_float(torch.randn(5, dtype=torch.float64), k=4)
    b = SCRR_Tensor.from_float(torch.randn(5, dtype=torch.float64), k=8)
    
    with pytest.raises(ValueError, match="Mixed precision SCRR ops not supported yet"):
        a + b


def test_device_handling():
    """Проверяет корректную работу с устройствами."""
    if torch.cuda.is_available():
        # Тест на GPU
        components_gpu = torch.randn(5, 4, dtype=torch.float64, device='cuda')
        scrr_gpu = SCRR_Tensor(components_gpu)
        
        assert scrr_gpu.device.type == 'cuda'
        assert scrr_gpu.components.device.type == 'cuda'
    
    # Тест на CPU
    components_cpu = torch.randn(5, 4, dtype=torch.float64, device='cpu')
    scrr_cpu = SCRR_Tensor(components_cpu)
    
    assert scrr_cpu.device.type == 'cpu'
    assert scrr_cpu.components.device.type == 'cpu' 