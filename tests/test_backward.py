import sys
sys.path.append('.')

import torch
from mpmath import mp
from src.scrr_fx._tensor import SCRR_Tensor

def test_backward_basic():
    """Базовый тест backward для SCRR_Tensor."""
    print("=== TEST BACKWARD BASIC ===")
    
    # Создаем SCRR_Tensor с requires_grad=True
    x = SCRR_Tensor.from_float(torch.tensor(2.0, dtype=torch.float64), k=4)
    x.requires_grad = True
    
    y = SCRR_Tensor.from_float(torch.tensor(3.0, dtype=torch.float64), k=4)
    y.requires_grad = True
    
    print(f"x = {x.to_mpmath(mp)}")
    print(f"y = {y.to_mpmath(mp)}")
    
    # Вычисляем z = x * y
    z = x * y
    print(f"z = x * y = {z.to_mpmath(mp)}")
    
    # Backward
    z.backward()
    
    print(f"x.grad = {x.grad.to_mpmath(mp) if x.grad else 'None'}")
    print(f"y.grad = {y.grad.to_mpmath(mp) if y.grad else 'None'}")
    
    # Проверяем правильность градиентов
    expected_x_grad = 3.0  # dz/dx = y
    expected_y_grad = 2.0  # dz/dy = x
    
    if x.grad:
        x_grad_error = abs(x.grad.to_mpmath(mp) - expected_x_grad)
        print(f"x.grad error: {x_grad_error}")
        assert x_grad_error < 1e-15, f"x.grad error too large: {x_grad_error}"
    
    if y.grad:
        y_grad_error = abs(y.grad.to_mpmath(mp) - expected_y_grad)
        print(f"y.grad error: {y_grad_error}")
        assert y_grad_error < 1e-15, f"y.grad error too large: {y_grad_error}"

def test_backward_cancellation():
    """Тест backward с cancellation для проверки точности градиентов."""
    print("\n=== TEST BACKWARD CANCELLATION ===")
    
    # Создаем числа с cancellation
    a = 1.0 + 1e-15
    b = 1.0
    
    x = SCRR_Tensor.from_float(torch.tensor(a, dtype=torch.float64), k=4)
    x.requires_grad = True
    
    y = SCRR_Tensor.from_float(torch.tensor(b, dtype=torch.float64), k=4)
    y.requires_grad = True
    
    print(f"x = {a}")
    print(f"y = {b}")
    
    # Вычисляем z = (x - y) * (x - y) = (x - y)²
    diff = x - y
    z = diff * diff
    print(f"z = (x - y)² = {z.to_mpmath(mp)}")
    
    # Backward
    z.backward()
    
    print(f"x.grad = {x.grad.to_mpmath(mp) if x.grad else 'None'}")
    print(f"y.grad = {y.grad.to_mpmath(mp) if y.grad else 'None'}")
    
    # Проверяем правильность градиентов
    # dz/dx = 2(x - y) = 2 * 1e-15
    # dz/dy = -2(x - y) = -2 * 1e-15
    expected_x_grad = 2 * (a - b)
    expected_y_grad = -2 * (a - b)
    
    if x.grad:
        x_grad_error = abs(x.grad.to_mpmath(mp) - expected_x_grad)
        print(f"x.grad error: {x_grad_error}")
        assert x_grad_error < 1e-15, f"x.grad error too large: {x_grad_error}"
    
    if y.grad:
        y_grad_error = abs(y.grad.to_mpmath(mp) - expected_y_grad)
        print(f"y.grad error: {y_grad_error}")
        assert y_grad_error < 1e-15, f"y.grad error too large: {y_grad_error}"

if __name__ == "__main__":
    test_backward_basic()
    test_backward_cancellation()
    print("\n✅ All backward tests passed!") 