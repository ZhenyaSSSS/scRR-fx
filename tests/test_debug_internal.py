"""Детальная диагностика внутренних операций SCRR."""

import torch
import numpy as np
from src.scrr_fx import SCRR_Tensor


def test_scrr_internal_debug():
    """Проверяем что SCRR НЕ крашится в ноль, а действительно вычисляет."""
    print("\n" + "="*80)
    print("🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА ВНУТРЕННИХ ОПЕРАЦИЙ SCRR")
    print("="*80)
    
    # Тест 1: Простое сложение - должно давать НЕ ноль
    print("\n1️⃣ ТЕСТ: 1000000.1 + 0.9 (должно дать 1000001.0)")
    
    a_val = 1000000.1
    b_val = 0.9
    expected = a_val + b_val  # 1000001.0
    
    a_scrr = SCRR_Tensor.from_float(torch.tensor(a_val, dtype=torch.float64), k=4)
    b_scrr = SCRR_Tensor.from_float(torch.tensor(b_val, dtype=torch.float64), k=4)
    
    print(f"a_scrr компоненты: {a_scrr.components}")
    print(f"b_scrr компоненты: {b_scrr.components}")
    
    result_scrr = a_scrr + b_scrr
    print(f"(a+b) компоненты: {result_scrr.components}")
    
    final_result = result_scrr.to_float().item()
    print(f"Результат SCRR: {final_result}")
    print(f"Ожидаемый: {expected}")
    print(f"Ошибка: {abs(final_result - expected)}")
    
    if abs(final_result) < 1e-10:
        print("🚨 АЛЕРТ: SCRR выдал почти ноль - возможно краш!")
    else:
        print("✅ SCRR работает корректно")
    
    # Тест 2: Умножение - должно давать НЕ ноль  
    print("\n2️⃣ ТЕСТ: 1000.5 * 999.5 (должно дать ~999999)")
    
    a_val = 1000.5
    b_val = 999.5  
    expected = a_val * b_val
    
    a_scrr = SCRR_Tensor.from_float(torch.tensor(a_val, dtype=torch.float64), k=4)
    b_scrr = SCRR_Tensor.from_float(torch.tensor(b_val, dtype=torch.float64), k=4)
    
    print(f"a_scrr компоненты: {a_scrr.components}")
    print(f"b_scrr компоненты: {b_scrr.components}")
    
    result_scrr = a_scrr * b_scrr
    print(f"(a*b) компоненты: {result_scrr.components}")
    
    final_result = result_scrr.to_float().item()
    print(f"Результат SCRR: {final_result}")
    print(f"Ожидаемый: {expected}")
    print(f"Ошибка: {abs(final_result - expected)}")
    
    if abs(final_result) < 1e-10:
        print("🚨 АЛЕРТ: SCRR выдал почти ноль - возможно краш!")
    else:
        print("✅ SCRR работает корректно")
    
    # Тест 3: Проблемная функция ИЗ ОРИГИНАЛЬНОГО ТЕСТА
    print("\n3️⃣ ТЕСТ: 1.0 - (1.0 - 1e-16) - тот самый проблемный случай!")
    
    one = torch.tensor(1.0, dtype=torch.float64)
    small = torch.tensor(1e-16, dtype=torch.float64)
    
    # float64 результат
    intermediate_float64 = one - small  # 1.0 - 1e-16
    result_float64 = one - intermediate_float64  # 1.0 - (1.0 - 1e-16)
    
    print(f"float64: 1.0 - 1e-16 = {intermediate_float64.item():.20e}")
    print(f"float64: 1.0 - (1.0 - 1e-16) = {result_float64.item():.20e}")
    
    # SCRR результат
    one_scrr = SCRR_Tensor.from_float(one, k=8)
    small_scrr = SCRR_Tensor.from_float(small, k=8)
    
    print(f"\none_scrr компоненты: {one_scrr.components}")
    print(f"small_scrr компоненты: {small_scrr.components}")
    
    intermediate_scrr = one_scrr - small_scrr
    print(f"(1.0 - 1e-16) SCRR компоненты: {intermediate_scrr.components}")
    
    result_scrr_tensor = one_scrr - intermediate_scrr  
    print(f"1.0 - (1.0 - 1e-16) SCRR компоненты: {result_scrr_tensor.components}")
    
    result_scrr = result_scrr_tensor.to_float().item()
    
    print(f"\nИтоговые результаты:")
    print(f"float64: {result_float64.item():.20e}")
    print(f"SCRR:    {result_scrr:.20e}")
    
    if abs(result_scrr) < 1e-20:
        print("🚨 АЛЕРТ: SCRR выдал практически ноль!")
        print("   Это может быть либо:")
        print("   1) Краш/ошибка в коде")
        print("   2) Действительно точный результат (но сомнительно)")
    else:
        print("✅ SCRR дает ненулевой результат")
    
    # Тест 4: Известная функция с известным ненулевым ответом
    print("\n4️⃣ КОНТРОЛЬНЫЙ ТЕСТ: sqrt(2) * sqrt(2) - 2 (должно быть ~0, но НЕ точно 0)")
    
    sqrt2_approx = 1.4142135623730951  # Приближение sqrt(2)
    
    sqrt2_scrr = SCRR_Tensor.from_float(torch.tensor(sqrt2_approx, dtype=torch.float64), k=8)
    two_scrr = SCRR_Tensor.from_float(torch.tensor(2.0, dtype=torch.float64), k=8)
    
    print(f"sqrt2_scrr компоненты: {sqrt2_scrr.components}")
    
    squared_scrr = sqrt2_scrr * sqrt2_scrr
    print(f"sqrt2² компоненты: {squared_scrr.components}")
    
    result_scrr_tensor = squared_scrr - two_scrr
    print(f"(sqrt2² - 2) компоненты: {result_scrr_tensor.components}")
    
    result_scrr = result_scrr_tensor.to_float().item()
    
    # float64 для сравнения  
    result_float64 = (sqrt2_approx * sqrt2_approx - 2.0)
    
    print(f"\nРезультаты sqrt(2)² - 2:")
    print(f"float64: {result_float64:.20e}")
    print(f"SCRR:    {result_scrr:.20e}")
    
    if abs(result_scrr) < 1e-20 and abs(result_float64) > 1e-17:
        print("🚨 ПОДОЗРИТЕЛЬНО: SCRR дает ноль там где float64 дает ненулевой результат!")
    else:
        print("✅ Результаты согласуются")
    
    print("\n" + "="*80)
    print("🏁 ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("="*80)


if __name__ == "__main__":
    test_scrr_internal_debug() 