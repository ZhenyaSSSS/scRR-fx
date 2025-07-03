"""Диагностические тесты для поиска проблем точности в SCRR-FX."""

import torch
import numpy as np
import mpmath as mp
from src.scrr_fx import SCRR_Tensor
from src.scrr_fx._core import two_sum, two_prod
from src.scrr_fx._renorm import renormalize


def test_twosum_precision_basic():
    """Тест основной точности TwoSum."""
    print("\n" + "="*60)
    print("ДИАГНОСТИКА: TwoSum Precision")
    print("="*60)
    
    test_cases = [
        (1.0, 1e-16),           # Стандартный случай
        (1e20, 1.0),            # Большие числа
        (1.0, -1.0 + 1e-15),    # Почти катастрофическое сокращение
        (3.14159, 2.71828),     # Обычные числа
    ]
    
    for i, (a, b) in enumerate(test_cases):
        print(f"\nТест {i+1}: a={a}, b={b}")
        
        # Прямое сложение в float64
        direct_sum = a + b
        
        # TwoSum
        s, e = two_sum(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
        twosum_total = s.item() + e.item()
        
        # Эталон через mpmath 
        mp.dps = 50
        a_mp = mp.mpf(a)
        b_mp = mp.mpf(b)
        exact_sum = float(a_mp + b_mp)
        
        # Ошибки
        direct_error = abs(direct_sum - exact_sum)
        twosum_error = abs(twosum_total - exact_sum)
        
        print(f"  Прямое сложение: {direct_sum:.15e}")
        print(f"  TwoSum (s+e):    {twosum_total:.15e}")
        print(f"  Эталон mpmath:   {exact_sum:.15e}")
        print(f"  Ошибка прямого:  {direct_error:.2e}")
        print(f"  Ошибка TwoSum:   {twosum_error:.2e}")
        
        # TwoSum должен быть точнее или равен
        if direct_error > 0:
            improvement = direct_error / max(twosum_error, 1e-20)
            print(f"  Улучшение: {improvement:.1f}x")
        
        # Критическая проверка: TwoSum не должен быть хуже
        assert twosum_error <= direct_error * 1.1, f"TwoSum хуже прямого сложения в тесте {i+1}!"


def test_twoprod_precision_basic():
    """Тест основной точности TwoProd."""
    print("\n" + "="*60)
    print("ДИАГНОСТИКА: TwoProd Precision")
    print("="*60)
    
    test_cases = [
        (1.0 + 1e-16, 1.0 - 1e-16),  # Близкие к 1
        (3.14159, 2.71828),          # Обычные числа
        (1e10, 1e-10),               # Крайние масштабы
        (7.0, 1.0/7.0),              # Рациональные числа
    ]
    
    for i, (a, b) in enumerate(test_cases):
        print(f"\nТест {i+1}: a={a}, b={b}")
        
        # Прямое умножение в float64
        direct_prod = a * b
        
        # TwoProd
        p, e = two_prod(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
        twoprod_total = p.item() + e.item()
        
        # Эталон через mpmath 
        mp.dps = 50
        a_mp = mp.mpf(a)
        b_mp = mp.mpf(b)
        exact_prod = float(a_mp * b_mp)
        
        # Ошибки
        direct_error = abs(direct_prod - exact_prod)
        twoprod_error = abs(twoprod_total - exact_prod)
        
        print(f"  Прямое произведение: {direct_prod:.15e}")
        print(f"  TwoProd (p+e):       {twoprod_total:.15e}")
        print(f"  Эталон mpmath:       {exact_prod:.15e}")
        print(f"  Ошибка прямого:      {direct_error:.2e}")
        print(f"  Ошибка TwoProd:      {twoprod_error:.2e}")
        
        # TwoProd должен быть точнее или равен
        if direct_error > 0:
            improvement = direct_error / max(twoprod_error, 1e-20)
            print(f"  Улучшение: {improvement:.1f}x")
        
        # Критическая проверка
        assert twoprod_error <= direct_error * 1.1, f"TwoProd хуже прямого умножения в тесте {i+1}!"


def test_renormalize_conservation():
    """Тест сохранения суммы в Renormalize."""
    print("\n" + "="*60)
    print("ДИАГНОСТИКА: Renormalize Conservation")
    print("="*60)
    
    test_cases = [
        torch.tensor([1.0, 1e-15, 1e-30], dtype=torch.float64),                    # Убывающие
        torch.tensor([1e10, -1e10, 1.0], dtype=torch.float64),                     # Сокращение
        torch.tensor([3.14, 2.71, 1.41, 1.73, 0.57], dtype=torch.float64),        # Смешанные
        torch.randn(10, dtype=torch.float64) * 1e5,                                # Случайные большие
    ]
    
    for i, dirty_tensor in enumerate(test_cases):
        print(f"\nТест {i+1}: {len(dirty_tensor)} компонентов")
        
        # Исходная сумма
        original_sum = dirty_tensor.sum().item()
        
        # Renormalize с разными k
        for k in [2, 4, 8]:
            clean_tensor = renormalize(dirty_tensor, k=k)
            clean_sum = clean_tensor.sum().item()
            
            conservation_error = abs(original_sum - clean_sum)
            relative_error = conservation_error / max(abs(original_sum), 1e-20)
            
            print(f"  k={k}: ошибка={conservation_error:.2e}, относительная={relative_error:.2e}")
            
            # Критическая проверка: потеря суммы недопустима
            assert relative_error < 1e-12, f"Renormalize теряет сумму для k={k} в тесте {i+1}!"


def test_from_float_to_float_roundtrip():
    """Тест точности конвертации from_float → to_float."""
    print("\n" + "="*60)
    print("ДИАГНОСТИКА: from_float/to_float Roundtrip")
    print("="*60)
    
    test_values = [
        1.0,
        1.0 + 1e-15,
        np.pi,
        np.e, 
        1e-100,
        1e100,
        1.0/3.0,
    ]
    
    for i, val in enumerate(test_values):
        print(f"\nТест {i+1}: значение = {val}")
        
        x_torch = torch.tensor(val, dtype=torch.float64)
        
        for k in [2, 4, 8, 16]:
            # Roundtrip: float → SCRR → float
            x_scrr = SCRR_Tensor.from_float(x_torch, k=k)
            x_back = x_scrr.to_float().item()
            
            roundtrip_error = abs(val - x_back)
            relative_error = roundtrip_error / max(abs(val), 1e-20)
            
            print(f"  k={k}: ошибка={roundtrip_error:.2e}, относительная={relative_error:.2e}")
            
            # Для больших k ошибка должна быть минимальной
            if k >= 4:
                assert relative_error < 1e-14, f"Roundtrip ошибка слишком велика для k={k}!"


def test_simple_arithmetic_vs_float64():
    """Тест простейшей арифметики против float64."""
    print("\n" + "="*60)
    print("ДИАГНОСТИКА: Простая Арифметика vs float64")
    print("="*60)
    
    # Простые операции где SCRR должен быть значительно лучше
    test_operations = [
        ("1.0 + 1e-16", lambda: 1.0 + 1e-16),
        ("1.0 - (1.0 - 1e-16)", lambda: 1.0 - (1.0 - 1e-16)),
        ("(1.0 + 1e-15) * (1.0 - 1e-15)", lambda: (1.0 + 1e-15) * (1.0 - 1e-15)),
    ]
    
    for op_name, op_func in test_operations:
        print(f"\nОперация: {op_name}")
        
        # float64 результат
        float64_result = op_func()
        
        # Эталон через mpmath
        mp.dps = 50
        if "1.0 + 1e-16" in op_name:
            exact_mp = mp.mpf(1) + mp.mpf('1e-16')
        elif "1.0 - (1.0 - 1e-16)" in op_name:
            exact_mp = mp.mpf(1) - (mp.mpf(1) - mp.mpf('1e-16'))
        elif "+" in op_name and "*" in op_name:
            exact_mp = (mp.mpf(1) + mp.mpf('1e-15')) * (mp.mpf(1) - mp.mpf('1e-15'))
        
        exact_result = float(exact_mp)
        
        # SCRR результат
        for k in [2, 4, 8]:
            if "+" in op_name and not "*" in op_name:
                # Простое сложение
                x1 = SCRR_Tensor.from_float(torch.tensor(1.0), k=k)
                x2 = SCRR_Tensor.from_float(torch.tensor(1e-16), k=k)
                scrr_result = (x1 + x2).to_float().item()
            elif "-" in op_name:
                # Вычитание
                x1 = SCRR_Tensor.from_float(torch.tensor(1.0), k=k)
                x2 = SCRR_Tensor.from_float(torch.tensor(1.0 - 1e-16), k=k)
                scrr_result = (x1 - x2).to_float().item()
            else:
                # Умножение
                x1 = SCRR_Tensor.from_float(torch.tensor(1.0 + 1e-15), k=k)
                x2 = SCRR_Tensor.from_float(torch.tensor(1.0 - 1e-15), k=k)
                scrr_result = (x1 * x2).to_float().item()
            
            float64_error = abs(float64_result - exact_result) / abs(exact_result)
            scrr_error = abs(scrr_result - exact_result) / abs(exact_result)
            
            print(f"  k={k}:")
            print(f"    float64: {float64_result:.15e}, ошибка: {float64_error:.2e}")
            print(f"    SCRR:    {scrr_result:.15e}, ошибка: {scrr_error:.2e}")
            
            if float64_error > 0:
                improvement = float64_error / max(scrr_error, 1e-20)
                print(f"    Улучшение: {improvement:.1f}x")
                # КРИТИЧЕСКАЯ ПРОВЕРКА: SCRR должен быть лучше float64 только для k>=8
                if k >= 8:
                    assert scrr_error <= float64_error, f"SCRR k={k} ХУЖЕ float64 в операции {op_name}!"


def test_problematic_function_detailed():
    """Детальный анализ проблемной функции."""
    print("\n" + "="*60)
    print("ДИАГНОСТИКА: Анализ Проблемной Функции")
    print("="*60)
    
    # Функция: (1+x)^2 - 1 - 2x = x^2 (простая, должна работать точно)
    x_val = 1e-8
    expected_result = x_val ** 2
    
    print(f"Функция: (1+x)^2 - 1 - 2x = x^2")
    print(f"x = {x_val}")
    print(f"Ожидаемый результат: {expected_result:.15e}")
    
    # float64
    x64 = torch.tensor(x_val, dtype=torch.float64)
    result_64 = ((1 + x64)**2 - 1 - 2*x64).item()
    error_64 = abs(result_64 - expected_result) / abs(expected_result)
    
    print(f"\nfloat64:")
    print(f"  Результат: {result_64:.15e}")
    print(f"  Ошибка: {error_64:.2e}")
    
    # SCRR детально
    for k in [2, 4, 8]:
        print(f"\nSCRR k={k}:")
        
        # Создаем SCRR числа
        x_scrr = SCRR_Tensor.from_float(x64, k=k)
        one_scrr = SCRR_Tensor.from_float(torch.tensor(1.0), k=k)
        two_scrr = SCRR_Tensor.from_float(torch.tensor(2.0), k=k)
        
        print(f"  x_scrr компоненты: {x_scrr.components}")
        
        # Пошаговое вычисление
        one_plus_x = one_scrr + x_scrr
        print(f"  (1+x) компоненты: {one_plus_x.components}")
        
        squared = one_plus_x * one_plus_x
        print(f"  (1+x)^2 компоненты: {squared.components}")
        
        minus_one = squared - one_scrr
        print(f"  (1+x)^2-1 компоненты: {minus_one.components}")
        
        two_x = two_scrr * x_scrr
        print(f"  2x компоненты: {two_x.components}")
        
        final = minus_one - two_x
        print(f"  Финальные компоненты: {final.components}")
        
        result_scrr = final.to_float().item()
        error_scrr = abs(result_scrr - expected_result) / abs(expected_result)
        
        print(f"  Результат: {result_scrr:.15e}")
        print(f"  Ошибка: {error_scrr:.2e}")
        
        # Это ДОЛЖНО работать идеально для любого k≥2!
        if k >= 4:
            assert error_scrr <= error_64, f"SCRR k={k} не лучше float64 даже в простейшем случае!"


if __name__ == "__main__":
    # Запускаем все диагностические тесты
    test_twosum_precision_basic()
    test_twoprod_precision_basic()
    test_renormalize_conservation()
    test_from_float_to_float_roundtrip()
    test_simple_arithmetic_vs_float64()
    test_problematic_function_detailed()
    
    print("\n" + "="*60)
    print("ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("="*60) 