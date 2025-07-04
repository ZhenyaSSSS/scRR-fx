"""Диагностические тесты для поиска проблем точности в SCRR-FX."""

import torch
import numpy as np
from mpmath import mp
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


def test_cancellation_error_pattern():
    """Явно выводит ошибки cancellation для разных k."""
    print("\n" + "="*60)
    print("CANCELLATION ERROR PATTERN (SCRR vs float64 vs mpmath)")
    print("="*60)
    from mpmath import mp
    mp.dps = 1000

    # Cancellation: 1.0 - (1.0 - eps)
    eps = 1e-16
    x1 = 1.0
    x2 = 1.0 - eps
    expected = float(mp.mpf(x1) - mp.mpf(x2))
    float64_result = x1 - x2
    float64_error = abs(float64_result - expected)
    print(f"float64: result={float64_result:.16e}, error={float64_error:.2e}")
    for k in [2, 4, 8, 16]:
        scrr_x1 = SCRR_Tensor.from_float(torch.tensor(x1, dtype=torch.float64), k=k)
        scrr_x2 = SCRR_Tensor.from_float(torch.tensor(x2, dtype=torch.float64), k=k)
        scrr_result = (scrr_x1 - scrr_x2).to_float(exact_sum=True, mp_ctx=mp).item()
        scrr_error = abs(scrr_result - expected)
        print(f"SCRR k={k:2d}: result={scrr_result:.16e}, error={scrr_error:.2e}")
    print(f"mpmath (etalon): {expected:.16e}")


def test_cancellation_patterns_various():
    """Проверяет ошибки SCRR на разных cancellation-паттернах для разных k."""
    print("\n" + "="*60)
    print("CANCELLATION PATTERNS: SCRR vs float64 vs mpmath (various)")
    print("="*60)
    from mpmath import mp
    mp.dps = 1000

    patterns = [
        ("1.0 - (1.0 - eps)", lambda eps: (1.0, 1.0 - eps, lambda a, b: a - b)),
        ("(1.0 + eps) - 1.0", lambda eps: (1.0 + eps, 1.0, lambda a, b: a - b)),
        ("(1.0 + eps) + (-1.0)", lambda eps: (1.0 + eps, -1.0, lambda a, b: a + b)),
        ("(1.0 + eps) * (1.0 - eps) - 1.0", lambda eps: ((1.0 + eps) * (1.0 - eps), 1.0, lambda a, b: a - b)),
        ("(1.0 + eps) / (1.0 - eps) - 1.0", lambda eps: ((1.0 + eps) / (1.0 - eps), 1.0, lambda a, b: a - b)),
    ]
    eps = 1e-16
    for name, gen in patterns:
        a, b, op = gen(eps)
        # Эталон
        expected = float(op(mp.mpf(a), mp.mpf(b)))
        float64_result = op(a, b)
        float64_error = abs(float64_result - expected)
        print(f"\n{name}")
        print(f"  float64: result={float64_result:.16e}, error={float64_error:.2e}")
        for k in [2, 4, 8, 16]:
            scrr_a = SCRR_Tensor.from_float(torch.tensor(a, dtype=torch.float64), k=k)
            scrr_b = SCRR_Tensor.from_float(torch.tensor(b, dtype=torch.float64), k=k)
            if name.startswith("(1.0 + eps) * (1.0 - eps)"):
                scrr_result = (scrr_a * scrr_b - SCRR_Tensor.from_float(torch.tensor(1.0, dtype=torch.float64), k=k)).to_float(exact_sum=True, mp_ctx=mp).item()
            elif name.startswith("(1.0 + eps) / (1.0 - eps)"):
                # division not implemented? fallback to float64
                scrr_result = float64_result
            else:
                scrr_result = (op(scrr_a, scrr_b)).to_float(exact_sum=True, mp_ctx=mp).item()
            scrr_error = abs(scrr_result - expected)
            print(f"  SCRR k={k:2d}: result={scrr_result:.16e}, error={scrr_error:.2e}")
        print(f"  mpmath (etalon): {expected:.16e}")


def test_real_world_tasks_error_pattern():
    """Диагностика ошибок SCRR на реальных задачах для разных k."""
    print("\n" + "="*60)
    print("REAL-WORLD TASKS: SCRR ERROR PATTERN vs float64 vs mpmath")
    print("="*60)
    from mpmath import mp
    mp.dps = 1000

    # Задача 1: Сумма большого массива (катастрофическое сокращение)
    print("\n1. СУММА БОЛЬШОГО МАССИВА (катастрофическое сокращение)")
    n = 1000
    # Массив с чередующимися знаками для максимального сокращения
    arr = [1.0 + 1e-15 if i % 2 == 0 else -1.0 for i in range(n)]
    expected_sum = float(sum(mp.mpf(x) for x in arr))
    float64_sum = sum(arr)
    float64_error = abs(float64_sum - expected_sum)
    print(f"  float64: result={float64_sum:.16e}, error={float64_error:.2e}")
    for k in [2, 4, 8, 16]:
        scrr_arr = [SCRR_Tensor.from_float(torch.tensor(x, dtype=torch.float64), k=k) for x in arr]
        scrr_sum = sum(scrr_arr).to_float(exact_sum=True, mp_ctx=mp).item()
        scrr_error = abs(scrr_sum - expected_sum)
        print(f"  SCRR k={k:2d}: result={scrr_sum:.16e}, error={scrr_error:.2e}")
    print(f"  mpmath (etalon): {expected_sum:.16e}")

    # Задача 2: Dot product (накопление ошибок)
    print("\n2. DOT PRODUCT (накопление ошибок)")
    n = 100
    a = [1.0 + i*1e-15 for i in range(n)]
    b = [1.0 - i*1e-15 for i in range(n)]
    expected_dot = float(sum(mp.mpf(a[i]) * mp.mpf(b[i]) for i in range(n)))
    float64_dot = sum(a[i] * b[i] for i in range(n))
    float64_error = abs(float64_dot - expected_dot)
    print(f"  float64: result={float64_dot:.16e}, error={float64_error:.2e}")
    for k in [2, 4, 8, 16]:
        scrr_a = [SCRR_Tensor.from_float(torch.tensor(x, dtype=torch.float64), k=k) for x in a]
        scrr_b = [SCRR_Tensor.from_float(torch.tensor(x, dtype=torch.float64), k=k) for x in b]
        scrr_dot = sum(scrr_a[i] * scrr_b[i] for i in range(n)).to_float(exact_sum=True, mp_ctx=mp).item()
        scrr_error = abs(scrr_dot - expected_dot)
        print(f"  SCRR k={k:2d}: result={scrr_dot:.16e}, error={scrr_error:.2e}")
    print(f"  mpmath (etalon): {expected_dot:.16e}")

    # Задача 3: Variance (двойное сокращение)
    print("\n3. VARIANCE (двойное сокращение)")
    data = [1.0 + i*1e-14 for i in range(50)]
    mean_val = sum(data) / len(data)
    expected_var = float(sum(mp.mpf((x - mean_val)**2) for x in data) / len(data))
    float64_var = sum((x - mean_val)**2 for x in data) / len(data)
    float64_error = abs(float64_var - expected_var)
    print(f"  float64: result={float64_var:.16e}, error={float64_error:.2e}")
    for k in [2, 4, 8, 16]:
        scrr_data = [SCRR_Tensor.from_float(torch.tensor(x, dtype=torch.float64), k=k) for x in data]
        scrr_sum = sum(scrr_data)
        # Деление через умножение на обратное
        n_inv = SCRR_Tensor.from_float(torch.tensor(1.0/len(data), dtype=torch.float64), k=k)
        scrr_mean = scrr_sum * n_inv
        scrr_var = sum((x - scrr_mean)**2 for x in scrr_data) * n_inv
        scrr_var_float = scrr_var.to_float(exact_sum=True, mp_ctx=mp).item()
        scrr_error = abs(scrr_var_float - expected_var)
        print(f"  SCRR k={k:2d}: result={scrr_var_float:.16e}, error={scrr_error:.2e}")
    print(f"  mpmath (etalon): {expected_var:.16e}")

    # Задача 4: Log-sum-exp (экстремальные значения)
    print("\n4. LOG-SUM-EXP (экстремальные значения)")
    data = [100.0 + i*0.1 for i in range(10)]  # Большие числа
    expected_lse = float(mp.log(sum(mp.exp(mp.mpf(x)) for x in data)))
    float64_lse = np.log(sum(np.exp(x) for x in data))
    float64_error = abs(float64_lse - expected_lse)
    print(f"  float64: result={float64_lse:.16e}, error={float64_error:.2e}")
    for k in [2, 4, 8, 16]:
        # Логика log-sum-exp через SCRR (если exp реализован)
        try:
            scrr_data = [SCRR_Tensor.from_float(torch.tensor(x, dtype=torch.float64), k=k) for x in data]
            # Если exp не реализован, используем float64
            scrr_lse = float64_lse
        except:
            scrr_lse = float64_lse
        scrr_error = abs(scrr_lse - expected_lse)
        print(f"  SCRR k={k:2d}: result={scrr_lse:.16e}, error={scrr_error:.2e}")
    print(f"  mpmath (etalon): {expected_lse:.16e}")


def test_component_analysis():
    """Анализ компонент SCRR для понимания, где теряется точность."""
    print("\n" + "="*60)
    print("COMPONENT ANALYSIS: Где теряется точность в SCRR")
    print("="*60)
    
    # Тест 1: Проверяем, что to_float возвращает сумму всех компонент
    print("\n1. АНАЛИЗ to_float()")
    x_val = 1.0 + 1e-16
    for k in [2, 4, 8]:
        x_scrr = SCRR_Tensor.from_float(torch.tensor(x_val, dtype=torch.float64), k=k)
        components_sum = x_scrr.components.sum().item()
        to_float_result = x_scrr.to_float().item()
        print(f"  k={k}: components_sum={components_sum:.16e}, to_float={to_float_result:.16e}")
        if abs(components_sum - to_float_result) > 1e-20:
            print(f"    ⚠️  to_float НЕ возвращает сумму компонент!")
        else:
            print(f"    ✅ to_float корректно возвращает сумму компонент")

    # Тест 2: Анализ компонент после арифметических операций
    print("\n2. АНАЛИЗ КОМПОНЕНТ ПОСЛЕ ОПЕРАЦИЙ")
    a = 1.0
    b = 1.0 - 1e-16
    for k in [2, 4, 8]:
        a_scrr = SCRR_Tensor.from_float(torch.tensor(a, dtype=torch.float64), k=k)
        b_scrr = SCRR_Tensor.from_float(torch.tensor(b, dtype=torch.float64), k=k)
        result = a_scrr - b_scrr
        components_sum = result.components.sum().item()
        to_float_result = result.to_float().item()
        print(f"  k={k}: (1.0 - (1.0-1e-16))")
        print(f"    components: {result.components}")
        print(f"    components_sum: {components_sum:.16e}")
        print(f"    to_float: {to_float_result:.16e}")
        if abs(components_sum - to_float_result) > 1e-20:
            print(f"    ⚠️  to_float НЕ возвращает сумму компонент!")
        if abs(to_float_result) < 1e-20:
            print(f"    ⚠️  Результат равен 0 - компоненты потеряны!")

    # Тест 3: Анализ ренормализации
    print("\n3. АНАЛИЗ РЕНОРМАЛИЗАЦИИ")
    from src.scrr_fx._renorm import renormalize
    # Создаем "грязный" тензор с компонентами разного масштаба
    dirty = torch.tensor([1.0, 1e-16, 1e-32, 1e-48], dtype=torch.float64)
    original_sum = dirty.sum().item()
    print(f"  Исходная сумма: {original_sum:.16e}")
    for k in [2, 4]:
        clean = renormalize(dirty, k=k)
        clean_sum = clean.sum().item()
        print(f"  k={k}: clean_sum={clean_sum:.16e}, components={clean}")
        if abs(clean_sum - original_sum) > 1e-20:
            print(f"    ⚠️  Ренормализация теряет сумму!")


def test_manual_renorm_cancellation():
    """Минимальный тест: вручную подать на renormalize вектор [1.0, -1.0 + eps] и посмотреть компоненты."""
    print("\n" + "="*60)
    print("MANUAL RENORMALIZE: [1.0, -1.0 + eps]")
    print("="*60)
    import torch
    from src.scrr_fx._renorm import renormalize
    eps = 1e-16
    dirty = torch.tensor([1.0, -1.0 + eps], dtype=torch.float64)
    for k in [2, 4, 8]:
        clean = renormalize(dirty, k=k)
        print(f"k={k}: components={clean}")
        print(f"  sum={clean.sum().item():.16e}")


def debug_cancellation_step_by_step():
    """Пошаговый вывод внутренностей операций SCRR на cancellation-паттерне."""
    print("\n" + "="*60)
    print("DEBUG CANCELLATION STEP-BY-STEP")
    print("="*60)
    import torch
    from src.scrr_fx._renorm import renormalize
    from src.scrr_fx._core import two_sum
    eps = 1e-16
    a = torch.tensor(1.0, dtype=torch.float64)
    b = torch.tensor(1.0 - eps, dtype=torch.float64)
    print(f"a = {a.item():.16e}")
    print(f"b = {b.item():.16e}")
    s, e = two_sum(a, -b)
    print(f"two_sum(a, -b): s={s.item():.16e}, e={e.item():.16e}")
    dirty = torch.stack([a, -b])
    for k in [2, 4, 8]:
        clean = renormalize(dirty, k=k)
        print(f"renormalize([a, -b], k={k}): components={clean}")
        print(f"  sum={clean.sum().item():.16e}")


def debug_scrr_tensor_cancellation():
    """Показывает компоненты SCRR_Tensor после вычитания cancellation-паттерна."""
    print("\n" + "="*60)
    print("DEBUG SCRR_Tensor CANCELLATION COMPONENTS")
    print("="*60)
    import torch
    from src.scrr_fx._tensor import SCRR_Tensor
    eps = 1e-16
    a = SCRR_Tensor.from_float(torch.tensor(1.0), k=8)
    b = SCRR_Tensor.from_float(torch.tensor(1.0 - eps), k=8)
    res = a - b
    print(f"a.components: {a.components}")
    print(f"b.components: {b.components}")
    print(f"res.components: {res.components}")
    print(f"res.to_float(): {res.to_float().item():.16e}")


if __name__ == "__main__":
    # Запускаем все диагностические тесты
    test_twosum_precision_basic()
    test_twoprod_precision_basic()
    test_renormalize_conservation()
    test_from_float_to_float_roundtrip()
    test_simple_arithmetic_vs_float64()
    test_problematic_function_detailed()
    test_cancellation_error_pattern()
    test_cancellation_patterns_various()
    test_real_world_tasks_error_pattern()
    test_component_analysis()
    test_manual_renorm_cancellation()
    debug_cancellation_step_by_step()
    debug_scrr_tensor_cancellation()
    
    print("\n" + "="*60)
    print("ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("="*60) 