"""
Бенчмарк точности SCRR-FX: mpmath -> float64/SCRR -> mpmath сравнение.

Этот тест реализует полный пайплайн:
1. Генерируем случайное число в mpmath (512/1024 бита)
2. Вычисляем функцию и градиент в mpmath (эталон)
3. Конвертируем в float64 и SCRR_Tensor с разными k
4. Повторяем вычисления в этих типах
5. Конвертируем обратно в mpmath и сравниваем с эталоном
"""

import torch
from mpmath import mp
import numpy as np
import random
from typing import Tuple, List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scrr_fx import SCRR_Tensor


def generate_random_mpmath(precision_bits: int = 512) -> mp.mpf:
    """
    Генерирует случайное число в mpmath с заданной точностью.
    
    Args:
        precision_bits: Количество бит точности (512 или 1024)
    
    Returns:
        Случайное число mpmath.mpf с заданной точностью
    """
    # Устанавливаем точность mpmath
    mp.dps = precision_bits // 3  # Примерно 3 десятичных знака на бит
    
    # Генерируем случайное число в диапазоне [1e-100, 1e100]
    mantissa = random.uniform(1.0, 10.0)
    exponent = random.randint(-100, 100)
    
    # Создаем число с высокой точностью
    base_number = mp.mpf(f"{mantissa}e{exponent}")
    
    # Добавляем случайные младшие разряды для максимальной точности
    for _ in range(50):  # Добавляем 50 случайных младших разрядов
        small_addition = mp.mpf(f"{random.uniform(0, 1)}e{random.randint(-200, -150)}")
        base_number += small_addition
    
    return base_number


def generate_long_decimal_mpmath(num_digits: int = 100) -> mp.mpf:
    """
    Генерирует число с огромным количеством знаков после запятой, не представимое в float64.
    """
    digits = [str(random.randint(0, 9)) for _ in range(num_digits)]
    mantissa = '0.' + ''.join(digits)
    return mp.mpf(mantissa)


def test_function_mpmath(x: mp.mpf) -> Tuple[mp.mpf, mp.mpf]:
    """
    Катастрофическое сокращение: f(x) = (x+1)^10 - sum_{k=0}^{10} C(10,k) x^k
    f(x) == 0 в точной арифметике, но float64 даст ошибку.
    Градиент: f'(x) = 10*(x+1)^9 - sum_{k=1}^{10} C(10,k) * k * x^{k-1}
    """
    n = 10
    f_x = mp.power(x + 1, n)
    for k in range(n + 1):
        f_x -= mp.binomial(n, k) * mp.power(x, k)
    # Градиент
    grad = n * mp.power(x + 1, n - 1)
    for k in range(1, n + 1):
        grad -= mp.binomial(n, k) * k * mp.power(x, k - 1)
    return f_x, grad


def convert_mpmath_to_float64(x_mp: mp.mpf) -> float:
    """
    Конвертирует mpmath число в float64 (с потерей точности).
    
    Args:
        x_mp: Число в mpmath
    
    Returns:
        Число в float64
    """
    return float(x_mp)


def convert_float64_to_mpmath(x_float: float) -> mp.mpf:
    """
    Конвертирует float64 обратно в mpmath.
    
    Args:
        x_float: Число в float64
    
    Returns:
        Число в mpmath
    """
    return mp.mpf(x_float)


def convert_mpmath_to_scrr(x_mp: mp.mpf, k: int) -> SCRR_Tensor:
    """
    Конвертирует mpmath число в SCRR_Tensor с заданным k.
    
    Args:
        x_mp: Число в mpmath
        k: Количество компонентов
    
    Returns:
        SCRR_Tensor
    """
    return SCRR_Tensor.from_mpmath(mp, x_mp, k=k)


def convert_scrr_to_mpmath(x_scrr: SCRR_Tensor) -> mp.mpf:
    """
    Конвертирует SCRR_Tensor обратно в mpmath (точная сумма).
    
    Args:
        x_scrr: SCRR_Tensor
    
    Returns:
        Число в mpmath
    """
    # Используем точное восстановление через mpmath
    return x_scrr.to_mpmath(mp)


def test_function_float64(x: float) -> Tuple[float, float]:
    n = 10
    f_x = (x + 1) ** n
    for k in range(n + 1):
        f_x -= float(mp.binomial(n, k)) * (x ** k)
    grad = n * (x + 1) ** (n - 1)
    for k in range(1, n + 1):
        grad -= float(mp.binomial(n, k)) * k * (x ** (k - 1))
    return f_x, grad


def test_function_scrr(x_scrr: SCRR_Tensor) -> Tuple[SCRR_Tensor, SCRR_Tensor]:
    n = 10
    k_prec = x_scrr.precision_k
    
    # Создаем константы через mpmath для точности
    C = [SCRR_Tensor.from_mpmath(mp, mp.binomial(n, i), k=k_prec) for i in range(n + 1)]
    C_grad = [SCRR_Tensor.from_mpmath(mp, mp.binomial(n, i) * i, k=k_prec) for i in range(1, n + 1)]
    
    f_x = (x_scrr + 1) ** n
    for k in range(n + 1):
        f_x = f_x - C[k] * (x_scrr ** k)
    
    grad = SCRR_Tensor.from_mpmath(mp, n, k=k_prec) * (x_scrr + 1) ** (n - 1)
    for k in range(1, n + 1):
        grad = grad - C_grad[k-1] * (x_scrr ** (k - 1))
    
    return f_x, grad


def calculate_relative_error(exact: mp.mpf, computed: mp.mpf) -> float:
    """
    Вычисляет относительную ошибку.
    
    Args:
        exact: Точное значение
        computed: Вычисленное значение
    
    Returns:
        Относительная ошибка
    """
    if exact == 0:
        return float(abs(computed))
    return float(abs((computed - exact) / exact))


def run_precision_benchmark(precision_bits: int = 512, k_values: List[int] = None):
    """
    Запускает полный бенчмарк точности.
    
    Args:
        precision_bits: Точность mpmath в битах
        k_values: Список значений k для SCRR_Tensor
    """
    if k_values is None:
        k_values = [2, 4, 8, 16, 32]
    
    print("="*100)
    print(f"БЕНЧМАРК ТОЧНОСТИ SCRR-FX (катастрофическое сокращение)")
    print(f"Тестируемые k: {k_values}")
    print("="*100)
    
    # Генерируем число с огромным количеством знаков
    print("\n1. ГЕНЕРАЦИЯ СЛОЖНОГО ЧИСЛА")
    x_mp_exact = generate_long_decimal_mpmath(100)
    print(f"Сложное число (mpmath): {x_mp_exact}")
    print(f"Количество знаков после запятой: {str(x_mp_exact)[2:].find('.') if '.' in str(x_mp_exact)[2:] else len(str(x_mp_exact)[2:])}")
    
    # Вычисляем эталон в mpmath
    print("\n2. ВЫЧИСЛЕНИЕ ЭТАЛОНА В MPMATH")
    f_mp_exact, f_prime_mp_exact = test_function_mpmath(x_mp_exact)
    print(f"f(x) = {f_mp_exact}")
    print(f"f'(x) = {f_prime_mp_exact}")
    
    # Результаты для сравнения
    results = {
        'float64': {'input_error': 0.0, 'output_error': 0.0, 'gradient_error': 0.0},
    }
    
    # Тест float64
    print("\n3. ТЕСТ FLOAT64")
    x_float64 = convert_mpmath_to_float64(x_mp_exact)
    f_float64, f_prime_float64 = test_function_float64(x_float64)
    f_float64_mp = convert_float64_to_mpmath(f_float64)
    f_prime_float64_mp = convert_float64_to_mpmath(f_prime_float64)
    
    input_error = calculate_relative_error(x_mp_exact, convert_float64_to_mpmath(x_float64))
    output_error = calculate_relative_error(f_mp_exact, f_float64_mp)
    gradient_error = calculate_relative_error(f_prime_mp_exact, f_prime_float64_mp)
    
    results['float64'] = {
        'input_error': input_error,
        'output_error': output_error,
        'gradient_error': gradient_error
    }
    
    print(f"Входная ошибка: {input_error:.2e}")
    print(f"Выходная ошибка: {output_error:.2e}")
    print(f"Ошибка градиента: {gradient_error:.2e}")
    
    # Тест SCRR_Tensor с разными k
    print("\n4. ТЕСТ SCRR_TENSOR")
    for k in k_values:
        print(f"\n--- k = {k} ---")
        
        # Конвертация в SCRR
        x_scrr = convert_mpmath_to_scrr(x_mp_exact, k)
        
        # Вычисление в SCRR
        f_scrr, f_prime_scrr = test_function_scrr(x_scrr)
        
        # Конвертация обратно в mpmath
        x_scrr_back = convert_scrr_to_mpmath(x_scrr)
        f_scrr_back = convert_scrr_to_mpmath(f_scrr)
        f_prime_scrr_back = convert_scrr_to_mpmath(f_prime_scrr)
        
        # Вычисление ошибок
        input_error = calculate_relative_error(x_mp_exact, x_scrr_back)
        output_error = calculate_relative_error(f_mp_exact, f_scrr_back)
        gradient_error = calculate_relative_error(f_prime_mp_exact, f_prime_scrr_back)
        
        results[f'SCRR_k{k}'] = {
            'input_error': input_error,
            'output_error': output_error,
            'gradient_error': gradient_error
        }
        
        print(f"Входная ошибка: {input_error:.2e}")
        print(f"Выходная ошибка: {output_error:.2e}")
        print(f"Ошибка градиента: {gradient_error:.2e}")
        
        # Проверяем, что SCRR не хуже float64 для входных данных
        if input_error > results['float64']['input_error'] * 1.1:
            print(f"⚠️  ВНИМАНИЕ: SCRR k={k} хуже float64 по входным данным!")
    
    # Сводка результатов
    print("\n" + "="*100)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("="*100)
    
    print(f"{'Тип':<15} {'Вход':<15} {'Выход':<15} {'Градиент':<15}")
    print("-" * 60)
    
    for method, errors in results.items():
        print(f"{method:<15} {errors['input_error']:<15.2e} {errors['output_error']:<15.2e} {errors['gradient_error']:<15.2e}")
    
    # Анализ
    print("\n" + "="*100)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*100)
    
    # Находим лучший результат для каждого типа ошибки
    best_input = min(results.items(), key=lambda x: x[1]['input_error'])
    best_output = min(results.items(), key=lambda x: x[1]['output_error'])
    best_gradient = min(results.items(), key=lambda x: x[1]['gradient_error'])
    
    print(f"Лучшая входная точность: {best_input[0]} (ошибка: {best_input[1]['input_error']:.2e})")
    print(f"Лучшая выходная точность: {best_output[0]} (ошибка: {best_output[1]['output_error']:.2e})")
    print(f"Лучшая точность градиента: {best_gradient[0]} (ошибка: {best_gradient[1]['gradient_error']:.2e})")
    
    # Проверяем сходимость с ростом k
    scrr_results = {k: results[f'SCRR_k{k}'] for k in k_values if f'SCRR_k{k}' in results}
    
    print(f"\nСХОДИМОСТЬ С РОСТОМ K:")
    for k in k_values:
        if f'SCRR_k{k}' in results:
            print(f"k={k:2d}: выходная ошибка = {results[f'SCRR_k{k}']['output_error']:.2e}")
    
    # Проверяем, что ошибка уменьшается с ростом k
    k_list = sorted([k for k in k_values if f'SCRR_k{k}' in results])
    for i in range(1, len(k_list)):
        prev_k = k_list[i-1]
        curr_k = k_list[i]
        prev_error = results[f'SCRR_k{prev_k}']['output_error']
        curr_error = results[f'SCRR_k{curr_k}']['output_error']
        
        if curr_error > prev_error * 1.1:
            print(f"⚠️  ВНИМАНИЕ: Ошибка не уменьшается с ростом k: k={prev_k} -> k={curr_k}")
        else:
            improvement = prev_error / curr_error if curr_error > 0 else float('inf')
            print(f"✅ k={curr_k} лучше k={prev_k} в {improvement:.1f} раз")


def test_scrr_precision_detailed():
    """
    Детальный тест точности SCRR_Tensor на длинных числах.
    Показывает, как SCRR_Tensor восстанавливает числа с огромным количеством знаков.
    """
    print("\n" + "="*100)
    print("ДЕТАЛЬНЫЙ ТЕСТ ТОЧНОСТИ SCRR_TENSOR")
    print("="*100)
    
    from mpmath import mp
    mp.dps = 200  # Высокая точность для вычислений
    
    # Создаем число с огромным количеством знаков после запятой
    long_number = mp.mpf('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
    
    print(f"Исходное число (полностью):")
    print(f"{mp.nstr(long_number, 100)}")
    print(f"Количество знаков после запятой: {len(str(long_number).split('.')[1])}")
    print()
    
    # Тестируем разные k
    for k in [1, 2, 4, 8, 16]:
        print(f"--- k = {k} ---")
        
        # Создаем SCRR_Tensor
        scrr = SCRR_Tensor.from_mpmath(mp, long_number, k=k)
        
        # Восстанавливаем число через правильный метод to_mpmath
        recovered = scrr.to_mpmath(mp)
        
        print(f"Восстановленное число:")
        print(f"{mp.nstr(recovered, 100)}")
        print(f"Ошибка: {abs(long_number - recovered)}")
        print(f"Относительная ошибка: {abs(long_number - recovered) / abs(long_number)}")
        print()
        
        # Проверяем, что ошибка уменьшается с ростом k
        if k > 1:
            expected_error = mp.mpf('1e-16') ** k  # Ожидаемая ошибка ~ float64^k
            actual_error = abs(long_number - recovered)
            print(f"Ожидаемая ошибка ~ {expected_error}")
            print(f"Фактическая ошибка: {actual_error}")
            print(f"Соотношение: {actual_error / expected_error}")
            print()


if __name__ == "__main__":
    # Запускаем детальный тест точности
    test_scrr_precision_detailed()
    
    # Запускаем бенчмарк с разными параметрами
    print("Запуск бенчмарка точности SCRR-FX...")
    
    # Тест 1: Стандартная точность
    run_precision_benchmark(precision_bits=512, k_values=[2, 4, 8, 16, 32])
    
    # Тест 2: Высокая точность
    print("\n" + "="*100)
    print("ТЕСТ С ВЫСОКОЙ ТОЧНОСТЬЮ (1024 бита)")
    print("="*100)
    run_precision_benchmark(precision_bits=1024, k_values=[2, 4, 8, 16, 32, 64]) 