"""
Тестування лабораторних робіт 3-4
Testing Laboratory Works 3-4

Казакова Вікторія 34-9
"""

import numpy as np
import sys
from lab3_interpolation import (
    lagrange_interpolation,
    newton_interpolation,
    newton_divided_difference
)
from lab4_numerical_integration import (
    rectangle_method,
    trapezoidal_method,
    simpson_method,
    monte_carlo_integration
)


def test_interpolation():
    """Тестування методів інтерполяції"""
    print("=== Тестування інтерполяції ===\n")
    
    # Тест 1: Лінійна функція
    print("Тест 1: Лінійна функція y = 2x + 1")
    x_points = np.array([0, 1, 2])
    y_points = np.array([1, 3, 5])
    test_x = 1.5
    expected = 2 * test_x + 1
    
    lagrange_result = lagrange_interpolation(x_points, y_points, test_x)
    newton_result = newton_interpolation(x_points, y_points, test_x)
    
    print(f"  Очікуване значення: {expected:.6f}")
    print(f"  Лагранж: {lagrange_result:.6f} (похибка: {abs(lagrange_result - expected):.2e})")
    print(f"  Ньютон: {newton_result:.6f} (похибка: {abs(newton_result - expected):.2e})")
    
    success1 = abs(lagrange_result - expected) < 1e-10 and abs(newton_result - expected) < 1e-10
    print(f"  Статус: {'✓ ПРОЙДЕНО' if success1 else '✗ НЕ ПРОЙДЕНО'}\n")
    
    # Тест 2: Квадратична функція
    print("Тест 2: Квадратична функція y = x²")
    x_points = np.array([0, 1, 2, 3])
    y_points = np.array([0, 1, 4, 9])
    test_x = 2.5
    expected = test_x ** 2
    
    lagrange_result = lagrange_interpolation(x_points, y_points, test_x)
    newton_result = newton_interpolation(x_points, y_points, test_x)
    
    print(f"  Очікуване значення: {expected:.6f}")
    print(f"  Лагранж: {lagrange_result:.6f} (похибка: {abs(lagrange_result - expected):.2e})")
    print(f"  Ньютон: {newton_result:.6f} (похибка: {abs(newton_result - expected):.2e})")
    
    success2 = abs(lagrange_result - expected) < 1e-10 and abs(newton_result - expected) < 1e-10
    print(f"  Статус: {'✓ ПРОЙДЕНО' if success2 else '✗ НЕ ПРОЙДЕНО'}\n")
    
    return success1 and success2


def test_integration():
    """Тестування методів чисельного інтегрування"""
    print("=== Тестування чисельного інтегрування ===\n")
    
    # Тест 1: Константна функція
    print("Тест 1: Константна функція y = 5 від 0 до 10")
    f = lambda x: 5
    a, b = 0, 10
    exact = 50
    n = 100
    
    rect_result = rectangle_method(f, a, b, n, 'middle')
    trap_result = trapezoidal_method(f, a, b, n)
    simp_result = simpson_method(f, a, b, n)
    
    print(f"  Очікуване значення: {exact}")
    print(f"  Прямокутники: {rect_result:.6f} (похибка: {abs(rect_result - exact):.2e})")
    print(f"  Трапеції: {trap_result:.6f} (похибка: {abs(trap_result - exact):.2e})")
    print(f"  Сімпсон: {simp_result:.6f} (похибка: {abs(simp_result - exact):.2e})")
    
    success1 = (abs(rect_result - exact) < 1e-10 and 
                abs(trap_result - exact) < 1e-10 and 
                abs(simp_result - exact) < 1e-10)
    print(f"  Статус: {'✓ ПРОЙДЕНО' if success1 else '✗ НЕ ПРОЙДЕНО'}\n")
    
    # Тест 2: Лінійна функція
    print("Тест 2: Лінійна функція y = 2x від 0 до 5")
    f = lambda x: 2 * x
    a, b = 0, 5
    exact = 25  # [x²]₀⁵ = 25
    n = 100
    
    rect_result = rectangle_method(f, a, b, n, 'middle')
    trap_result = trapezoidal_method(f, a, b, n)
    simp_result = simpson_method(f, a, b, n)
    
    print(f"  Очікуване значення: {exact}")
    print(f"  Прямокутники: {rect_result:.6f} (похибка: {abs(rect_result - exact):.2e})")
    print(f"  Трапеції: {trap_result:.6f} (похибка: {abs(trap_result - exact):.2e})")
    print(f"  Сімпсон: {simp_result:.6f} (похибка: {abs(simp_result - exact):.2e})")
    
    success2 = (abs(rect_result - exact) < 1e-6 and 
                abs(trap_result - exact) < 1e-10 and 
                abs(simp_result - exact) < 1e-10)
    print(f"  Статус: {'✓ ПРОЙДЕНО' if success2 else '✗ НЕ ПРОЙДЕНО'}\n")
    
    # Тест 3: Синус
    print("Тест 3: sin(x) від 0 до π")
    f = np.sin
    a, b = 0, np.pi
    exact = 2.0
    n = 100
    
    rect_result = rectangle_method(f, a, b, n, 'middle')
    trap_result = trapezoidal_method(f, a, b, n)
    simp_result = simpson_method(f, a, b, n)
    
    print(f"  Очікуване значення: {exact}")
    print(f"  Прямокутники: {rect_result:.6f} (похибка: {abs(rect_result - exact):.2e})")
    print(f"  Трапеції: {trap_result:.6f} (похибка: {abs(trap_result - exact):.2e})")
    print(f"  Сімпсон: {simp_result:.6f} (похибка: {abs(simp_result - exact):.2e})")
    
    success3 = (abs(rect_result - exact) < 1e-4 and 
                abs(trap_result - exact) < 1e-3 and 
                abs(simp_result - exact) < 1e-6)
    print(f"  Статус: {'✓ ПРОЙДЕНО' if success3 else '✗ НЕ ПРОЙДЕНО'}\n")
    
    return success1 and success2 and success3


def run_all_tests():
    """Запуск всіх тестів"""
    print("\n" + "="*60)
    print("  ТЕСТУВАННЯ ЛАБОРАТОРНИХ РОБІТ 3-4")
    print("  Казакова Вікторія 34-9")
    print("="*60 + "\n")
    
    interpolation_passed = test_interpolation()
    integration_passed = test_integration()
    
    print("="*60)
    print("  ПІДСУМОК")
    print("="*60)
    print(f"Інтерполяція: {'✓ ВСІ ТЕСТИ ПРОЙДЕНО' if interpolation_passed else '✗ ДЕЯКІ ТЕСТИ НЕ ПРОЙДЕНО'}")
    print(f"Інтегрування: {'✓ ВСІ ТЕСТИ ПРОЙДЕНО' if integration_passed else '✗ ДЕЯКІ ТЕСТИ НЕ ПРОЙДЕНО'}")
    print("="*60 + "\n")
    
    if interpolation_passed and integration_passed:
        print("✓ УСПІШНО: Всі тести пройдено!")
        return 0
    else:
        print("✗ ПОМИЛКА: Деякі тести не пройдено")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
