"""
Лабораторна робота №4: Чисельне інтегрування
Laboratory Work #4: Numerical Integration

Казакова Вікторія 34-9
"""

import numpy as np
import matplotlib.pyplot as plt


def rectangle_method(f, a, b, n, method='middle'):
    """
    Метод прямокутників
    Rectangle method for numerical integration
    
    Args:
        f: функція для інтегрування
        a: нижня межа
        b: верхня межа
        n: кількість відрізків
        method: 'left', 'right', або 'middle'
    
    Returns:
        Наближене значення інтеграла
    """
    h = (b - a) / n
    result = 0
    
    for i in range(n):
        if method == 'left':
            x = a + i * h
        elif method == 'right':
            x = a + (i + 1) * h
        else:  # middle
            x = a + (i + 0.5) * h
        result += f(x)
    
    return result * h


def trapezoidal_method(f, a, b, n):
    """
    Метод трапецій
    Trapezoidal method for numerical integration
    
    Args:
        f: функція для інтегрування
        a: нижня межа
        b: верхня межа
        n: кількість відрізків
    
    Returns:
        Наближене значення інтеграла
    """
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    
    for i in range(1, n):
        x = a + i * h
        result += f(x)
    
    return result * h


def simpson_method(f, a, b, n):
    """
    Метод Сімпсона (парабол)
    Simpson's method for numerical integration
    
    Args:
        f: функція для інтегрування
        a: нижня межа
        b: верхня межа
        n: кількість відрізків (має бути парним)
    
    Returns:
        Наближене значення інтеграла
    """
    if n % 2 != 0:
        n += 1  # Забезпечуємо парність
    
    h = (b - a) / n
    result = f(a) + f(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)
    
    return result * h / 3


def monte_carlo_integration(f, a, b, n):
    """
    Метод Монте-Карло
    Monte Carlo method for numerical integration
    
    Args:
        f: функція для інтегрування
        a: нижня межа
        b: верхня межа
        n: кількість випадкових точок
    
    Returns:
        Наближене значення інтеграла
    """
    x_random = np.random.uniform(a, b, n)
    y_values = f(x_random)
    return (b - a) * np.mean(y_values)


def compare_methods(f, a, b, exact_value, n_values):
    """
    Порівняння точності різних методів
    Compare accuracy of different methods
    """
    results = {
        'Прямокутники (середні)': [],
        'Трапеції': [],
        'Сімпсон': [],
        'Монте-Карло': []
    }
    
    errors = {
        'Прямокутники (середні)': [],
        'Трапеції': [],
        'Сімпсон': [],
        'Монте-Карло': []
    }
    
    for n in n_values:
        # Обчислення інтегралів
        rect = rectangle_method(f, a, b, n, 'middle')
        trap = trapezoidal_method(f, a, b, n)
        simp = simpson_method(f, a, b, n)
        mc = monte_carlo_integration(f, a, b, n * 10)
        
        # Збереження результатів
        results['Прямокутники (середні)'].append(rect)
        results['Трапеції'].append(trap)
        results['Сімпсон'].append(simp)
        results['Монте-Карло'].append(mc)
        
        # Обчислення похибок
        errors['Прямокутники (середні)'].append(abs(rect - exact_value))
        errors['Трапеції'].append(abs(trap - exact_value))
        errors['Сімпсон'].append(abs(simp - exact_value))
        errors['Монте-Карло'].append(abs(mc - exact_value))
    
    return results, errors


def demonstrate_integration():
    """Демонстрація методів чисельного інтегрування"""
    
    # Тестова функція: sin(x)
    f = np.sin
    a, b = 0, np.pi
    exact_value = 2.0  # Точне значення інтеграла sin(x) від 0 до π
    
    # Кількість відрізків для порівняння
    n_values = [10, 20, 50, 100, 200]
    
    # Порівняння методів
    results, errors = compare_methods(f, a, b, exact_value, n_values)
    
    # Виведення результатів
    print("=== Результати чисельного інтегрування ===")
    print(f"Функція: sin(x), межі: [{a}, {b}]")
    print(f"Точне значення: {exact_value}\n")
    
    for method_name in results:
        print(f"\n{method_name}:")
        for i, n in enumerate(n_values):
            print(f"  n={n:3d}: Результат = {results[method_name][i]:.8f}, "
                  f"Похибка = {errors[method_name][i]:.2e}")
    
    # Побудова графіків
    plt.figure(figsize=(14, 5))
    
    # Графік 1: Збіжність методів
    plt.subplot(1, 2, 1)
    for method_name in results:
        plt.plot(n_values, results[method_name], 'o-', label=method_name, linewidth=2)
    plt.axhline(y=exact_value, color='r', linestyle='--', label='Точне значення')
    plt.xlabel('Кількість відрізків (n)')
    plt.ylabel('Значення інтеграла')
    plt.title('Збіжність методів чисельного інтегрування')
    plt.legend()
    plt.grid(True)
    
    # Графік 2: Похибки
    plt.subplot(1, 2, 2)
    for method_name in errors:
        plt.loglog(n_values, errors[method_name], 'o-', label=method_name, linewidth=2)
    plt.xlabel('Кількість відрізків (n)')
    plt.ylabel('Абсолютна похибка')
    plt.title('Похибки методів (логарифмічна шкала)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('integration_results.png', dpi=300, bbox_inches='tight')
    print("\n\nГрафік збережено у файл: integration_results.png")
    
    # Додатковий приклад: інтегрування складнішої функції
    print("\n\n=== Додатковий приклад: ∫(x² + 2x + 1)dx від 0 до 2 ===")
    f2 = lambda x: x**2 + 2*x + 1
    a2, b2 = 0, 2
    exact2 = 22/3  # Точне значення: [x³/3 + x² + x] від 0 до 2
    n = 100
    
    print(f"Точне значення: {exact2:.8f}")
    print(f"\nРезультати (n={n}):")
    print(f"Прямокутники: {rectangle_method(f2, a2, b2, n, 'middle'):.8f}")
    print(f"Трапеції: {trapezoidal_method(f2, a2, b2, n):.8f}")
    print(f"Сімпсон: {simpson_method(f2, a2, b2, n):.8f}")
    print(f"Монте-Карло: {monte_carlo_integration(f2, a2, b2, n*10):.8f}")


if __name__ == "__main__":
    demonstrate_integration()
