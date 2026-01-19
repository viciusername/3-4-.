"""
Лабораторна робота №3: Методи інтерполяції
Laboratory Work #3: Interpolation Methods

Казакова Вікторія 34-9
"""

import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(x_points, y_points, x):
    """
    Інтерполяція методом Лагранжа
    Lagrange interpolation method
    
    Args:
        x_points: масив точок x
        y_points: масив точок y
        x: точка для обчислення
    
    Returns:
        Інтерпольоване значення
    """
    n = len(x_points)
    result = 0
    
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    
    return result


def newton_divided_difference(x_points, y_points):
    """
    Обчислення розділених різниць Ньютона
    Calculate Newton's divided differences
    
    Args:
        x_points: масив точок x
        y_points: масив точок y
    
    Returns:
        Таблиця розділених різниць
    """
    n = len(x_points)
    table = np.zeros((n, n))
    table[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / \
                         (x_points[i + j] - x_points[i])
    
    return table


def newton_interpolation(x_points, y_points, x):
    """
    Інтерполяція методом Ньютона
    Newton interpolation method
    
    Args:
        x_points: масив точок x
        y_points: масив точок y
        x: точка для обчислення
    
    Returns:
        Інтерпольоване значення
    """
    table = newton_divided_difference(x_points, y_points)
    n = len(x_points)
    result = table[0][0]
    product = 1
    
    for i in range(1, n):
        product *= (x - x_points[i - 1])
        result += table[0][i] * product
    
    return result


def cubic_spline_interpolation(x_points, y_points, x):
    """
    Кубічна сплайн-інтерполяція
    Cubic spline interpolation
    
    Args:
        x_points: масив точок x
        y_points: масив точок y
        x: точка для обчислення
    
    Returns:
        Інтерпольоване значення
    """
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x_points, y_points)
    return cs(x)


def demonstrate_interpolation():
    """Демонстрація методів інтерполяції"""
    
    # Вихідні точки / Original points
    x_points = np.array([0, 1, 2, 3, 4, 5])
    y_points = np.array([0, 0.8414, 0.9093, 0.1411, -0.7568, -0.9589])
    
    # Точки для інтерполяції
    x_interp = np.linspace(0, 5, 100)
    
    # Лагранж
    y_lagrange = [lagrange_interpolation(x_points, y_points, x) for x in x_interp]
    
    # Ньютон
    y_newton = [newton_interpolation(x_points, y_points, x) for x in x_interp]
    
    # Побудова графіків
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_points, y_points, 'ro', label='Вихідні точки', markersize=8)
    plt.plot(x_interp, y_lagrange, 'b-', label='Лагранж')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Інтерполяція методом Лагранжа')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_points, y_points, 'ro', label='Вихідні точки', markersize=8)
    plt.plot(x_interp, y_newton, 'g-', label='Ньютон')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Інтерполяція методом Ньютона')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('interpolation_results.png', dpi=300, bbox_inches='tight')
    print("Графік збережено у файл: interpolation_results.png")
    
    # Виведення результатів
    print("\n=== Результати інтерполяції ===")
    test_x = 2.5
    print(f"\nТестова точка x = {test_x}")
    print(f"Лагранж: y = {lagrange_interpolation(x_points, y_points, test_x):.6f}")
    print(f"Ньютон: y = {newton_interpolation(x_points, y_points, test_x):.6f}")


if __name__ == "__main__":
    demonstrate_interpolation()
