#!/usr/bin/env python3
"""
Скрипт для визуализации решения задачи Пуассона.
Читает файл solution_40x40.txt и строит цветную карту решения.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def read_solution(filename='solution_40x40.txt'):
    """Читает решение из файла."""
    try:
        data = np.loadtxt(filename)
        x = data[:, 0]
        y = data[:, 1]
        u = data[:, 2]
        return x, y, u
    except Exception as e:
        print(f"Ошибка чтения файла {filename}: {e}")
        sys.exit(1)

def plot_2d(x, y, u):
    """Строит 2D визуализацию решения."""
    # Определяем сетку
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    M = len(x_unique)
    N = len(y_unique)
    
    # Создаем матрицу решения
    U = np.zeros((N, M))
    for i in range(len(x)):
        ix = np.where(x_unique == x[i])[0][0]
        iy = np.where(y_unique == y[i])[0][0]
        U[iy, ix] = u[i]
    
    # Рисуем
    plt.figure(figsize=(10, 8))
    
    # Цветная карта
    plt.subplot(1, 2, 1)
    im = plt.contourf(x_unique, y_unique, U, levels=20, cmap='viridis')
    plt.colorbar(im, label='u(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Решение задачи Пуассона (сапожок)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Изолинии
    plt.subplot(1, 2, 2)
    cs = plt.contour(x_unique, y_unique, U, levels=15, colors='black', linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.contourf(x_unique, y_unique, U, levels=15, cmap='coolwarm', alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Изолинии решения')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('solution_2d.png', dpi=150)
    print("2D визуализация сохранена в solution_2d.png")
    plt.show()

def plot_3d(x, y, u):
    """Строит 3D визуализацию решения."""
    # Определяем сетку
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    M = len(x_unique)
    N = len(y_unique)
    
    # Создаем матрицу решения
    U = np.zeros((N, M))
    for i in range(len(x)):
        ix = np.where(x_unique == x[i])[0][0]
        iy = np.where(y_unique == y[i])[0][0]
        U[iy, ix] = u[i]
    
    # Создаем сетку для 3D
    X, Y = np.meshgrid(x_unique, y_unique)
    
    # Рисуем
    fig = plt.figure(figsize=(12, 5))
    
    # 3D поверхность
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, U, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x, y)')
    ax1.set_title('3D визуализация решения')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Вид сверху с проекцией
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, U, cmap='plasma', edgecolor='k', linewidth=0.1, alpha=0.7)
    ax2.contour(X, Y, U, levels=10, offset=np.min(U), cmap='coolwarm', linewidths=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x, y)')
    ax2.set_title('3D + контуры')
    ax2.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('solution_3d.png', dpi=150)
    print("3D визуализация сохранена в solution_3d.png")
    plt.show()

def main():
    """Основная функция."""
    print("Визуализация решения задачи Пуассона")
    print("=" * 50)
    
    # Читаем данные
    x, y, u = read_solution()
    
    print(f"Прочитано точек: {len(x)}")
    print(f"Диапазон x: [{np.min(x):.3f}, {np.max(x):.3f}]")
    print(f"Диапазон y: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"Диапазон u: [{np.min(u):.6e}, {np.max(u):.6e}]")
    print()
    
    # Строим графики
    plot_2d(x, y, u)
    plot_3d(x, y, u)
    
    print("\nГотово!")

if __name__ == '__main__':
    main()
