#pragma once

#include <algorithm>
#include <cmath>

// Вспомогательные функции для двумерного разбиения прямоугольной сетки
// на Px × Py доменов.
//
// Глобальная сетка задаётся числом интервалов M и N.
// Внутренние узлы (не лежащие на границе прямоугольника) имеют индексы
//   i = 1..M-1 по x,
//   j = 1..N-1 по y.
//
// Требования к разбиению (из условия задачи):
//   1) Отношение количества узлов по x и y в каждом домене ∈ [0.5, 2].
//   2) Количество узлов по каждой координате в любых двух доменах
//      отличается не более чем на единицу.
//
// Эти функции реализуют простой блочный алгоритм, удовлетворяющий этим
// условиям. Он используется как в MPI-коде, так и в тестах.

// Выбор разбиения числа процессов P на Px × Py так, чтобы домены были
// как можно более "квадратными" (ближе к одинаковому числу узлов по x и y).
inline void choose_process_grid(int M, int N, int P, int &Px, int &Py) {
    // По умолчанию все процессы в одном столбце
    Px = 1;
    Py = P;

    // Общие числа внутренних узлов
    const int Nx_total = std::max(1, M - 1);
    const int Ny_total = std::max(1, N - 1);

    double best_cost = 1e100;

    for (int px = 1; px <= P; ++px) {
        if (P % px != 0) continue;
        int py = P / px;

        // Оценка формы домена: отношение усреднённых размеров по осям
        double nx = static_cast<double>(Nx_total) / px;
        double ny = static_cast<double>(Ny_total) / py;
        if (ny == 0.0) continue;
        double ratio = nx / ny;
        if (ratio <= 0.0) continue;

        double cost = std::max(ratio, 1.0 / ratio); // ближе к 1 — лучше
        if (cost < best_cost) {
            best_cost = cost;
            Px = px;
            Py = py;
        }
    }
}

// Вычисление локального поддомена для процесса с координатами (px, py)
// в декартовой решётке Px × Py.
//
// На выходе:
//   ix0, ix1 — глобальные индексы внутренних узлов по x (1..M-1),
//   iy0, iy1 — глобальные индексы внутренних узлов по y (1..N-1),
//   nx, ny   — число внутренних узлов по x и y в данном домене.
inline void get_local_subdomain_nodes(int M, int N,
                                      int Px, int Py,
                                      int px, int py,
                                      int &ix0, int &ix1,
                                      int &iy0, int &iy1,
                                      int &nx,  int &ny) {
    const int Nx_total = std::max(1, M - 1);
    const int Ny_total = std::max(1, N - 1);

    // Базовое количество узлов и остаток по каждой координате
    const int base_nx = Nx_total / Px;
    const int rem_x  = Nx_total % Px;

    const int base_ny = Ny_total / Py;
    const int rem_y  = Ny_total % Py;

    // Число узлов в данном блоке
    nx = base_nx + (px < rem_x ? 1 : 0);
    ny = base_ny + (py < rem_y ? 1 : 0);

    // Смещение начала блока по каждой координате (в терминах внутренних узлов)
    const int offset_x = base_nx * px + std::min(px, rem_x);
    const int offset_y = base_ny * py + std::min(py, rem_y);

    ix0 = 1 + offset_x;
    ix1 = ix0 + nx - 1;

    iy0 = 1 + offset_y;
    iy1 = iy0 + ny - 1;
}
