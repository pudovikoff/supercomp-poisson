#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace std;

// Параметры области (вариант 7 - "сапожок")
// Область: {(x,y): -1 < x,y < 1} \ {(x,y): 0 < x,y < 1}
const double A1 = -1.0, B1 = 1.0;  // границы по x
const double A2 = -1.0, B2 = 1.0;  // границы по y

// Проверка, находится ли точка (x, y) в области D (сапожок)
bool is_in_domain(double x, double y) {
    // Точка в большом квадрате [-1, 1] x [-1, 1]
    if (x <= -1.0 || x >= 1.0 || y <= -1.0 || y >= 1.0) return false;
    // Исключаем правый верхний квадрант [0, 1] x [0, 1]
    if (x >= 0.0 && y >= 0.0) return false;
    return true;
}

// Вычисляем длину отрезка [P1, P2], лежащую в области D
// Для упрощения используем аналитический подход для сапожка
double segment_length_in_D(double x1, double y1, double x2, double y2) {
    bool p1_in = is_in_domain(x1, y1);
    bool p2_in = is_in_domain(x2, y2);
    
    if (p1_in && p2_in) {
        return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }
    if (!p1_in && !p2_in) {
        return 0.0;
    }
    
    // Один конец в области, другой нет
    // Граница вырезанной части - это линия x=0 (для y>=0) или y=0 (для x>=0)
    double len = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    
    // Если отрезок вертикальный (x1 == x2)
    if (fabs(x1 - x2) < 1e-12) {
        if (x1 < 0.0) return len;  // слева от границы - полностью в D
        if (x1 > 0.0) {  // справа - нужно найти пересечение с y=0
            double y_min = min(y1, y2);
            double y_max = max(y1, y2);
            if (y_max <= 0.0) return len;  // ниже границы
            if (y_min >= 0.0) return 0.0;  // выше границы
            return fabs(y_min);  // от y_min до 0
        }
        // x1 == 0 - на границе
        double y_min = min(y1, y2);
        if (y_min < 0.0) return fabs(y_min);
        return 0.0;
    }
    
    // Если отрезок горизонтальный (y1 == y2)
    if (fabs(y1 - y2) < 1e-12) {
        if (y1 < 0.0) return len;  // ниже границы - полностью в D
        if (y1 > 0.0) {  // выше - нужно найти пересечение с x=0
            double x_min = min(x1, x2);
            double x_max = max(x1, x2);
            if (x_max <= 0.0) return len;  // левее границы
            if (x_min >= 0.0) return 0.0;  // правее границы
            return fabs(x_min);  // от x_min до 0
        }
        // y1 == 0 - на границе
        double x_min = min(x1, x2);
        if (x_min < 0.0) return fabs(x_min);
        return 0.0;
    }
    
    // Упрощенная оценка для случая, когда у нас отрезок пересекает границу области
    // один из концов отрезка находится за границей области, а второй нет
    return len * 0.5;
}

// Вычисляем площадь пересечения прямоугольника с областью D
double cell_area_in_D(double x_left, double x_right, double y_bottom, double y_top) {
    double full_area = (x_right - x_left) * (y_top - y_bottom);
    
    // Если прямоугольник полностью в D
    if (x_right <= 0.0 || y_top <= 0.0) return full_area;
    
    // Если прямоугольник полностью в вырезанной части
    if (x_left >= 0.0 && y_bottom >= 0.0) return 0.0;
    
    // Прямоугольник пересекает границу вырезанной области
    // Вырезанная область: x >= 0 AND y >= 0
    double x_int = max(0.0, x_left);  // левая граница пересечения
    double y_int = max(0.0, y_bottom);  // нижняя граница пересечения
    
    double removed_area = max(0.0, x_right - x_int) * max(0.0, y_top - y_int);
    
    return full_area - removed_area;
}

// Класс для хранения сеточных функций и решения задачи
class PoissonSolver {
public:
    int M, N;  // количество интервалов по x и y
    double h1, h2;  // шаги сетки
    double eps;  // параметр метода фиктивных областей
    
    vector<vector<double>> a;  // коэффициенты a[i][j] (M+1 x N+1)
    vector<vector<double>> b;  // коэффициенты b[i][j] (M+1 x N+1)
    vector<vector<double>> F;  // правая часть F[i][j] (M-1 x N-1)
    vector<vector<double>> D_diag;  // диагональный предобуславливатель
    
    // Таймеры для детального анализа
    double time_coefficients_init = 0.0;  // Инициализация коэффициентов
    double time_apply_A = 0.0;            // Применение оператора A
    double time_apply_D_inv = 0.0;        // Применение предобуславливателя D^{-1}
    double time_vector_ops = 0.0;         // Векторные операции
    double time_reductions = 0.0;         // Редукции (скалярные произведения, нормы)
    double time_total = 0.0;              // Общее время работы solve_CG
    
    PoissonSolver(int M_, int N_) : M(M_), N(N_) {
        h1 = (B1 - A1) / M;
        h2 = (B2 - A2) / N;
        eps = max(h1, h2) * max(h1, h2);  // ε = h²
        
        // Выделяем память
        a.assign(M + 1, vector<double>(N + 1, 0.0));
        b.assign(M + 1, vector<double>(N + 1, 0.0));
        F.assign(M - 1, vector<double>(N - 1, 0.0));
        D_diag.assign(M - 1, vector<double>(N - 1, 0.0));
        
        compute_coefficients();
    }
    
    // Вычисление коэффициентов a, b, F
    void compute_coefficients() {
        double t0 = MPI_Wtime();  // Начало измерения
        
        // Коэффициенты a[i][j] для i = 1..M, j = 1..N
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                double x = A1 + (i - 0.5) * h1;
                double y1 = A2 + (j - 0.5) * h2;
                double y2 = y1 + h2;
                
                // Интегрируем k(x, y) по вертикальному отрезку
                double len_total = h2;
                double len_in_D = segment_length_in_D(x, y1, x, y2);
                double len_out_D = len_total - len_in_D;
                
                // a[i][j] = (1/h2) * integral k(x, y) dy
                a[i][j] = (len_in_D * 1.0 + len_out_D / eps) / h2;
            }
        }
        
        // Коэффициенты b[i][j] для i = 1..M, j = 1..N
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                double x1 = A1 + (i - 0.5) * h1;
                double x2 = x1 + h1;
                double y = A2 + (j - 0.5) * h2;
                
                // Интегрируем k(x, y) по горизонтальному отрезку
                double len_total = h1;
                double len_in_D = segment_length_in_D(x1, y, x2, y);
                double len_out_D = len_total - len_in_D;
                
                // b[i][j] = (1/h1) * integral k(x, y) dx
                b[i][j] = (len_in_D * 1.0 + len_out_D / eps) / h1;
            }
        }
        
        // Правая часть F[i][j] для i = 0..M-2, j = 0..N-2 (внутренние узлы i=1..M-1, j=1..N-1)
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                double x = A1 + (i + 1) * h1;
                double y = A2 + (j + 1) * h2;
                
                // Границы ячейки вокруг узла (x, y)
                double x_left = x - 0.5 * h1;
                double x_right = x + 0.5 * h1;
                double y_bottom = y - 0.5 * h2;
                double y_top = y + 0.5 * h2;
                
                double area_in_D = cell_area_in_D(x_left, x_right, y_bottom, y_top);
                
                // F(x, y) = f(x, y) в D, f = 1
                F[i][j] = area_in_D / (h1 * h2);
            }
        }
        
        // Диагональный предобуславливатель D
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                D_diag[i][j] = (a[i + 2][j + 1] + a[i + 1][j + 1]) / (h1 * h1) +
                               (b[i + 1][j + 2] + b[i + 1][j + 1]) / (h2 * h2);
            }
        }
        
        time_coefficients_init += MPI_Wtime() - t0;  // Конец измерения
    }
    
    // Применение оператора A: Aw = result
    void apply_A(const vector<vector<double>>& w, vector<vector<double>>& result) {
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                // Индексы в массиве w: i, j соответствуют узлу (i+1, j+1)
                double w_ij = w[i][j];
                double w_im1j = (i > 0) ? w[i - 1][j] : 0.0;
                double w_ip1j = (i < M - 2) ? w[i + 1][j] : 0.0;
                double w_ijm1 = (j > 0) ? w[i][j - 1] : 0.0;
                double w_ijp1 = (j < N - 2) ? w[i][j + 1] : 0.0;
                
                // A = -(a*w_x)_x - (b*w_y)_y
                double d2w_dx2 = (a[i + 2][j + 1] * (w_ip1j - w_ij) / h1 -
                                  a[i + 1][j + 1] * (w_ij - w_im1j) / h1) / h1;
                double d2w_dy2 = (b[i + 1][j + 2] * (w_ijp1 - w_ij) / h2 -
                                  b[i + 1][j + 1] * (w_ij - w_ijm1) / h2) / h2;
                
                result[i][j] = -(d2w_dx2 + d2w_dy2);
            }
        }
    }
    
    // Применение предобуславливателя D: Dz = r => z = D^(-1)r
    void apply_D_inv(const vector<vector<double>>& r, vector<vector<double>>& z) {
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                z[i][j] = r[i][j] / D_diag[i][j];
            }
        }
    }
    
    // Скалярное произведение
    double dot_product(const vector<vector<double>>& u, const vector<vector<double>>& v) {
        double sum = 0.0;
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                sum += u[i][j] * v[i][j];
            }
        }
        return sum * h1 * h2;
    }
    
    // Евклидова норма
    double euclidean_norm(const vector<vector<double>>& u) {
        return sqrt(dot_product(u, u));
    }
    
    // Максимум норма (C-норма)
    double max_norm(const vector<vector<double>>& u) {
        double max_val = 0.0;
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                max_val = max(max_val, fabs(u[i][j]));
            }
        }
        return max_val;
    }
    
    // Метод сопряженных градиентов с предобуславливанием
    void solve_CG(vector<vector<double>>& w, double delta, int max_iter, int& iter_count, double& solve_time) {
        double time_cg_start = MPI_Wtime();  // Начало общего таймера
        auto start_time = chrono::high_resolution_clock::now();
        
        // Инициализация
        vector<vector<double>> r(M - 1, vector<double>(N - 1, 0.0));
        vector<vector<double>> z(M - 1, vector<double>(N - 1, 0.0));
        vector<vector<double>> p(M - 1, vector<double>(N - 1, 0.0));
        vector<vector<double>> Ap(M - 1, vector<double>(N - 1, 0.0));
        
        // w(0) = 0 (начальное приближение)
        w.assign(M - 1, vector<double>(N - 1, 0.0));
        
        // r(0) = B - A*w(0) = F
        r = F;
        
        // Dz(0) = r(0)
        double t0 = MPI_Wtime();
        apply_D_inv(r, z);
        time_apply_D_inv += MPI_Wtime() - t0;
        
        // p(1) = z(0)
        t0 = MPI_Wtime();
        p = z;
        time_vector_ops += MPI_Wtime() - t0;
        
        t0 = MPI_Wtime();
        double rz_old = dot_product(z, r);
        time_reductions += MPI_Wtime() - t0;
        
        // Для контроля монотонности
        double H_prev = 0.0;
        apply_A(w, Ap);
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                H_prev += (F[i][j] + r[i][j]) * w[i][j];
            }
        }
        H_prev *= h1 * h2;
        
        // Итерационный процесс
        for (int k = 0; k < max_iter; ++k) {
            // Ap = A * p
            t0 = MPI_Wtime();
            apply_A(p, Ap);
            time_apply_A += MPI_Wtime() - t0;
            
            // alpha = (z, r) / (Ap, p)
            t0 = MPI_Wtime();
            double alpha = rz_old / dot_product(Ap, p);
            time_reductions += MPI_Wtime() - t0;
            
            // w(k+1) = w(k) + alpha * p
            t0 = MPI_Wtime();
            vector<vector<double>> w_new = w;
            for (int i = 0; i < M - 1; ++i) {
                for (int j = 0; j < N - 1; ++j) {
                    w_new[i][j] = w[i][j] + alpha * p[i][j];
                }
            }
            time_vector_ops += MPI_Wtime() - t0;
            
            // Проверка условия остановки
            t0 = MPI_Wtime();
            double diff_norm = 0.0;
            for (int i = 0; i < M - 1; ++i) {
                for (int j = 0; j < N - 1; ++j) {
                    double diff = w_new[i][j] - w[i][j];
                    diff_norm += diff * diff;
                }
            }
            diff_norm = sqrt(diff_norm * h1 * h2);
            time_reductions += MPI_Wtime() - t0;
            
            w = w_new;
            
            if (diff_norm < delta) {
                iter_count = k + 1;
                time_total += MPI_Wtime() - time_cg_start;  // Конец общего таймера
                auto end_time = chrono::high_resolution_clock::now();
                solve_time = chrono::duration<double>(end_time - start_time).count();
                return;
            }
            
            // r(k+1) = r(k) - alpha * Ap
            t0 = MPI_Wtime();
            for (int i = 0; i < M - 1; ++i) {
                for (int j = 0; j < N - 1; ++j) {
                    r[i][j] = r[i][j] - alpha * Ap[i][j];
                }
            }
            time_vector_ops += MPI_Wtime() - t0;
            
            // Dz(k+1) = r(k+1)
            t0 = MPI_Wtime();
            apply_D_inv(r, z);
            time_apply_D_inv += MPI_Wtime() - t0;
            
            t0 = MPI_Wtime();
            double rz_new = dot_product(z, r);
            time_reductions += MPI_Wtime() - t0;
            
            // Контроль монотонности H(w)
            double H_curr = 0.0;
            apply_A(w, Ap);
            for (int i = 0; i < M - 1; ++i) {
                for (int j = 0; j < N - 1; ++j) {
                    H_curr += (F[i][j] + r[i][j]) * w[i][j];
                }
            }
            H_curr *= h1 * h2;
            
            if (H_curr < H_prev) {
                // Монотонность нарушена - перезапуск
                cout << "  Перезапуск на итерации " << k + 1 << endl;
                r = F;
                apply_A(w, Ap);
                for (int i = 0; i < M - 1; ++i) {
                    for (int j = 0; j < N - 1; ++j) {
                        r[i][j] = F[i][j] - Ap[i][j];
                    }
                }
                apply_D_inv(r, z);
                p = z;
                rz_old = dot_product(z, r);
                H_prev = H_curr;
                continue;
            }
            
            H_prev = H_curr;
            
            // beta = (z(k+1), r(k+1)) / (z(k), r(k))
            double beta = rz_new / rz_old;
            
            // p(k+1) = z(k+1) + beta * p(k)
            t0 = MPI_Wtime();
            for (int i = 0; i < M - 1; ++i) {
                for (int j = 0; j < N - 1; ++j) {
                    p[i][j] = z[i][j] + beta * p[i][j];
                }
            }
            time_vector_ops += MPI_Wtime() - t0;
            
            rz_old = rz_new;
        }
        
        time_total += MPI_Wtime() - time_cg_start;  // Конец общего таймера
        iter_count = max_iter;
        auto end_time = chrono::high_resolution_clock::now();
        solve_time = chrono::duration<double>(end_time - start_time).count();
    }
    
    // Сохранение решения в файл
    void save_solution(const vector<vector<double>>& w, const string& filename) {
        ofstream file(filename);
        file << scientific << setprecision(10);
        
        for (int j = N - 2; j >= 0; --j) {
            for (int i = 0; i < M - 1; ++i) {
                double x = A1 + (i + 1) * h1;
                double y = A2 + (j + 1) * h2;
                file << x << " " << y << " " << w[i][j] << "\n";
            }
        }
        
        file.close();
    }
};

int main(int argc, char* argv[]) {
    // Параметры расчетов
    vector<pair<int, int>> grid_sizes = {{10, 10}, {20, 20}, {40, 40}};
    double delta = 1e-6;  // точность метода
    int max_iter = 100000;
    
    cout << "=== Решение задачи Пуассона (вариант 7 - сапожок) ===" << endl;
    cout << "Область: {(x,y): -1<x,y<1} \\ {(x,y): 0<x,y<1}" << endl << endl;
    
    for (const auto& grid : grid_sizes) {
        int M = grid.first;
        int N = grid.second;
        
        cout << "Сетка: " << M << " x " << N << endl;
        
        PoissonSolver solver(M, N);
        vector<vector<double>> w;
        
        int iter_count;
        double solve_time;
        
        solver.solve_CG(w, delta, max_iter, iter_count, solve_time);
        
        double norm_E = solver.euclidean_norm(w);
        double norm_C = solver.max_norm(w);
        
        cout << "  Итераций: " << iter_count << endl;
        cout << "  Время решения: " << fixed << setprecision(6) << solve_time << " с" << endl;
        cout << "  Норма ||w||_E: " << scientific << norm_E << endl;
        cout << "  Норма ||w||_C: " << scientific << norm_C << endl;
        
        // Вывод детального тайминга
        cout << "\n=== Timing Breakdown ===" << endl;
        cout << "Coefficients init:    " << fixed << setprecision(6) << solver.time_coefficients_init << " s" << endl;
        cout << "apply_A:              " << fixed << setprecision(6) << solver.time_apply_A << " s" << endl;
        cout << "apply_D_inv:          " << fixed << setprecision(6) << solver.time_apply_D_inv << " s" << endl;
        cout << "Vector operations:    " << fixed << setprecision(6) << solver.time_vector_ops << " s" << endl;
        cout << "Local reductions:     " << fixed << setprecision(6) << solver.time_reductions << " s" << endl;
        cout << "---" << endl;
        cout << "Total time (solve):   " << fixed << setprecision(6) << solver.time_total << " s" << endl << endl;
        
        // Сохраняем решение для самой мелкой сетки
        if (M == 40) {
            solver.save_solution(w, "results/solution_40x40.txt");
            cout << "  Решение сохранено в results/solution_40x40.txt" << endl << endl;
        }
    }
    
    return 0;
}
