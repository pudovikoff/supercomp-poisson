#pragma once
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include "domain_decomposition.h"

using namespace std;

bool is_in_domain(double x, double y) {
    if (x <= -1.0 || x >= 1.0 || y <= -1.0 || y >= 1.0) return false;
    if (x >= 0.0 && y >= 0.0) return false; // вырезанный квадрант
    return true;
}
    
double segment_length_in_D(double x1, double y1, double x2, double y2) {
    bool p1_in = is_in_domain(x1, y1);
    bool p2_in = is_in_domain(x2, y2);
    if (p1_in && p2_in) return hypot(x2 - x1, y2 - y1);
    if (!p1_in && !p2_in) return 0.0;
    double len = hypot(x2 - x1, y2 - y1);
    if (fabs(x1 - x2) < 1e-12) {
        if (x1 < 0.0) return len;
        if (x1 > 0.0) {
            double y_min = min(y1, y2), y_max = max(y1, y2);
            if (y_max <= 0.0) return len;
            if (y_min >= 0.0) return 0.0;
            return fabs(y_min);
        }
        double y_min = min(y1, y2);
        return y_min < 0.0 ? fabs(y_min) : 0.0;
    }
    if (fabs(y1 - y2) < 1e-12) {
        if (y1 < 0.0) return len;
        if (y1 > 0.0) {
            double x_min = min(x1, x2), x_max = max(x1, x2);
            if (x_max <= 0.0) return len;
            if (x_min >= 0.0) return 0.0;
            return fabs(x_min);
        }
        double x_min = min(x1, x2);
        return x_min < 0.0 ? fabs(x_min) : 0.0;
    }
    return 0.5 * len; // общее приближение
}
    
double cell_area_in_D(double x_left, double x_right, double y_bottom, double y_top) {
    double full_area = (x_right - x_left) * (y_top - y_bottom);
    if (x_right <= 0.0 || y_top <= 0.0) return full_area;
    if (x_left >= 0.0 && y_bottom >= 0.0) return 0.0;
    double x_int = max(0.0, x_left);
    double y_int = max(0.0, y_bottom);
    double removed = max(0.0, x_right - x_int) * max(0.0, y_top - y_int);
    return full_area - removed;
}

// Обёртка над 2D массивом с призрачными слоями [0..nx+1][0..ny+1]
struct Grid2D {
    int nx, ny; // внутренние узлы
    vector<double> data; // (nx+2)*(ny+2)
    
    Grid2D() : nx(0), ny(0) {}
    Grid2D(int nx_, int ny_) { reset(nx_, ny_); }
    
    void reset(int nx_, int ny_) { 
        nx = nx_; 
        ny = ny_; 
        data.assign((nx+2)*(ny+2), 0.0); 
    }
    
    inline double& at(int i, int j) { 
        return data[(i)*(ny+2) + j]; 
    }
    
    inline const double& at(int i, int j) const { 
        return data[(i)*(ny+2) + j]; 
    } // i \in [0..nx+1], j \in [0..ny+1]
};

class PoissonSolverMPI {
public:
    // Глобальная сетка
    int M, N;
    double h1, h2, eps;
    
    // Локальный поддомен
    int nx, ny;           // число внутренних узлов в этом процессе
    int ix0, ix1, iy0, iy1; // глобальные индексы
    
    // MPI параметры
    MPI_Comm cart_comm;
    int world_rank, world_size;
    int Px, Py;           // размеры сетки процессов
    int px, py;           // координаты этого процесса в сетке
    int nbr_left, nbr_right, nbr_down, nbr_up; // соседи
    
    // Коэффициенты на рёбрах
    vector<double> a_face_x; // (nx+1) × ny
    vector<double> b_face_y; // nx × (ny+1)
    
    // Правая часть и диагональ
    vector<double> F;
    vector<double> Ddiag;
    
    // Встроенные функции доступа к коэффициентам
    inline double& afx(int il, int jl) {
        return a_face_x[il*ny + jl]; // il∈[0..nx], jl∈[0..ny-1]
    }
    
    inline double& bfy(int il, int jl) {
        return b_face_y[il*(ny+1) + jl]; // il∈[0..nx-1], jl∈[0..ny]
    }
    
    inline double& fij(int il, int jl) {
        return F[(il-1)*ny + (jl-1)]; // il∈[1..nx], jl∈[1..ny]
    }
    
    inline double& dij(int il, int jl) {
        return Ddiag[(il-1)*ny + (jl-1)];
    }
    
    // Геометрия области (вариант 7 — «сапожок»)
    static constexpr double A1 = -1.0, B1 = 1.0;
    static constexpr double A2 = -1.0, B2 = 1.0;
    
    // Конструктор
    PoissonSolverMPI(int M_, int N_, MPI_Comm comm_)
        : M(M_), N(N_), cart_comm(comm_) {
        
        MPI_Comm_rank(cart_comm, &world_rank);
        MPI_Comm_size(cart_comm, &world_size);
        
        // Получить топологию
        int dims[2], periods[2], coords[2];
        MPI_Cart_get(cart_comm, 2, dims, periods, coords);
        Px = dims[0];
        Py = dims[1];
        px = coords[0];
        py = coords[1];
        
        h1 = (B1 - A1) / M;
        h2 = (B2 - A2) / N;
        eps = max(h1, h2) * max(h1, h2);
        
        // Получить локальный поддомен
        get_local_subdomain_nodes(M, N, Px, Py, px, py, ix0, ix1, iy0, iy1, nx, ny);
        
        // Получить соседей
        MPI_Cart_shift(cart_comm, 0, +1, &nbr_left, &nbr_right); // x
        MPI_Cart_shift(cart_comm, 1, +1, &nbr_down, &nbr_up);    // y
        
        // Инициализировать коэффициенты
        a_face_x.assign((nx+1)*ny, 0.0);
        b_face_y.assign(nx*(ny+1), 0.0);
        F.assign(nx*ny, 0.0);
        Ddiag.assign(nx*ny, 0.0);
        
        compute_coefficients();
    }
    
private:
    void compute_coefficients() {
        // Заполнение a_face_x
        for (int il = 0; il <= nx; ++il) {
            int ig = ix0 + il - 1; // глобальный индекс
            double x = A1 + (ig - 0.5) * h1;
            for (int jl = 1; jl <= ny; ++jl) {
                int jg = iy0 + jl - 1;
                double y1 = A2 + (jg - 0.5) * h2;
                double y2 = y1 + h2;
                double lin = segment_length_in_D(x, y1, x, y2);
                double lout = h2 - lin;
                afx(il, jl - 1) = (lin + lout / eps) / h2;
            }
        }
        
        // Заполнение b_face_y
        for (int il = 1; il <= nx; ++il) {
            int ig = ix0 + il - 1;
            for (int jl = 0; jl <= ny; ++jl) {
                int jg = iy0 + jl - 1;
                double y = A2 + (jg - 0.5) * h2;
                double x1 = A1 + (ig - 0.5) * h1;
                double x2 = x1 + h1;
                double lin = segment_length_in_D(x1, y, x2, y);
                double lout = h1 - lin;
                bfy(il - 1, jl) = (lin + lout / eps) / h1;
            }
        }
        
        // Правая часть F и диагональ D_diag
        for (int il = 1; il <= nx; ++il) {
            int ig = ix0 + il - 1;
            double x = A1 + ig * h1;
            double xl = x - 0.5*h1, xr = x + 0.5*h1;
            for (int jl = 1; jl <= ny; ++jl) {
                int jg = iy0 + jl - 1;
                double y = A2 + jg * h2;
                double yb = y - 0.5*h2, yt = y + 0.5*h2;
                double area = cell_area_in_D(xl, xr, yb, yt);
                fij(il, jl) = area / (h1*h2); // f=1 в D
                
                double aL = afx(il - 1, jl - 1), aR = afx(il, jl - 1);
                double bD = bfy(il - 1, jl - 1), bU = bfy(il - 1, jl);
                dij(il, jl) = (aL + aR) / (h1*h1) + (bD + bU) / (h2*h2);
            }
        }
    }
    
public:
    void exchange(Grid2D& U) {
        MPI_Status st;
        
        // Вдоль Y (нижняя/верхняя границы) — целые строки длиной nx
        vector<double> send_bottom(nx), recv_bottom(nx), send_top(nx), recv_top(nx);
        for (int il = 1; il <= nx; ++il) {
            send_bottom[il - 1] = U.at(il, 1);
            send_top[il - 1] = U.at(il, ny);
        }
        
        // Обмен вдоль y
        MPI_Sendrecv(send_bottom.data(), nx, MPI_DOUBLE, nbr_down, 100,
                     recv_top.data(), nx, MPI_DOUBLE, nbr_up, 100,
                     cart_comm, &st);
        MPI_Sendrecv(send_top.data(), nx, MPI_DOUBLE, nbr_up, 101,
                     recv_bottom.data(), nx, MPI_DOUBLE, nbr_down, 101,
                     cart_comm, &st);
        
        if (nbr_up != MPI_PROC_NULL)
            for (int il = 1; il <= nx; ++il) U.at(il, ny + 1) = recv_top[il - 1];
        else
            for (int il = 1; il <= nx; ++il) U.at(il, ny + 1) = 0.0;
            
        if (nbr_down != MPI_PROC_NULL)
            for (int il = 1; il <= nx; ++il) U.at(il, 0) = recv_bottom[il - 1];
        else
            for (int il = 1; il <= nx; ++il) U.at(il, 0) = 0.0;
        
        // Вдоль X (левая/правая границы) — столбцы длиной ny
        vector<double> send_left(ny), recv_left(ny), send_right(ny), recv_right(ny);
        for (int jl = 1; jl <= ny; ++jl) {
            send_left[jl - 1] = U.at(1, jl);
            send_right[jl - 1] = U.at(nx, jl);
        }
        
        MPI_Sendrecv(send_left.data(), ny, MPI_DOUBLE, nbr_left, 102,
                     recv_right.data(), ny, MPI_DOUBLE, nbr_right, 102,
                     cart_comm, &st);
        MPI_Sendrecv(send_right.data(), ny, MPI_DOUBLE, nbr_right, 103,
                     recv_left.data(), ny, MPI_DOUBLE, nbr_left, 103,
                     cart_comm, &st);
        
        if (nbr_right != MPI_PROC_NULL)
            for (int jl = 1; jl <= ny; ++jl) U.at(nx + 1, jl) = recv_right[jl - 1];
        else
            for (int jl = 1; jl <= ny; ++jl) U.at(nx + 1, jl) = 0.0;
            
        if (nbr_left != MPI_PROC_NULL)
            for (int jl = 1; jl <= ny; ++jl) U.at(0, jl) = recv_left[jl - 1];
        else
            for (int jl = 1; jl <= ny; ++jl) U.at(0, jl) = 0.0;
    }
    
    void apply_A(const Grid2D& U, Grid2D& RES) {
        for (int il = 1; il <= nx; ++il) {
            for (int jl = 1; jl <= ny; ++jl) {
                double aL = afx(il - 1, jl - 1), aR = afx(il, jl - 1);
                double bD = bfy(il - 1, jl - 1), bU = bfy(il - 1, jl);
                double d2x = (aR * (U.at(il + 1, jl) - U.at(il, jl)) / h1 - 
                             aL * (U.at(il, jl) - U.at(il - 1, jl)) / h1) / h1;
                double d2y = (bU * (U.at(il, jl + 1) - U.at(il, jl)) / h2 - 
                             bD * (U.at(il, jl) - U.at(il, jl - 1)) / h2) / h2;
                RES.at(il, jl) = -(d2x + d2y);
            }
        }
    }
    
    void apply_D_inv(const Grid2D& R, Grid2D& Z) {
        for (int il = 1; il <= nx; ++il)
            for (int jl = 1; jl <= ny; ++jl)
                Z.at(il, jl) = R.at(il, jl) / dij(il, jl);
    }
    
    double dot_product_local(const Grid2D& U, const Grid2D& V) {
        double s = 0.0;
        for (int il = 1; il <= nx; ++il)
            for (int jl = 1; jl <= ny; ++jl)
                s += U.at(il, jl) * V.at(il, jl);
        return s * h1 * h2;
    }
    
    double dot_product_global(const Grid2D& U, const Grid2D& V) {
        double local = dot_product_local(U, V);
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        return global;
    }
    
    double max_norm_local(const Grid2D& U) {
        double m = 0.0;
        for (int il = 1; il <= nx; ++il)
            for (int jl = 1; jl <= ny; ++jl)
                m = std::max(m, fabs(U.at(il, jl)));
        return m;
    }
    
    double max_norm_global(const Grid2D& U) {
        double local = max_norm_local(U);
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
        return global;
    }
    
    double euclidean_norm_local(const Grid2D& U) {
        return sqrt(dot_product_local(U, U));
    }
    
    double euclidean_norm_global(const Grid2D& U) {
        return sqrt(dot_product_global(U, U));
    }
    
    void solve_CG(Grid2D& w, double delta, int max_iter, int& iters, double& tsec) {
        double t0 = MPI_Wtime();
        
        Grid2D r(nx, ny), z(nx, ny), p(nx, ny), Ap(nx, ny);
        
        // Инициализация
        for (int il = 1; il <= nx; ++il) {
            for (int jl = 1; jl <= ny; ++jl) {
                w.at(il, jl) = 0.0;
                r.at(il, jl) = fij(il, jl); // r(0) = F
            }
        }
        
        apply_D_inv(r, z); // z(0)
        for (int il = 1; il <= nx; ++il)
            for (int jl = 1; jl <= ny; ++jl)
                p.at(il, jl) = z.at(il, jl); // p(1)
        
        double rz_global = dot_product_global(z, r);
        
        // Мониторинг H(w)
        double H_prev = 0.0;
        
        for (int k = 0; k < max_iter; ++k) {
            // Ap = A*p (нужны актуальные значения для p)
            exchange(p);
            apply_A(p, Ap);
            
            double denom_local = dot_product_local(Ap, p), denom = 0.0;
            MPI_Allreduce(&denom_local, &denom, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
            double alpha = rz_global / denom;
            
            // w = w + alpha * p; также считаем норму шага
            double diff_sq_local = 0.0;
            for (int il = 1; il <= nx; ++il) {
                for (int jl = 1; jl <= ny; ++jl) {
                    double inc = alpha * p.at(il, jl);
                    w.at(il, jl) += inc;
                    diff_sq_local += inc * inc;
                }
            }
            double diff_sq = 0.0;
            MPI_Allreduce(&diff_sq_local, &diff_sq, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
            double diff_norm = sqrt(diff_sq * h1 * h2);
            if (diff_norm < delta) {
                iters = k + 1;
                tsec = MPI_Wtime() - t0;
                return;
            }
            
            // r = r - alpha * Ap
            for (int il = 1; il <= nx; ++il)
                for (int jl = 1; jl <= ny; ++jl)
                    r.at(il, jl) -= alpha * Ap.at(il, jl);
            
            // z = D^{-1} r
            apply_D_inv(r, z);
            
            double rz_new_local = dot_product_local(z, r), rz_new = 0.0;
            MPI_Allreduce(&rz_new_local, &rz_new, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
            
            // Контроль монотонности H(w)
            double H_curr_local = 0.0;
            for (int il = 1; il <= nx; ++il)
                for (int jl = 1; jl <= ny; ++jl)
                    H_curr_local += (fij(il, jl) + r.at(il, jl)) * w.at(il, jl);
            double H_curr = 0.0;
            MPI_Allreduce(&H_curr_local, &H_curr, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
            H_curr *= h1 * h2;
            
            if (k == 0) H_prev = H_curr;
            if (H_curr < H_prev) {
                // Перезапуск
                for (int il = 1; il <= nx; ++il) {
                    for (int jl = 1; jl <= ny; ++jl) {
                        r.at(il, jl) = fij(il, jl);
                        w.at(il, jl) = 0.0;
                    }
                }
                apply_D_inv(r, z);
                for (int il = 1; il <= nx; ++il)
                    for (int jl = 1; jl <= ny; ++jl)
                        p.at(il, jl) = z.at(il, jl);
                rz_global = dot_product_global(z, r);
                H_prev = H_curr;
                continue;
            }
            H_prev = H_curr;
            
            double beta = rz_new / rz_global;
            for (int il = 1; il <= nx; ++il)
                for (int jl = 1; jl <= ny; ++jl)
                    p.at(il, jl) = z.at(il, jl) + beta * p.at(il, jl);
            
            rz_global = rz_new;
            iters = k + 1;
        }
        tsec = MPI_Wtime() - t0;
    }
};
