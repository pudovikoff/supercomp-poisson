#pragma once
#include <cuda_runtime.h>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include "domain_decomposition.h"

using namespace std;

// Те же функции геометрии области
bool is_in_domain(double x, double y) {
    if (x <= -1.0 || x >= 1.0 || y <= -1.0 || y >= 1.0) return false;
    if (x >= 0.0 && y >= 0.0) return false;
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
    return 0.5 * len;
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

struct Grid2D {
    int nx, ny;
    vector<double> data;
    
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
    }
};

class PoissonSolverMPICUDA {
public:
    // Глобальная сетка
    int M, N;
    double h1, h2, eps;
    
    // Локальный поддомен
    int nx, ny;
    int ix0, ix1, iy0, iy1;
    
    // MPI параметры
    MPI_Comm cart_comm;
    int world_rank, world_size;
    int Px, Py;
    int px, py;
    int nbr_left, nbr_right, nbr_down, nbr_up;
    
    // Коэффициенты на хосте
    vector<double> a_face_x;
    vector<double> b_face_y;
    vector<double> F;
    vector<double> Ddiag;
    
    // GPU указатели
    double *w_dev;           // (nx+2)*(ny+2) - с граничными слоями
    double *w_interior_dev;  // nx*ny - только внутренние узлы w (для CG)
    double *r_dev, *p_dev, *Ap_dev, *z_dev;  // nx*ny - только внутренние узлы
    double *a_face_x_dev, *b_face_y_dev, *Ddiag_dev, *F_dev;
    
    // Буферы для GPU-редукций
    double *reduction_buffer_dev;  // Для промежуточных сумм
    double *reduction_buffer_host; // Pinned memory для быстрого копирования
    int num_reduction_blocks;
    int reduction_threads_per_block;
    
    // Таймеры для отчёта
    double time_init_gpu;
    double time_apply_A;
    double time_apply_D_inv;
    double time_vector_ops;
    double time_gpu_to_cpu;
    double time_cpu_to_gpu;
    double time_mpi_exchange;
    double time_mpi_allreduce;
    double time_cpu_reductions;
    
    // CUDA events для измерения
    cudaEvent_t event_start, event_stop;
    
    inline double& afx(int il, int jl) {
        return a_face_x[il*ny + jl];
    }
    
    inline double& bfy(int il, int jl) {
        return b_face_y[il*(ny+1) + jl];
    }
    
    inline double& fij(int il, int jl) {
        return F[(il-1)*ny + (jl-1)];
    }
    
    inline double& dij(int il, int jl) {
        return Ddiag[(il-1)*ny + (jl-1)];
    }
    
    // Геометрия области (вариант 7 — «сапожок»)
    static const double A1, B1, A2, B2;
    
    PoissonSolverMPICUDA(int M_, int N_, MPI_Comm comm_);
    ~PoissonSolverMPICUDA();
    
    void solve_CG_GPU(Grid2D& w, double delta, int max_iter, int& iters, double& tsec);
    
    double euclidean_norm_global(const Grid2D& w);
    double max_norm_global(const Grid2D& w);
    
private:
    void compute_coefficients();
    void allocate_device_memory();
    void copy_coefficients_to_device();
    void exchange_gpu(Grid2D& U);
    
    // CPU редукции
    double dot_product_cpu(const double* vec1, const double* vec2, int n);
    double max_norm_cpu(const double* vec, int n);
    
    // GPU редукция (2-ступенчатая: GPU partial + CPU final)
    double dot_product_gpu(const double* vec1_dev, const double* vec2_dev, int n);
};

// Объявления CUDA ядер (определены в .cu файле)
void launch_apply_A_kernel(const double* p_dev, double* Ap_dev,
                          const double* a_x_dev, const double* b_y_dev,
                          int nx, int ny, double h1, double h2,
                          cudaStream_t stream);

void launch_apply_D_inv_kernel(const double* r_dev, double* z_dev,
                               const double* Ddiag_dev,
                               int nx, int ny, cudaStream_t stream);

void launch_axpy_kernel(double* y_dev, const double* x_dev,
                       double alpha, int n, cudaStream_t stream);

void launch_vector_update_kernel(double* p_dev, const double* z_dev,
                                double beta, int n, cudaStream_t stream);

void launch_copy_interior_to_device(double* w_dev, const double* w_host,
                                   int nx, int ny, cudaStream_t stream);

void launch_copy_interior_from_device(double* w_host, const double* w_dev,
                                     int nx, int ny, cudaStream_t stream);

void launch_dot_product_partial(const double* vec1_dev, const double* vec2_dev,
                               double* block_sums_dev, int n,
                               int num_blocks, int threads_per_block,
                               cudaStream_t stream);
