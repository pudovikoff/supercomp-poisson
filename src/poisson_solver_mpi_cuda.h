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
    
    // Флаг для оптимизации (один ГПУ)
    bool is_single_gpu;
    
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
    
    // Буферы для граничных полос (оптимизация копирования)
    double *boundary_left_dev, *boundary_right_dev;  // размер ny каждый
    double *boundary_down_dev, *boundary_up_dev;     // размер nx каждый
    
    // Буферы для GPU-редукций
    double *reduction_buffer_dev;  // Для промежуточных сумм
    double *reduction_buffer_host; // Буфер на CPU для резултатов редукции
    int num_reduction_blocks;
    int reduction_threads_per_block;
    
    // Флаг сходимости на GPU (для одного ГПУ)
    bool *converged_dev;
    bool converged_host;
    
    // Device-скаляры для single-GPU оптимизации (1 двойная точка или переиспользуем reduction_buffer_dev)
    double *alpha_dev;      // коэффициент alpha
    double *beta_dev;       // коэффициент beta
    double *rz_prev_dev;    // предыдущее (z,r) для beta
    
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
    double time_cg_preprocessing; // инициализация CG (до цикла)
    double time_cg_loop;          // главный цикл CG
    double time_gpu_reductions;   // редукции на GPU (dot product, reduce)
    double time_gpu_overhead;     // kernel launches, синхронизации, D2D копирования
    
    // GPU устройство
    int num_devices;
    int device_id;
    
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
    void exchange_gpu(vector<double>& boundary_left_host,
                     vector<double>& boundary_right_host,
                     vector<double>& boundary_down_host,
                     vector<double>& boundary_up_host);
    
    // CPU редукции
    double dot_product_cpu(const double* vec1, const double* vec2, int n);
    double max_norm_cpu(const double* vec, int n);
    
    // GPU редукция (2-ступенчатая: GPU partial + CPU final)
    double dot_product_gpu(const double* vec1_dev, const double* vec2_dev, int n);
    
    // GPU редукция без копирования на CPU - ретурнит GPU пойнтер
    double* dot_product_gpu_ptr(const double* vec1_dev, const double* vec2_dev, int n);
    
    // Копирование дного элемента с GPU для редукции и проверки
    double copy_result_from_gpu(const double* result_dev);
};

// Объявления CUDA ядер (определены в .cu файле)
void launch_apply_A_kernel(const double* p_dev, double* Ap_dev,
                          const double* a_x_dev, const double* b_y_dev,
                          int nx, int ny, double h1, double h2,
                          cudaStream_t stream);

void launch_apply_D_inv_kernel(const double* r_dev, double* z_dev,
                               const double* Ddiag_dev,
                               int nx, int ny, cudaStream_t stream);
// y += a * x
void launch_axpy_kernel(double* y_dev, const double* x_dev,
                       double alpha, int n, cudaStream_t stream);

// update after A
void launch_vector_update_kernel(double* p_dev, const double* z_dev,
                                double beta, int n, cudaStream_t stream);

void launch_copy_interior_to_device(double* w_dev, const double* w_host,
                                   int nx, int ny, cudaStream_t stream);

void launch_copy_interior_from_device(double* w_host, const double* w_dev,
                                     int nx, int ny, cudaStream_t stream);

void launch_dot_product_partial(const double* vec1_dev, const double* vec2_dev,
                               double* block_results_dev, int n,
                               int num_blocks, int threads_per_block,
                               cudaStream_t stream);

void launch_reduce_blocks(const double* in_dev, double* out_dev,
                         int num_elems, cudaStream_t stream);

void launch_update_w_and_compute_diff(double* w_interior_dev, const double* p_dev,
                                     double alpha, double* thread_diffs_dev,
                                     int n_interior, int num_blocks, int threads_per_block,
                                     cudaStream_t stream);

void launch_extract_boundaries(const double* w_dev,
                              double* boundary_left_dev, double* boundary_right_dev,
                              double* boundary_down_dev, double* boundary_up_dev,
                              int nx, int ny, cudaStream_t stream);

void launch_inject_boundaries(double* w_dev,
                             const double* boundary_left_dev, const double* boundary_right_dev,
                             const double* boundary_down_dev, const double* boundary_up_dev,
                             int nx, int ny, cudaStream_t stream);

void launch_check_convergence(const double* diff_sum_dev, bool* converged_dev,
                             double delta, double h1, double h2, cudaStream_t stream);

// Device-scalar kernels for single-GPU path
void launch_compute_alpha(const double* rz_dev, const double* denom_dev, double* alpha_dev,
                        cudaStream_t stream);
void launch_compute_beta(const double* rz_new_dev, const double* rz_prev_dev, double* beta_dev,
                       cudaStream_t stream);

void launch_axpy_dev_scalar(double* y_dev, const double* x_dev, const double* alpha_dev,
                           double scale, int n, cudaStream_t stream);

void launch_vector_update_dev_scalar(double* p_dev, const double* z_dev, const double* beta_dev,
                                    int n, cudaStream_t stream);

void launch_update_w_and_compute_diff_dev_scalar(double* w_interior_dev, const double* p_dev,
                                                const double* alpha_dev, double* thread_diffs_dev,
                                                int n_interior, int num_blocks, int threads_per_block,
                                                cudaStream_t stream);
