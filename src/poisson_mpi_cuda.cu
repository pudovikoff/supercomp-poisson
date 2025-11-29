#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "poisson_solver_mpi_cuda.h"

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// ========== CUDA ядра (kernels) ==========

// Применение оператора A: 5-точечная формула
__global__ void apply_A_kernel(const double* p, double* Ap,
                               const double* a_x, const double* b_y,
                               int nx, int ny, double h1, double h2) {
    int il = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jl = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (il <= nx && jl <= ny) {
        int idx = (il-1) * ny + (jl-1); // индекс в плоском массиве внутренних узлов
        int idx_full = il * (ny+2) + jl; // индекс в массиве с граничными слоями
        
        // Получаем коэффициенты
        double aL = a_x[(il-1)*ny + (jl-1)];
        double aR = a_x[il*ny + (jl-1)];
        double bD = b_y[(il-1)*(ny+1) + (jl-1)];
        double bU = b_y[(il-1)*(ny+1) + jl];
        
        // Получаем значения p из массива с граничными слоями
        double p_ij = p[idx_full];
        double p_left = p[idx_full - (ny+2)];    // p[il-1, jl]
        double p_right = p[idx_full + (ny+2)];   // p[il+1, jl]
        double p_down = p[idx_full - 1];         // p[il, jl-1]
        double p_up = p[idx_full + 1];           // p[il, jl+1]
        
        // Вычисляем вторые производные
        double d2x = (aR * (p_right - p_ij) / h1 - aL * (p_ij - p_left) / h1) / h1;
        double d2y = (bU * (p_up - p_ij) / h2 - bD * (p_ij - p_down) / h2) / h2;
        
        Ap[idx] = -(d2x + d2y);
    }
}

// Применение предобуславливателя D^{-1}
__global__ void apply_D_inv_kernel(const double* r, double* z,
                                   const double* Ddiag,
                                   int nx, int ny) {
    int il = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jl = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (il <= nx && jl <= ny) {
        int idx = (il-1) * ny + (jl-1);
        z[idx] = r[idx] / Ddiag[idx];
    }
}

// Операция y = y + alpha*x
__global__ void axpy_kernel(double* y, const double* x, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

// Операция p = z + beta*p
__global__ void vector_update_kernel(double* p, const double* z, double beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p[idx] = z[idx] + beta * p[idx];
    }
}

// Копирование внутренних узлов из массива с границами в плоский массив
__global__ void copy_interior_to_flat_kernel(const double* src_with_ghost, double* dst_flat,
                                             int nx, int ny) {
    int il = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jl = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (il <= nx && jl <= ny) {
        int idx_ghost = il * (ny+2) + jl;
        int idx_flat = (il-1) * ny + (jl-1);
        dst_flat[idx_flat] = src_with_ghost[idx_ghost];
    }
}

// Копирование из плоского массива в массив с границами
__global__ void copy_flat_to_interior_kernel(const double* src_flat, double* dst_with_ghost,
                                             int nx, int ny) {
    int il = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jl = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (il <= nx && jl <= ny) {
        int idx_flat = (il-1) * ny + (jl-1);
        int idx_ghost = il * (ny+2) + jl;
        dst_with_ghost[idx_ghost] = src_flat[idx_flat];
    }
}

// Обновление w_interior += alpha * p и вычисление ||alpha*p||^2 для проверки сходимости
__global__ void update_w_and_compute_diff_kernel(double* w_interior, const double* p,
                                                 double alpha, double* thread_diffs,
                                                 int n, int num_blocks) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;
    
    double local_diff_sq = 0.0;
    
    for (int i = idx; i < n; i += grid_size) {
        double inc = alpha * p[i];
        w_interior[i] += inc;
        local_diff_sq += inc * inc;
    }
    
    __syncthreads();
    
    // Записываем результат каждой нити
    double* thread_results = thread_diffs + blockIdx.x * blockDim.x;
    thread_results[tid] = local_diff_sq;
    
    __syncthreads();
    
    // Первая нить блока суммирует
    if (tid == 0) {
        double block_sum = 0.0;
        for (int i = 0; i < blockDim.x; i++) {
            block_sum += thread_results[i];
        }
        thread_diffs[blockIdx.x] = block_sum;
    }
}

// Скалярное произведение: этап 1 - каждый блок вычисляет частичную сумму
// Используем только глобальную память (правило 10 запрещает shared memory для редукции)
__global__ void dot_product_partial_kernel(const double* vec1, const double* vec2,
                                           double* block_sums, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;
    
    double sum = 0.0;
    
    // Grid-stride loop для обработки больших массивов (из лекции - ILP)
    for (int i = idx; i < n; i += grid_size) {
        sum += vec1[i] * vec2[i];
    }
    
    // Последовательная редукция внутри блока через глобальную память
    // Каждая нить записывает свой результат
    // Только первая нить блока суммирует результаты всех нитей блока
    __syncthreads();
    
    // Временный буфер в глобальной памяти для нитей блока
    // Выделяется извне и передается как block_sums + blockIdx.x * blockDim.x
    double* thread_results = block_sums + blockIdx.x * blockDim.x;
    thread_results[tid] = sum;
    
    __syncthreads();
    
    // Первая нить суммирует результаты всех нитей блока
    if (tid == 0) {
        double block_sum = 0.0;
        for (int i = 0; i < blockDim.x; i++) {
            block_sum += thread_results[i];
        }
        block_sums[blockIdx.x] = block_sum;
    }
}

// ========== Функции запуска ядер ==========

void launch_apply_A_kernel(const double* p_dev, double* Ap_dev,
                          const double* a_x_dev, const double* b_y_dev,
                          int nx, int ny, double h1, double h2,
                          cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    apply_A_kernel<<<grid, block, 0, stream>>>(p_dev, Ap_dev, a_x_dev, b_y_dev, nx, ny, h1, h2);
}

void launch_apply_D_inv_kernel(const double* r_dev, double* z_dev,
                               const double* Ddiag_dev,
                               int nx, int ny, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    apply_D_inv_kernel<<<grid, block, 0, stream>>>(r_dev, z_dev, Ddiag_dev, nx, ny);
}

void launch_axpy_kernel(double* y_dev, const double* x_dev,
                       double alpha, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    axpy_kernel<<<grid, block, 0, stream>>>(y_dev, x_dev, alpha, n);
}

void launch_vector_update_kernel(double* p_dev, const double* z_dev,
                                double beta, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    vector_update_kernel<<<grid, block, 0, stream>>>(p_dev, z_dev, beta, n);
}

void launch_copy_interior_to_device(double* flat_dev, const double* ghost_dev,
                                   int nx, int ny, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    copy_interior_to_flat_kernel<<<grid, block, 0, stream>>>(ghost_dev, flat_dev, nx, ny);
}

void launch_copy_interior_from_device(double* ghost_dev, const double* flat_dev,
                                     int nx, int ny, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    copy_flat_to_interior_kernel<<<grid, block, 0, stream>>>(flat_dev, ghost_dev, nx, ny);
}

void launch_dot_product_partial(const double* vec1_dev, const double* vec2_dev,
                               double* block_sums_dev, int n, 
                               int num_blocks, int threads_per_block,
                               cudaStream_t stream) {
    // Выделяем достаточно места: num_blocks * threads_per_block для промежуточных результатов
    dot_product_partial_kernel<<<num_blocks, threads_per_block, 0, stream>>>
        (vec1_dev, vec2_dev, block_sums_dev, n);
}

// ========== Реализация класса PoissonSolverMPICUDA ==========

// Определения статических членов
const double PoissonSolverMPICUDA::A1 = -1.0;
const double PoissonSolverMPICUDA::B1 = 1.0;
const double PoissonSolverMPICUDA::A2 = -1.0;
const double PoissonSolverMPICUDA::B2 = 1.0;

PoissonSolverMPICUDA::PoissonSolverMPICUDA(int M_, int N_, MPI_Comm comm_)
    : M(M_), N(N_), cart_comm(comm_) {
    
    double t_init_start = MPI_Wtime();
    
    MPI_Comm_rank(cart_comm, &world_rank);
    MPI_Comm_size(cart_comm, &world_size);
    
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(cart_comm, 2, dims, periods, coords);
    Px = dims[0];
    Py = dims[1];
    px = coords[0];
    py = coords[1];
    
    h1 = (B1 - A1) / M;
    h2 = (B2 - A2) / N;
    double h = max(h1, h2);
    eps = h * h;
    
    get_local_subdomain_nodes(M, N, Px, Py, px, py, ix0, ix1, iy0, iy1, nx, ny);
    
    MPI_Cart_shift(cart_comm, 0, +1, &nbr_left, &nbr_right);
    MPI_Cart_shift(cart_comm, 1, +1, &nbr_down, &nbr_up);
    
    // Инициализация таймеров
    time_init_gpu = 0.0;
    time_apply_A = 0.0;
    time_apply_D_inv = 0.0;
    time_vector_ops = 0.0;
    time_gpu_to_cpu = 0.0;
    time_cpu_to_gpu = 0.0;
    time_mpi_exchange = 0.0;
    time_mpi_allreduce = 0.0;
    time_cpu_reductions = 0.0;
    
    // Выбор GPU устройства по номеру MPI ранга
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device_id = world_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Создание CUDA events
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_stop));
    
    // Вычисление коэффициентов на хосте
    a_face_x.assign((nx+1)*ny, 0.0);
    b_face_y.assign(nx*(ny+1), 0.0);
    F.assign(nx*ny, 0.0);
    Ddiag.assign(nx*ny, 0.0);
    
    compute_coefficients();
    
    // Выделение памяти на GPU
    allocate_device_memory();
    copy_coefficients_to_device();
    
    time_init_gpu = MPI_Wtime() - t_init_start;
}

PoissonSolverMPICUDA::~PoissonSolverMPICUDA() {
    CUDA_CHECK(cudaFree(w_dev));
    CUDA_CHECK(cudaFree(w_interior_dev));
    CUDA_CHECK(cudaFree(r_dev));
    CUDA_CHECK(cudaFree(p_dev));
    CUDA_CHECK(cudaFree(Ap_dev));
    CUDA_CHECK(cudaFree(z_dev));
    CUDA_CHECK(cudaFree(a_face_x_dev));
    CUDA_CHECK(cudaFree(b_face_y_dev));
    CUDA_CHECK(cudaFree(Ddiag_dev));
    CUDA_CHECK(cudaFree(F_dev));
    CUDA_CHECK(cudaFree(reduction_buffer_dev));
    free(reduction_buffer_host);  // Обычный free вместо cudaFreeHost
    
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_stop));
}

void PoissonSolverMPICUDA::compute_coefficients() {
    // Заполнение a_face_x
    for (int il = 0; il <= nx; ++il) {
        int ig = ix0 + il - 1;
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
            fij(il, jl) = area / (h1*h2);
            
            double aL = afx(il - 1, jl - 1), aR = afx(il, jl - 1);
            double bD = bfy(il - 1, jl - 1), bU = bfy(il - 1, jl);
            dij(il, jl) = (aL + aR) / (h1*h1) + (bD + bU) / (h2*h2);
        }
    }
}

void PoissonSolverMPICUDA::allocate_device_memory() {
    int n_interior = nx * ny;
    int n_with_ghost = (nx + 2) * (ny + 2);
    
    CUDA_CHECK(cudaMalloc(&w_dev, n_with_ghost * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w_interior_dev, n_interior * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r_dev, n_interior * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&p_dev, n_interior * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Ap_dev, n_interior * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&z_dev, n_interior * sizeof(double)));
    
    CUDA_CHECK(cudaMalloc(&a_face_x_dev, (nx+1) * ny * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b_face_y_dev, nx * (ny+1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Ddiag_dev, n_interior * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&F_dev, n_interior * sizeof(double)));
    
    // Буферы для редукций: выбираем 256 нитей/блок
    reduction_threads_per_block = 256;
    num_reduction_blocks = (n_interior + reduction_threads_per_block - 1) / reduction_threads_per_block;
    // Ограничиваем число блоков для эффективности
    if (num_reduction_blocks > 256) num_reduction_blocks = 256;
    
    // Выделяем: num_blocks * threads_per_block для промежуточных результатов нитей
    int buffer_size = num_reduction_blocks * reduction_threads_per_block;
    CUDA_CHECK(cudaMalloc(&reduction_buffer_dev, buffer_size * sizeof(double)));
    
    // Обычный malloc вместо pinned memory для совместимости с IBM Spectrum MPI
    // cudaHostAlloc конфликтует с libpami_cudahook.so на CUDA-aware MPI
    reduction_buffer_host = (double*)malloc(num_reduction_blocks * sizeof(double));
    if (!reduction_buffer_host) {
        fprintf(stderr, "Failed to allocate reduction_buffer_host\n");
        exit(EXIT_FAILURE);
    }
}

void PoissonSolverMPICUDA::copy_coefficients_to_device() {
    double t0 = MPI_Wtime();
    
    CUDA_CHECK(cudaMemcpy(a_face_x_dev, a_face_x.data(), 
                         (nx+1) * ny * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_face_y_dev, b_face_y.data(), 
                         nx * (ny+1) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Ddiag_dev, Ddiag.data(), 
                         nx * ny * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(F_dev, F.data(), 
                         nx * ny * sizeof(double), cudaMemcpyHostToDevice));
    
    time_cpu_to_gpu += MPI_Wtime() - t0;
}

void PoissonSolverMPICUDA::exchange_gpu(Grid2D& U) {
    double t0 = MPI_Wtime();
    
    MPI_Status st;
    
    // Обмен вдоль Y
    vector<double> send_bottom(nx), recv_bottom(nx), send_top(nx), recv_top(nx);
    for (int il = 1; il <= nx; ++il) {
        send_bottom[il - 1] = U.at(il, 1);
        send_top[il - 1] = U.at(il, ny);
    }
    
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
    
    // Обмен вдоль X
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
    
    time_mpi_exchange += MPI_Wtime() - t0;
}

double PoissonSolverMPICUDA::dot_product_cpu(const double* vec1, const double* vec2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum * h1 * h2;
}

double PoissonSolverMPICUDA::dot_product_gpu(const double* vec1_dev, const double* vec2_dev, int n) {
    // Запуск GPU kernel для частичных сумм
    launch_dot_product_partial(vec1_dev, vec2_dev, reduction_buffer_dev, n,
                              num_reduction_blocks, reduction_threads_per_block, 0);
    
    // Копируем только num_reduction_blocks элементов (вместо nx*ny)
    double t0 = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(reduction_buffer_host, reduction_buffer_dev,
                         num_reduction_blocks * sizeof(double), cudaMemcpyDeviceToHost));
    time_gpu_to_cpu += MPI_Wtime() - t0;
    
    // Финальная редукция на CPU
    t0 = MPI_Wtime();
    double sum = 0.0;
    for (int i = 0; i < num_reduction_blocks; i++) {
        sum += reduction_buffer_host[i];
    }
    time_cpu_reductions += MPI_Wtime() - t0;
    
    return sum * h1 * h2;
}

double PoissonSolverMPICUDA::max_norm_cpu(const double* vec, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        m = max(m, fabs(vec[i]));
    }
    return m;
}

double PoissonSolverMPICUDA::euclidean_norm_global(const Grid2D& w) {
    double s_local = 0.0;
    for (int il = 1; il <= nx; ++il)
        for (int jl = 1; jl <= ny; ++jl)
            s_local += w.at(il, jl) * w.at(il, jl);
    s_local *= h1 * h2;
    
    double s_global = 0.0;
    MPI_Allreduce(&s_local, &s_global, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    return sqrt(s_global);
}

double PoissonSolverMPICUDA::max_norm_global(const Grid2D& w) {
    double m_local = 0.0;
    for (int il = 1; il <= nx; ++il)
        for (int jl = 1; jl <= ny; ++jl)
            m_local = max(m_local, fabs(w.at(il, jl)));
    
    double m_global = 0.0;
    MPI_Allreduce(&m_local, &m_global, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    return m_global;
}

void PoissonSolverMPICUDA::solve_CG_GPU(Grid2D& w, double delta, int max_iter, 
                                        int& iters, double& tsec) {
    double t_total_start = MPI_Wtime();
    
    int n_interior = nx * ny;
    
    // Инициализация w на GPU (нули)
    CUDA_CHECK(cudaMemset(w_dev, 0, (nx+2)*(ny+2)*sizeof(double)));
    CUDA_CHECK(cudaMemset(w_interior_dev, 0, n_interior*sizeof(double)));
    
    // Инициализация r = F на GPU
    double t0 = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(r_dev, F_dev, n_interior * sizeof(double), cudaMemcpyDeviceToDevice));
    time_cpu_to_gpu += MPI_Wtime() - t0;
    
    // z = D^{-1} * r
    CUDA_CHECK(cudaEventRecord(event_start));
    launch_apply_D_inv_kernel(r_dev, z_dev, Ddiag_dev, nx, ny, 0);
    CUDA_CHECK(cudaEventRecord(event_stop));
    CUDA_CHECK(cudaEventSynchronize(event_stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, event_start, event_stop));
    time_apply_D_inv += ms / 1000.0;
    
    // p = z
    t0 = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(p_dev, z_dev, n_interior * sizeof(double), cudaMemcpyDeviceToDevice));
    time_vector_ops += MPI_Wtime() - t0;
    
    // Вычисление rz_global = (z, r) на GPU
    double rz_local = dot_product_gpu(z_dev, r_dev, n_interior);
    
    double rz_global = 0.0;
    t0 = MPI_Wtime();
    MPI_Allreduce(&rz_local, &rz_global, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    time_mpi_allreduce += MPI_Wtime() - t0;
    
    for (int k = 0; k < max_iter; ++k) {
        // Копируем p на w_dev для применения оператора A (нужны граничные значения)
        launch_copy_interior_from_device(w_dev, p_dev, nx, ny, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Обмен граничных значений для p (через CPU)
        t0 = MPI_Wtime();
        CUDA_CHECK(cudaMemcpy(w.data.data(), w_dev, (nx+2)*(ny+2)*sizeof(double), cudaMemcpyDeviceToHost));
        time_gpu_to_cpu += MPI_Wtime() - t0;
        
        exchange_gpu(w);
        
        t0 = MPI_Wtime();
        CUDA_CHECK(cudaMemcpy(w_dev, w.data.data(), (nx+2)*(ny+2)*sizeof(double), cudaMemcpyHostToDevice));
        time_cpu_to_gpu += MPI_Wtime() - t0;
        
        // Применение оператора A: Ap = A * p
        CUDA_CHECK(cudaEventRecord(event_start));
        launch_apply_A_kernel(w_dev, Ap_dev, a_face_x_dev, b_face_y_dev, nx, ny, h1, h2, 0);
        CUDA_CHECK(cudaEventRecord(event_stop));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, event_start, event_stop));
        time_apply_A += ms / 1000.0;
        
        // Вычисление alpha на GPU
        double denom_local = dot_product_gpu(Ap_dev, p_dev, n_interior);
        
        double denom = 0.0;
        t0 = MPI_Wtime();
        MPI_Allreduce(&denom_local, &denom, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        time_mpi_allreduce += MPI_Wtime() - t0;
        
        double alpha = rz_global / denom;
        
        // w_interior += alpha * p и вычисление ||alpha*p||^2 на GPU
        update_w_and_compute_diff_kernel<<<num_reduction_blocks, reduction_threads_per_block>>>
            (w_interior_dev, p_dev, alpha, reduction_buffer_dev, n_interior, num_reduction_blocks);
        
        // Копируем частичные суммы diff
        t0 = MPI_Wtime();
        CUDA_CHECK(cudaMemcpy(reduction_buffer_host, reduction_buffer_dev,
                             num_reduction_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        time_gpu_to_cpu += MPI_Wtime() - t0;
        
        t0 = MPI_Wtime();
        double diff_sq_local = 0.0;
        for (int i = 0; i < num_reduction_blocks; i++) {
            diff_sq_local += reduction_buffer_host[i];
        }
        time_cpu_reductions += MPI_Wtime() - t0;
        
        double diff_sq = 0.0;
        t0 = MPI_Wtime();
        MPI_Allreduce(&diff_sq_local, &diff_sq, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        time_mpi_allreduce += MPI_Wtime() - t0;
        
        double diff_norm = sqrt(diff_sq * h1 * h2);
        if (diff_norm < delta) {
            iters = k + 1;
            tsec = MPI_Wtime() - t_total_start;
            
            // Копируем w_interior в w_dev для финального результата
            launch_copy_interior_from_device(w_dev, w_interior_dev, nx, ny, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Копируем на хост для вывода норм
            t0 = MPI_Wtime();
            CUDA_CHECK(cudaMemcpy(w.data.data(), w_dev, (nx+2)*(ny+2)*sizeof(double), cudaMemcpyDeviceToHost));
            time_gpu_to_cpu += MPI_Wtime() - t0;
            return;
        }
        
        // r = r - alpha * Ap (на GPU)
        CUDA_CHECK(cudaEventRecord(event_start));
        launch_axpy_kernel(r_dev, Ap_dev, -alpha, n_interior, 0);
        CUDA_CHECK(cudaEventRecord(event_stop));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, event_start, event_stop));
        time_vector_ops += ms / 1000.0;
        
        // z = D^{-1} * r (на GPU)
        CUDA_CHECK(cudaEventRecord(event_start));
        launch_apply_D_inv_kernel(r_dev, z_dev, Ddiag_dev, nx, ny, 0);
        CUDA_CHECK(cudaEventRecord(event_stop));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, event_start, event_stop));
        time_apply_D_inv += ms / 1000.0;
        
        // Вычисление rz_new на GPU
        double rz_new_local = dot_product_gpu(z_dev, r_dev, n_interior);
        
        double rz_new = 0.0;
        t0 = MPI_Wtime();
        MPI_Allreduce(&rz_new_local, &rz_new, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        time_mpi_allreduce += MPI_Wtime() - t0;
        
        double beta = rz_new / rz_global;
        
        // p = z + beta * p (на GPU)
        CUDA_CHECK(cudaEventRecord(event_start));
        launch_vector_update_kernel(p_dev, z_dev, beta, n_interior, 0);
        CUDA_CHECK(cudaEventRecord(event_stop));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, event_start, event_stop));
        time_vector_ops += ms / 1000.0;
        
        rz_global = rz_new;
        iters = k + 1;
    }
    
    tsec = MPI_Wtime() - t_total_start;
    
    // Копируем w_interior в w_dev для финального результата
    launch_copy_interior_from_device(w_dev, w_interior_dev, nx, ny, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Копируем финальное w обратно на хост
    t0 = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(w.data.data(), w_dev, (nx+2)*(ny+2)*sizeof(double), cudaMemcpyDeviceToHost));
    time_gpu_to_cpu += MPI_Wtime() - t0;
}

// ========== Главная программа ==========

struct Args {
    int M = 40, N = 40;
    int Px = 0, Py = 0;
    double delta = 1e-6;
    int max_iter = 200000;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--M") && i + 1 < argc) a.M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) a.N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--Px") && i + 1 < argc) a.Px = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--Py") && i + 1 < argc) a.Py = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--delta") && i + 1 < argc) a.delta = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max_iter") && i + 1 < argc) a.max_iter = atoi(argv[++i]);
    }
    return a;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    Args args = parse_args(argc, argv);
    
    int Px = args.Px, Py = args.Py;
    if (Px == 0 || Py == 0) {
        choose_process_grid(args.M, args.N, world_size, Px, Py);
    }
    
    if (Px * Py != world_size) {
        if (world_rank == 0) {
            printf("Error: Px (%d) * Py (%d) != world_size (%d)\n", Px, Py, world_size);
        }
        MPI_Finalize();
        return 1;
    }
    
    int dims[2] = {Px, Py};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    
    PoissonSolverMPICUDA solver(args.M, args.N, cart_comm);
    
    Grid2D w(solver.nx, solver.ny);
    for (int i = 0; i <= solver.nx + 1; ++i)
        for (int j = 0; j <= solver.ny + 1; ++j)
            w.at(i, j) = 0.0;
    
    int iters = 0;
    double tsec = 0.0;
    solver.solve_CG_GPU(w, args.delta, args.max_iter, iters, tsec);
    
    double nE = solver.euclidean_norm_global(w);
    double nC = solver.max_norm_global(w);
    
    if (world_rank == 0) {
        printf("=== MPI+CUDA Poisson Solver ===\n");
        printf("MPI grid: %d x %d (procs=%d)\n", solver.Px, solver.Py, world_size);
        printf("Global grid: %d x %d\n", args.M, args.N);
        printf("Iterations: %d\n", iters);
        printf("Total time: %.6f s\n", tsec);
        printf("||w||_E = %e, ||w||_C = %e\n", nE, nC);
        printf("\n=== Detailed timings ===\n");
        printf("GPU initialization:  %.6f s\n", solver.time_init_gpu);
        printf("apply_A kernels:     %.6f s\n", solver.time_apply_A);
        printf("apply_D_inv kernels: %.6f s\n", solver.time_apply_D_inv);
        printf("Vector ops kernels:  %.6f s\n", solver.time_vector_ops);
        printf("GPU->CPU copies:     %.6f s\n", solver.time_gpu_to_cpu);
        printf("CPU->GPU copies:     %.6f s\n", solver.time_cpu_to_gpu);
        printf("MPI exchange:        %.6f s\n", solver.time_mpi_exchange);
        printf("MPI allreduce:       %.6f s\n", solver.time_mpi_allreduce);
        printf("CPU reductions:      %.6f s\n", solver.time_cpu_reductions);
    }
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
