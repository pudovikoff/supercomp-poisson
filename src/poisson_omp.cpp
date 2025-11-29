#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <omp.h>

using namespace std;

// Область (вариант 7 — «сапожок»)
const double A1 = -1.0, B1 = 1.0;  // границы по x
const double A2 = -1.0, B2 = 1.0;  // границы по y

bool is_in_domain(double x, double y) {
    if (x <= -1.0 || x >= 1.0 || y <= -1.0 || y >= 1.0) return false;
    if (x >= 0.0 && y >= 0.0) return false; // вырезанный квадрант
    return true;
}

// Длина части отрезка [P1,P2], лежащей в D (упрощенно для «сапожка»)
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

class PoissonSolverOMP {
public:
    int M, N;        // интервалов по x и y
    double h1, h2;   // шаги сетки
    double eps;      // параметр фиктивных областей

    vector<vector<double>> a, b, F, D_diag;
    
    // Таймеры для детального анализа
    double time_coefficients_init = 0.0;  // Инициализация коэффициентов
    double time_apply_A = 0.0;            // Применение оператора A
    double time_apply_D_inv = 0.0;        // Применение предобуславливателя D^{-1}
    double time_vector_ops = 0.0;         // Векторные операции
    double time_reductions = 0.0;         // Редукции (скалярные произведения, нормы)
    double time_total = 0.0;              // Общее время работы solve_CG

    PoissonSolverOMP(int M_, int N_) : M(M_), N(N_) {
        h1 = (B1 - A1) / M; h2 = (B2 - A2) / N; eps = max(h1, h2) * max(h1, h2);
        a.assign(M + 1, vector<double>(N + 1, 0.0));
        b.assign(M + 1, vector<double>(N + 1, 0.0));
        F.assign(M - 1, vector<double>(N - 1, 0.0));
        D_diag.assign(M - 1, vector<double>(N - 1, 0.0));
        compute_coefficients();
    }

    void compute_coefficients() {
        double t0 = omp_get_wtime();  // Начало измерения
        
        // a[i][j]
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                double x = A1 + (i - 0.5) * h1;
                double y1c = A2 + (j - 0.5) * h2;
                double y2c = y1c + h2;
                double lin = segment_length_in_D(x, y1c, x, y2c);
                double lout = h2 - lin;
                a[i][j] = (lin + lout / eps) / h2;
            }
        }
        // b[i][j]
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                double x1c = A1 + (i - 0.5) * h1;
                double x2c = x1c + h1;
                double y = A2 + (j - 0.5) * h2;
                double lin = segment_length_in_D(x1c, y, x2c, y);
                double lout = h1 - lin;
                b[i][j] = (lin + lout / eps) / h1;
            }
        }
        // F[i][j]
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                double x = A1 + (i + 1) * h1;
                double y = A2 + (j + 1) * h2;
                double xl = x - 0.5 * h1, xr = x + 0.5 * h1;
                double yb = y - 0.5 * h2, yt = y + 0.5 * h2;
                double area = cell_area_in_D(xl, xr, yb, yt);
                F[i][j] = area / (h1 * h2); // f=1 в D
            }
        }
        // D_diag
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                D_diag[i][j] = (a[i + 2][j + 1] + a[i + 1][j + 1]) / (h1 * h1) +
                                (b[i + 1][j + 2] + b[i + 1][j + 1]) / (h2 * h2);
            }
        }
        
        time_coefficients_init += omp_get_wtime() - t0;  // Конец измерения
    }

    void apply_A(const vector<vector<double>>& w, vector<vector<double>>& res) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < M - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                double w_ij = w[i][j];
                double w_im1j = (i > 0)     ? w[i - 1][j] : 0.0;
                double w_ip1j = (i < M - 2) ? w[i + 1][j] : 0.0;
                double w_ijm1 = (j > 0)     ? w[i][j - 1] : 0.0;
                double w_ijp1 = (j < N - 2) ? w[i][j + 1] : 0.0;
                double d2x = (a[i + 2][j + 1] * (w_ip1j - w_ij) / h1 - a[i + 1][j + 1] * (w_ij - w_im1j) / h1) / h1;
                double d2y = (b[i + 1][j + 2] * (w_ijp1 - w_ij) / h2 - b[i + 1][j + 1] * (w_ij - w_ijm1) / h2) / h2;
                res[i][j] = -(d2x + d2y);
            }
        }
    }

    void apply_D_inv(const vector<vector<double>>& r, vector<vector<double>>& z) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < M - 1; ++i)
            for (int j = 0; j < N - 1; ++j)
                z[i][j] = r[i][j] / D_diag[i][j];
    }

    double dot_product(const vector<vector<double>>& u, const vector<vector<double>>& v) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static) collapse(2)
        for (int i = 0; i < M - 1; ++i)
            for (int j = 0; j < N - 1; ++j)
                sum += u[i][j] * v[i][j];
        return sum * h1 * h2;
    }

    double euclidean_norm(const vector<vector<double>>& u) { return sqrt(dot_product(u, u)); }

    double max_norm(const vector<vector<double>>& u) {
        double max_val = 0.0;
        #pragma omp parallel for reduction(max:max_val) schedule(static) collapse(2)
        for (int i = 0; i < M - 1; ++i)
            for (int j = 0; j < N - 1; ++j)
                max_val = max(max_val, fabs(u[i][j]));
        return max_val;
    }

    void solve_CG(vector<vector<double>>& w, double delta, int max_iter, int& iters, double& tsec) {
        double time_cg_start = omp_get_wtime();  // Начало общего таймера
        double t0 = omp_get_wtime();
        vector<vector<double>> r(M - 1, vector<double>(N - 1, 0.0));
        vector<vector<double>> z(M - 1, vector<double>(N - 1, 0.0));
        vector<vector<double>> p(M - 1, vector<double>(N - 1, 0.0));
        vector<vector<double>> Ap(M - 1, vector<double>(N - 1, 0.0));
        w.assign(M - 1, vector<double>(N - 1, 0.0));
        r = F; // r(0)
        
        t0 = omp_get_wtime();
        apply_D_inv(r, z); // z(0)
        time_apply_D_inv += omp_get_wtime() - t0;
        
        p = z; // p(1)
        t0 = omp_get_wtime();
        double rz_old = dot_product(z, r);
        time_reductions += omp_get_wtime() - t0;

        // Мониторинг монотонности
        double H_prev = 0.0;
        apply_A(w, Ap);
        #pragma omp parallel for reduction(+:H_prev) schedule(static) collapse(2)
        for (int i = 0; i < M - 1; ++i)
            for (int j = 0; j < N - 1; ++j)
                H_prev += (F[i][j] + r[i][j]) * w[i][j];
        H_prev *= h1 * h2;

        for (int k = 0; k < max_iter; ++k) {
            t0 = omp_get_wtime();
            apply_A(p, Ap);
            time_apply_A += omp_get_wtime() - t0;
            
            t0 = omp_get_wtime();
            double denom = dot_product(Ap, p);
            double alpha = rz_old / denom;
            time_reductions += omp_get_wtime() - t0;

            // w = w + alpha * p
            t0 = omp_get_wtime();
            #pragma omp parallel for schedule(static) collapse(2)
            for (int i = 0; i < M - 1; ++i)
                for (int j = 0; j < N - 1; ++j)
                    w[i][j] += alpha * p[i][j];
            time_vector_ops += omp_get_wtime() - t0;

            // diff_norm ~= ||alpha*p||_E
            t0 = omp_get_wtime();
            double diff_sq = 0.0;
            #pragma omp parallel for reduction(+:diff_sq) schedule(static) collapse(2)
            for (int i = 0; i < M - 1; ++i)
                for (int j = 0; j < N - 1; ++j)
                    diff_sq += (alpha * p[i][j]) * (alpha * p[i][j]);
            double diff_norm = sqrt(diff_sq * h1 * h2);
            time_reductions += omp_get_wtime() - t0;
            
            if (diff_norm < delta) {
                time_total += omp_get_wtime() - time_cg_start;  // Конец общего таймера
                iters = k + 1;
                tsec = omp_get_wtime() - t0;
                return;
            }

            // r = r - alpha * Ap
            t0 = omp_get_wtime();
            #pragma omp parallel for schedule(static) collapse(2)
            for (int i = 0; i < M - 1; ++i)
                for (int j = 0; j < N - 1; ++j)
                    r[i][j] -= alpha * Ap[i][j];
            time_vector_ops += omp_get_wtime() - t0;

            t0 = omp_get_wtime();
            apply_D_inv(r, z);
            time_apply_D_inv += omp_get_wtime() - t0;
            
            t0 = omp_get_wtime();
            double rz_new = dot_product(z, r);
            time_reductions += omp_get_wtime() - t0;

            // Контроль монотонности H(w)
            double H_curr = 0.0;
            apply_A(w, Ap);
            #pragma omp parallel for reduction(+:H_curr) schedule(static) collapse(2)
            for (int i = 0; i < M - 1; ++i)
                for (int j = 0; j < N - 1; ++j)
                    H_curr += (F[i][j] + r[i][j]) * w[i][j];
            H_curr *= h1 * h2;
            if (H_curr < H_prev) {
                r = F; apply_D_inv(r, z); p = z; rz_old = dot_product(z, r); H_prev = H_curr; continue;
            }
            H_prev = H_curr;

            double beta = rz_new / rz_old;
            t0 = omp_get_wtime();
            #pragma omp parallel for schedule(static) collapse(2)
            for (int i = 0; i < M - 1; ++i)
                for (int j = 0; j < N - 1; ++j)
                    p[i][j] = z[i][j] + beta * p[i][j];
            time_vector_ops += omp_get_wtime() - t0;
            
            rz_old = rz_new;
        }
        time_total += omp_get_wtime() - time_cg_start;  // Конец общего таймера
        iters = max_iter; tsec = omp_get_wtime() - t0;
    }

    void save_solution(const vector<vector<double>>& w, const string& filename) {
        ofstream f(filename);
        f << scientific << setprecision(10);
        for (int j = N - 2; j >= 0; --j)
            for (int i = 0; i < M - 1; ++i) {
                double x = A1 + (i + 1) * h1;
                double y = A2 + (j + 1) * h2;
                f << x << " " << y << " " << w[i][j] << "\n";
            }
    }
};

struct Args { int M = 40, N = 40; int threads = 0; double delta = 1e-6; int max_iter = 200000; string out = ""; };

Args parse_args(int argc, char* argv[]) {
    Args a; for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--M") && i + 1 < argc) a.M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) a.N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) a.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--delta") && i + 1 < argc) a.delta = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max_iter") && i + 1 < argc) a.max_iter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) a.out = argv[++i];
    } return a;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    if (args.threads > 0) omp_set_num_threads(args.threads);

    cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    cout << "Grid: " << args.M << " x " << args.N << "\n";

    PoissonSolverOMP solver(args.M, args.N);
    vector<vector<double>> w; int iters = 0; double tsec = 0.0;
    solver.solve_CG(w, args.delta, args.max_iter, iters, tsec);

    double nE = solver.euclidean_norm(w); double nC = solver.max_norm(w);
    cout << "Iterations: " << iters << "\n";
    cout << fixed << setprecision(6) << "Time: " << tsec << " s\n";
    cout << scientific << "||w||_E = " << nE << ", ||w||_C = " << nC << "\n";
    
    // Вывод детального тайминга
    cout << "\n=== Timing Breakdown ===" << endl;
    cout << "Coefficients init:    " << fixed << setprecision(6) << solver.time_coefficients_init << " s" << endl;
    cout << "apply_A:              " << fixed << setprecision(6) << solver.time_apply_A << " s" << endl;
    cout << "apply_D_inv:          " << fixed << setprecision(6) << solver.time_apply_D_inv << " s" << endl;
    cout << "Vector operations:    " << fixed << setprecision(6) << solver.time_vector_ops << " s" << endl;
    cout << "Local reductions:     " << fixed << setprecision(6) << solver.time_reductions << " s" << endl;
    cout << "---" << endl;
    cout << "Total time (solve):   " << fixed << setprecision(6) << solver.time_total << " s" << endl << endl;

    string out = args.out.empty() ? (string("results/solution_") + to_string(args.M) + "x" + to_string(args.N) + ".txt") : args.out;
    solver.save_solution(w, out);
    cout << "Saved: " << out << "\n";
    return 0;
}