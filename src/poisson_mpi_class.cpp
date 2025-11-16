#include <mpi.h>
#include <cstdio>
#include <cstring>
#include "poisson_solver_mpi.h"

struct Args {
    int M = 40, N = 40;
    double delta = 1e-6;
    int max_iter = 200000;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--M") && i + 1 < argc) a.M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) a.N = atoi(argv[++i]);
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

    // Создаём 2D декартовую топологию
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims); // сбалансированно
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    // Создаём solver
    PoissonSolverMPI solver(args.M, args.N, cart_comm);

    // Инициализируем решение
    Grid2D w(solver.nx, solver.ny);
    for (int i = 0; i <= solver.nx + 1; ++i)
        for (int j = 0; j <= solver.ny + 1; ++j)
            w.at(i, j) = 0.0;

    // Решаем
    int iters = 0;
    double tsec = 0.0;
    solver.solve_CG(w, args.delta, args.max_iter, iters, tsec);

    // Нормы решения
    double nE = solver.euclidean_norm_global(w);
    double nC = solver.max_norm_global(w);

    if (world_rank == 0) {
        printf("MPI grid: %d x %d (procs=%d)\n", solver.Px, solver.Py, world_size);
        printf("Global grid: %d x %d\n", args.M, args.N);
        printf("Iterations: %d\n", iters);
        printf("Time: %lf s\n", tsec);
        printf("||w||_E = %e, ||w||_C = %e\n", nE, nC);
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
