# Компиляторы
# Для обычного кода (OpenMP, sequential) используем g++/xlc
CXX ?= g++
# Для MPI ВСЕГДА используем mpicxx (он автоматически линкует библиотеки MPI)
MPICXX ?= mpicxx
CXXFLAGS_BASE = -std=c++11 -O2 -Wall

# CUDA компилятор и параметры (для MPI+CUDA версии)
NVCC ?= nvcc
ARCH ?= sm_35
HOST_COMP ?= mpicc
NVCCFLAGS = -arch=$(ARCH) -ccbin=$(HOST_COMP) -std=c++11 -O3 -Xcompiler -fPIC
NVCCLINKFLAGS = -lstdc++ -lm

# Модуль загрузки MPI (для HPC кластера)
MODULE_LOAD_MPI ?= module load SpectrumMPI 2>/dev/null ;

# Флаги OpenMP для разных компиляторов
# IBM XL C/C++: xlc_r, xlC_r, xlC
ifneq (,$(findstring xl,$(CXX)))
	OMPFLAGS = -qsmp=omp -qarch=pwr8 -mcpu=power8
else
	OMPFLAGS = -fopenmp
endif

# Директории
SRC_DIR = src
BIN_DIR = bin
RESULTS_DIR = results
TESTS_DIR = tests

# Файлы
SEQ_SRC = $(SRC_DIR)/poisson_sequential.cpp
OMP_SRC = $(SRC_DIR)/poisson_omp.cpp
MPI_SRC = $(SRC_DIR)/poisson_mpi.cpp
MPI_OMP_SRC = $(SRC_DIR)/poisson_mpi_omp.cpp
MPI_CUDA_SRC = $(SRC_DIR)/poisson_mpi_cuda.cu

SEQ_BIN = $(BIN_DIR)/poisson_sequential
OMP_BIN = $(BIN_DIR)/poisson_omp
MPI_BIN = $(BIN_DIR)/poisson_mpi
MPI_OMP_BIN = $(BIN_DIR)/poisson_mpi_omp
MPI_CUDA_BIN = $(BIN_DIR)/poisson_mpi_cuda

# Тесты
TEST_DECOMP_SRC = $(TESTS_DIR)/test_domain_decomposition.cpp
TEST_DECOMP_BIN = $(BIN_DIR)/test_domain_decomposition

# Цели сборки
# По умолчанию собираем всё (OpenMP может быть проблема на macOS)
all: $(SEQ_BIN) $(OMP_BIN) $(MPI_BIN) $(MPI_OMP_BIN) $(MPI_CUDA_BIN)

# Только MPI версия
mpi: $(MPI_BIN)

# OpenMP версия
omp: $(OMP_BIN)

# Sequential версия
seq: $(SEQ_BIN)

# Гибридная MPI+OpenMP версия
mpi_omp: $(MPI_OMP_BIN)

# MPI+CUDA версия
mpi_cuda: $(MPI_CUDA_BIN)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(RESULTS_DIR):
	@mkdir -p $(RESULTS_DIR)

$(SEQ_BIN): $(SEQ_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) -o $@ $<

$(OMP_BIN): $(OMP_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) $(OMPFLAGS) -o $@ $<

$(MPI_BIN): $(MPI_SRC) $(SRC_DIR)/domain_decomposition.h | $(BIN_DIR)
	$(MODULE_LOAD_MPI) $(MPICXX) $(CXXFLAGS_BASE) -o $@ $(MPI_SRC)

$(MPI_OMP_BIN): $(MPI_OMP_SRC) $(SRC_DIR)/poisson_solver_mpi_omp.h $(SRC_DIR)/domain_decomposition.h | $(BIN_DIR)
	$(MODULE_LOAD_MPI) $(MPICXX) $(CXXFLAGS_BASE) $(OMPFLAGS) -o $@ $(MPI_OMP_SRC)

$(MPI_CUDA_BIN): $(MPI_CUDA_SRC) $(SRC_DIR)/poisson_solver_mpi_cuda.h $(SRC_DIR)/domain_decomposition.h | $(BIN_DIR)
	$(MODULE_LOAD_MPI) $(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) $(NVCCLINKFLAGS) -o $@ $(MPI_CUDA_SRC)

$(TEST_DECOMP_BIN): $(TEST_DECOMP_SRC) $(SRC_DIR)/domain_decomposition.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) -o $@ $(TEST_DECOMP_SRC)

clean:
	rm -f $(SEQ_BIN) $(OMP_BIN) $(MPI_BIN) $(MPI_OMP_BIN) $(MPI_CUDA_BIN) $(TEST_DECOMP_BIN)
	rm -f $(RESULTS_DIR)/solution_*.txt

run_seq: $(SEQ_BIN) | $(RESULTS_DIR)
	$<

# Пример: make run_omp M=40 N=40 THREADS=4
run_omp: $(OMP_BIN) | $(RESULTS_DIR)
	OMP_NUM_THREADS=$(THREADS) $< --M $(M) --N $(N)

# Пример: make run_mpi NP=4 M=40 N=40
run_mpi: $(MPI_BIN) | $(RESULTS_DIR)
	$(MODULE_LOAD_MPI) mpirun -np $(NP) $< --M $(M) --N $(N)

# Пример: make run_mpi_omp NP=2 M=40 N=40 THREADS=4
run_mpi_omp: $(MPI_OMP_BIN) | $(RESULTS_DIR)
	$(MODULE_LOAD_MPI) mpirun -np $(NP) $< --M $(M) --N $(N) --threads $(THREADS)

# Пример: make run_mpi_cuda NP=4 M=400 N=400
run_mpi_cuda: $(MPI_CUDA_BIN) | $(RESULTS_DIR)
	$(MODULE_LOAD_MPI) mpirun -np $(NP) $< --M $(M) --N $(N)

# Локальный тест разбиения
test_decomp: $(TEST_DECOMP_BIN)
	$<

.PHONY: all mpi omp seq mpi_omp mpi_cuda clean run_seq run_omp run_mpi run_mpi_omp run_mpi_cuda test_decomp help

# Справка
help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  Makefile для Poisson Solver"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Цели сборки:"
	@echo "  make all        - собрать все версии (OpenMP + MPI + MPI+OpenMP)"
	@echo "  make mpi        - собрать MPI версию"
	@echo "  make omp        - собрать OpenMP версию"
	@echo "  make seq        - собрать последовательную версию"
	@echo "  make mpi_omp    - собрать гибридную MPI+OpenMP версию"
	@echo "  make mpi_cuda   - собрать MPI+CUDA версию (ARCH=sm_35 HOST_COMP=mpicc)"
	@echo ""
	@echo "Запуск:"
	@echo "  make run_mpi NP=4 M=40 N=40                - запустить MPI версию"
	@echo "  make run_omp M=40 N=40 THREADS=4           - запустить OpenMP версию"
	@echo "  make run_mpi_omp NP=2 M=40 N=40 THREADS=4  - запустить MPI+OpenMP версию"
	@echo "  make run_mpi_cuda NP=4 M=400 N=400         - запустить MPI+CUDA версию"
	@echo ""
	@echo "Примечание:"
	@echo "  - MPI версии используют mpicxx (автоматическая линковка библиотек)"
	@echo "  - Модуль SpectrumMPI загружается автоматически перед компиляцией"
	@echo "  - На локальной машине: make mpi (игнорирует module load если недоступен)"
	@echo "  - На кластере (Polus): make mpi (загружает SpectrumMPI автоматически)"
	@echo ""
	@echo "Прочее:"
	@echo "  make clean      - очистить бинарные файлы"
	@echo "  make test_decomp - запустить тест разбиения"
	@echo "  make help       - показать эту справку"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
