# Компилятор и флаги (поддержка OpenMP и MPI)
# По умолчанию используем g++; для Polus рекомендуется xlc_r и mpixlc++
CXX ?= g++
MPICXX ?= mpicxx
CXXFLAGS_BASE = -std=c++11 -O2 -Wall

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
MPI_CLASS_SRC = $(SRC_DIR)/poisson_mpi_class.cpp
SEQ_BIN = $(BIN_DIR)/poisson_sequential
OMP_BIN = $(BIN_DIR)/poisson_omp
MPI_BIN = $(BIN_DIR)/poisson_mpi
MPI_CLASS_BIN = $(BIN_DIR)/poisson_mpi_class

# Тесты
TEST_DECOMP_SRC = $(TESTS_DIR)/test_domain_decomposition.cpp
TEST_DECOMP_BIN = $(BIN_DIR)/test_domain_decomposition

# Цели сборки
all: $(OMP_BIN) $(MPI_BIN) $(MPI_CLASS_BIN)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(RESULTS_DIR):
	@mkdir -p $(RESULTS_DIR)

$(SEQ_BIN): $(SEQ_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) -o $@ $<

$(OMP_BIN): $(OMP_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) $(OMPFLAGS) -o $@ $<

$(MPI_BIN): $(MPI_SRC) $(SRC_DIR)/domain_decomposition.h | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS_BASE) -o $@ $(MPI_SRC)

$(MPI_CLASS_BIN): $(MPI_CLASS_SRC) $(SRC_DIR)/domain_decomposition.h $(SRC_DIR)/poisson_solver_mpi.h | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS_BASE) -o $@ $(MPI_CLASS_SRC)

$(TEST_DECOMP_BIN): $(TEST_DECOMP_SRC) $(SRC_DIR)/domain_decomposition.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) -o $@ $(TEST_DECOMP_SRC)

clean:
	rm -f $(SEQ_BIN) $(OMP_BIN) $(MPI_BIN) $(MPI_CLASS_BIN) $(TEST_DECOMP_BIN)
	rm -f $(RESULTS_DIR)/solution_*.txt

run_seq: $(SEQ_BIN) | $(RESULTS_DIR)
	$<

# Пример: make run_omp M=40 N=40 THREADS=4
run_omp: $(OMP_BIN) | $(RESULTS_DIR)
	OMP_NUM_THREADS=$(THREADS) $< --M $(M) --N $(N)

# Пример: make run_mpi NP=4 M=40 N=40
run_mpi: $(MPI_BIN) | $(RESULTS_DIR)
	mpirun -np $(NP) $< --M $(M) --N $(N)

# Пример: make run_mpi_class NP=4 M=40 N=40
run_mpi_class: $(MPI_CLASS_BIN) | $(RESULTS_DIR)
	mpirun -np $(NP) $< --M $(M) --N $(N)

# Локальный тест разбиения
test_decomp: $(TEST_DECOMP_BIN)
	$<

.PHONY: all clean run_seq run_omp run_mpi run_mpi_class test_decomp
