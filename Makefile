# Компиляторы
# Для обычного кода (OpenMP, sequential) используем g++/xlc
CXX ?= g++
# Для MPI ВСЕГДА используем mpicxx (он автоматически линкует библиотеки MPI)
MPICXX ?= mpicxx
CXXFLAGS_BASE = -std=c++11 -O2 -Wall

# Модуль загрузки MPI (для HPC кластера)
# На локальной машине это может быть пусто, на кластере нужен SpectrumMPI
MODULE_LOAD_MPI ?= module load SpectrumMPI 2>/dev/null ||

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

SEQ_BIN = $(BIN_DIR)/poisson_sequential
OMP_BIN = $(BIN_DIR)/poisson_omp
MPI_BIN = $(BIN_DIR)/poisson_mpi

# Тесты
TEST_DECOMP_SRC = $(TESTS_DIR)/test_domain_decomposition.cpp
TEST_DECOMP_BIN = $(BIN_DIR)/test_domain_decomposition

# Цели сборки
# По умолчанию собираем всё (может быть проблема с OpenMP на macOS)
all: $(OMP_BIN) $(MPI_BIN)

# Только MPI версия
mpi: $(MPI_BIN)

# OpenMP версия
omp: $(OMP_BIN)

# Sequential версия
seq: $(SEQ_BIN)

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

$(TEST_DECOMP_BIN): $(TEST_DECOMP_SRC) $(SRC_DIR)/domain_decomposition.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) -o $@ $(TEST_DECOMP_SRC)

clean:
	rm -f $(SEQ_BIN) $(OMP_BIN) $(MPI_BIN) $(TEST_DECOMP_BIN)
	rm -f $(RESULTS_DIR)/solution_*.txt

run_seq: $(SEQ_BIN) | $(RESULTS_DIR)
	$<

# Пример: make run_omp M=40 N=40 THREADS=4
run_omp: $(OMP_BIN) | $(RESULTS_DIR)
	OMP_NUM_THREADS=$(THREADS) $< --M $(M) --N $(N)

# Пример: make run_mpi NP=4 M=40 N=40
run_mpi: $(MPI_BIN) | $(RESULTS_DIR)
	$(MODULE_LOAD_MPI) mpirun -np $(NP) $< --M $(M) --N $(N)

# Локальный тест разбиения
test_decomp: $(TEST_DECOMP_BIN)
	$<

.PHONY: all mpi omp seq clean run_seq run_omp run_mpi test_decomp help

# Справка
help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  Makefile для Poisson Solver"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Цели сборки:"
	@echo "  make all        - собрать все версии (OpenMP + MPI)"
	@echo "  make mpi        - собрать MPI версию"
	@echo "  make omp        - собрать OpenMP версию"
	@echo "  make seq        - собрать последовательную версию"
	@echo ""
	@echo "Запуск:"
	@echo "  make run_mpi NP=4 M=40 N=40         - запустить MPI версию"
	@echo "  make run_omp M=40 N=40 THREADS=4    - запустить OpenMP версию"
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
