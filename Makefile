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
# По умолчанию собираем всё (может быть проблема с OpenMP на macOS)
all: $(OMP_BIN) $(MPI_BIN) $(MPI_CLASS_BIN)

# Только MPI версии (старая + новая с классом)
mpi: $(MPI_BIN) $(MPI_CLASS_BIN)

# Только новая MPI версия с классом
mpi_class: $(MPI_CLASS_BIN)

# Только старая MPI версия (без класса)
mpi_old: $(MPI_BIN)

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

.PHONY: all mpi mpi_class mpi_old omp seq clean run_seq run_omp run_mpi run_mpi_class test_decomp help

# Справка
help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  Makefile для Poisson Solver"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Цели сборки:"
	@echo "  make all        - собрать все версии (OpenMP + MPI)"
	@echo "  make mpi        - собрать обе MPI версии (старая + новая)"
	@echo "  make mpi_class  - собрать только новую MPI версию (с классом) ★"
	@echo "  make mpi_old    - собрать старую MPI версию (без класса)"
	@echo "  make omp        - собрать OpenMP версию"
	@echo "  make seq        - собрать последовательную версию"
	@echo ""
	@echo "Запуск:"
	@echo "  make run_mpi_class NP=4 M=40 N=40   - запустить новую MPI версию ★"
	@echo "  make run_mpi NP=4 M=40 N=40         - запустить старую MPI версию"
	@echo "  make run_omp M=40 N=40 THREADS=4    - запустить OpenMP версию"
	@echo ""
	@echo "Прочее:"
	@echo "  make clean      - очистить бинарные файлы"
	@echo "  make test_decomp - запустить тест разбиения"
	@echo "  make help       - показать эту справку"
	@echo ""
	@echo "★ = рекомендуемые цели для новой версии с классом"
	@echo "═══════════════════════════════════════════════════════════════"
