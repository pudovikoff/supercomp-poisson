# Компилятор и флаги (поддержка OpenMP и Polus)
# По умолчанию используем g++; для Polus рекомендуется xlc_r
CXX ?= g++
CXXFLAGS_BASE = -std=c++11 -O3 -Wall

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

# Файлы
SEQ_SRC = $(SRC_DIR)/poisson_sequential.cpp
OMP_SRC = $(SRC_DIR)/poisson_omp.cpp
SEQ_BIN = $(BIN_DIR)/poisson_sequential
OMP_BIN = $(BIN_DIR)/poisson_omp

# Цели сборки
all: $(OMP_BIN)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(RESULTS_DIR):
	@mkdir -p $(RESULTS_DIR)

$(SEQ_BIN): $(SEQ_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) -o $@ $<

$(OMP_BIN): $(OMP_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_BASE) $(OMPFLAGS) -o $@ $<

clean:
	rm -f $(SEQ_BIN) $(OMP_BIN)
	rm -f $(RESULTS_DIR)/solution_*.txt

run_seq: $(SEQ_BIN) | $(RESULTS_DIR)
	$<

# Пример: make run_omp M=40 N=40 THREADS=4
run_omp: $(OMP_BIN) | $(RESULTS_DIR)
	OMP_NUM_THREADS=$(THREADS) $< --M $(M) --N $(N)

.PHONY: all clean run_seq run_omp
