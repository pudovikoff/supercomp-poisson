# Компилятор и флаги
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall

# Директории
SRC_DIR = src
TEST_DIR = tests
RESULTS_DIR = results

# Цели сборки
all: poisson_sequential

poisson_sequential: $(SRC_DIR)/poisson_sequential.cpp
	$(CXX) $(CXXFLAGS) -o poisson_sequential $(SRC_DIR)/poisson_sequential.cpp

tests: $(TEST_DIR)/test_poisson.cpp
	$(CXX) $(CXXFLAGS) -o test_poisson $(TEST_DIR)/test_poisson.cpp

clean:
	rm -f poisson_sequential test_poisson
	rm -f $(RESULTS_DIR)/solution_*.txt

run: poisson_sequential
	./poisson_sequential

test: tests
	./test_poisson

.PHONY: all clean run test
