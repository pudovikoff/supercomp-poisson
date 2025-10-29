#!/usr/bin/env bash
# Простая отправка OpenMP-задачи на Polus через mpisubmit.pl
# Использование: ./submit_omp.sh <threads> [M] [N]
# Примеры:
#   ./submit_omp.sh 4              # 4 нити, сетка 40x40
#   ./submit_omp.sh 8 400 600      # 8 нитей, сетка 400x600

set -euo pipefail

THREADS=${1:-4}
M=${2:-40}
N=${3:-40}

BIN=bin/poisson_omp
if [[ ! -x "$BIN" ]]; then
  echo "Не найден $BIN. Сначала выполните: make"
  exit 1
fi

# Вывод в файлы
OUT=poisson_omp_${M}x${N}_t${THREADS}.out
ERR=poisson_omp_${M}x${N}_t${THREADS}.err

# На Polus достаточно одной MPI-задачи и t нитей OpenMP
# Переменная OMP_NUM_THREADS устанавливается автоматически mpisubmit.pl
mpisubmit.pl -p 1 -t ${THREADS} --stdout ${OUT} --stderr ${ERR} ${BIN} -- --M ${M} --N ${N}

echo "Отправлено. Логи: ${OUT}, ${ERR}"