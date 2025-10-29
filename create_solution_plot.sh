#!/bin/bash
# Создание визуализации решения с помощью gnuplot

if ! command -v gnuplot &> /dev/null; then
    echo "⚠️  gnuplot не установлен. Установите: brew install gnuplot"
    echo "Визуализация пропущена."
    exit 0
fi

cat > /tmp/solution.gnuplot << 'EOF'
set terminal pngcairo size 1200,500 enhanced font 'Arial,12'
set output 'results/plot_solution.png'

set multiplot layout 1,2

# Первый график - цветная карта
set title 'Решение задачи Пуассона (сапожок)' font ',14'
set xlabel 'x'
set ylabel 'y'
set size ratio -1
set xrange [-1:1]
set yrange [-1:1]
set cbrange [0:0.2]
set palette defined (0 "blue", 0.5 "green", 1 "red")
set pm3d map
set pm3d interpolate 2,2
splot 'results/solution_40x40.txt' using 1:2:3 with pm3d notitle

# Второй график - контурные линии
set title 'Изолинии решения' font ',14'
set xlabel 'x'
set ylabel 'y'
set size ratio -1
set xrange [-1:1]
set yrange [-1:1]
set contour base
set cntrparam levels 15
unset surface
set view map
set style line 1 lc rgb '#000000' lt 1 lw 1
splot 'results/solution_40x40.txt' using 1:2:3 with lines ls 1 notitle

unset multiplot
EOF

gnuplot /tmp/solution.gnuplot 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ Изображение решения создано: results/plot_solution.png"
else
    echo "❌ Ошибка при создании изображения"
    exit 1
fi
