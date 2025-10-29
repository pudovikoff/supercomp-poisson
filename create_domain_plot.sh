#!/bin/bash
# Простой способ создать изображение области без matplotlib

cat > /tmp/domain.gnuplot << 'EOF'
set terminal png size 800,800
set output 'results/plot_domain.png'
set size square
set xrange [-1.1:1.1]
set yrange [-1.1:1.1]
set xlabel 'x'
set ylabel 'y'
set title 'Область-сапожок (вариант 7)'
set grid
set style fill solid 0.3
set style rect fc rgb "lightblue" fs solid 0.3

# Рисуем область D (сапожок) - закрашиваем прямоугольники
set object 1 rect from -1,-1 to 1,0 fc rgb "lightblue"
set object 2 rect from -1,0 to 0,1 fc rgb "lightblue"
set object 3 rect from 0,-1 to 1,0 fc rgb "lightblue"

# Вырезанная часть (правый верхний квадрант)
set object 4 rect from 0,0 to 1,1 fc rgb "white"

# Границы
set arrow from -1,-1 to 1,-1 nohead lw 2 lc rgb "black"
set arrow from 1,-1 to 1,0 nohead lw 2 lc rgb "black"
set arrow from 1,0 to 0,0 nohead lw 2 lc rgb "black"
set arrow from 0,0 to 0,1 nohead lw 2 lc rgb "black"
set arrow from 0,1 to -1,1 nohead lw 2 lc rgb "black"
set arrow from -1,1 to -1,-1 nohead lw 2 lc rgb "black"

# Метки
set label "Область D" at -0.5, -0.5 center font ",16"
set label "Вырезано" at 0.5, 0.5 center font ",12"

plot NaN notitle
EOF

gnuplot /tmp/domain.gnuplot 2>/dev/null && echo "✓ Изображение области создано: results/plot_domain.png" || echo "gnuplot не установлен"
