Для запуска программ на вычислительном комплексе IBM Polus использовался планировщик IBM Spectrum LSF.

Таким образом все программы после их компиляции (Makefile с вызовом make all), запускались через команду bsub < *.lsf .

Для получения итоговых результатов использовались файлы:

- OpenMP_job.lsf с последовательным изменениями в нём

- MPI_classic.lsf для сравнения на маленьких сетках 

- MPI_scaling_test.lsf для получения графиков ускорений

- MPI_OpenMP_classic.lsf и MPI_OpenMP_scaling_test.lsf

- Для вычисления на большой сетке для пункта с CUDA использовались - final_sequential.lsf , final_openmp.lsf , final_mpi.lsf , MPI_OpenMP_scaling_test.lsf , run_mpi_cuda.lsf

- Запуске на CUDA происходили с помощью скрипта run_mpi_cuda.lsf
