srun -p gpunode -N 1 -A ipetrushin  /usr/local/cuda-11.1/bin/nvcc  kernel.cu -lcurand -o kernel
