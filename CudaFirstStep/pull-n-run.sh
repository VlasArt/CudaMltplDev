rm kernel
/usr/local/cuda-11.1/bin/nvcc  kernel.cu -lcurand -o kernel
./kernel > output-`date +%s`.txt
