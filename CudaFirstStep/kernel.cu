
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <curand.h>

bool stream_init(cudaStream_t* stream)
{
    int* num = new int;
    cudaGetDeviceCount(num);

    try {
        for (int i = 0;i < *num;i++) {
            cudaStreamCreate(&stream[i]);
        }
    }
    catch (...) {
        return false;
    }

    delete num;
    return true;
}

bool stream_dispose(cudaStream_t* stream)
{
    int* num = new int;
    cudaGetDeviceCount(num);

    try {
        for (int i = 0;i < *num;i++) {
            cudaStreamDestroy(stream[i]);
        }
    }
    catch (...) {
        return false;
    }

    delete num;
    return true;
}

bool generateRandArray(float* numArray, int arraySize)
{
    try {
        float* dev_a;
        srand(time(NULL));

        cudaSetDevice(0);
        cudaMalloc((void**)&dev_a, arraySize * sizeof(int));

        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, dev_a, arraySize);
        cudaMemcpy(numArray, dev_a, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        curandDestroyGenerator(gen);
    }
    catch (...) {
        return false;
    }
    return true;
}

bool askAboutMemory(int arraySize)
{
    char ch;
    
    std::cout << "\nThis programm need more then " << arraySize * sizeof(float) / 1024. / 1024 / 1024 * 2 << " GB RAM. Continue?\n(Y/N)_";
    std::cin >> ch;
    std::cout << std::endl;

    if (ch != 'y' && ch != 'Y')
        return false;

    return true;
}

void lazzzyArrayPrint(float* arr, int arrSize)
{
    std::cout << '\n' << "Array len: " << arrSize << '\n' << "Array items:" << '\n' << std::endl;
    std::cout << arr[0]<< " " << arr[1] << " " << arr[2] << std::endl; 
    std::cout <<  " ... " << std::endl;
    std::cout << arr[arrSize / 2 - 1] << " " << arr[arrSize/2] << " " << arr[arrSize / 2 + 1] << std::endl;
    std::cout << " ... " << std::endl;
    std::cout << arr[arrSize - 3] << " " << arr[arrSize - 2] << " " << arr[arrSize - 1] << '\n' << std::endl;
}

__global__ void _kernel(float* a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int d;
    
      
    cudaGetDevice(&d);
    printf("Kernel %d working\n", d);
    cudaDeviceGetAttribute(&d, cudaDevAttrPciBusId , d);
    a[i] = a[i] * a[i] + d;
}

int main()
{
    // Разминочный блок. Определяет количество гпу и выводит их характеристики    
    int num;
    cudaGetDeviceCount(&num);
    std::cout << "Detcted device count: " << num << '\n' << std::endl;

    for (int i = 0;i < num;i++) {
        // Query the device properties.
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device id: " << i << std::endl;
        std::cout << "Device name: " << prop.name << std::endl;
    }

    // Массив потоков. Каждый поток - это отдельная видеокарта.
    // Порядковый номер в массиве - id карты
    cudaStream_t *stream = new cudaStream_t[num];

    // Иниацилизация потоков по одному потоку на каждое устройство
    if (!stream_init(stream))
        return 1;

    const int arraySize = 2;// 1024 * 1024 * 512;   
    const int sizePerSt = arraySize / num;

    /*if (!askAboutMemory(arraySize))
        return 2;*/

    float *a = new float[arraySize];
    float** dev_a;
    if (arraySize % num == 0)
    {
        // Предпологаем что количество эл-тов кратно количеству гпу
        // Проверить что сработает создавать массив из массивов для разных ГПУ
        dev_a = new float*[num];
        for (int j = 0; j < num; j++)
            dev_a[j] = new float[arraySize];
    }
    /*else
    {
        dev_a = new float* [num];
        for (int j = 0; j < num; j++)
            dev_a[j] = new float[sizePerSt];
    }   */

    if (!generateRandArray(a, arraySize))
        return 1;

    lazzzyArrayPrint(a, arraySize);

    //Подготовка к запуску ядра
    dim3 threads = dim3(2);
    dim3 blocks = dim3(1);

    // Подготовка и передач данных на карты
    try {
        for (int i = 0; i < num; i++)
        {
            cudaSetDevice(i);
            cudaMalloc((void**)&dev_a[i], arraySize * sizeof(float));
            cudaMemcpyAsync(dev_a[i], a, arraySize * sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    catch (...) {
        return 3;
    }

    // Запуск ядра
    try {
        for (int i = 0; i < num; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
            int d;
            cudaGetDevice(&d);
            cudaDeviceGetAttribute(&d, cudaDevAttrPciDeviceId, d);

            printf("Kernel %d started\n", d);
            _kernel <<<blocks, threads, 0, stream[i]>>> (dev_a[i]);
            //printf("Kernel stoped\n");
        }
    }
    catch (...) {
        return 3;
    }

    // получение данных обратно
    try {
        for (int i = 0; i < num; i++)
        {
            delete a;
            cudaSetDevice(i);            
            cudaMemcpy(a, dev_a[i], arraySize * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Theoretecly data from " << i << " device." << std::endl;
            lazzzyArrayPrint(a, arraySize);
        }
    }
    catch (...) {
        return 3;
    }

    cudaFree(dev_a);

    //lazzzyArrayPrint(a, arraySize);

    // Убийство всех потоков
    if (!stream_dispose(stream))
        return 1;
    
    return 0;
}