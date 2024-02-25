#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>


// Print data disregarding data size
__global__ void printMemTransfer(int* data)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("tid : %d, gid : %d| value : %d\n", threadIdx.x, gid, data[gid]);
}

// Print data within limits of data size
__global__ void printMemTransfer_s(int* data, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < size)
    printf("tid : %d, gid : %d| value : %d\n", threadIdx.x, gid, data[gid]);
}

int main()
{
    int size = 128;  // 50
    int byte_size = sizeof(int) * size;

    int *h_input;
    h_input = (int*)malloc(byte_size);


    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) 
    {
        h_input[i] = (int)(rand() % 0xff);
    }

    int *d_input;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // 32 threads in x * 5 grids in x = 128 total threads
    dim3 block(64); // 32
    dim3 grid(2);   // 5

    //printMemTransfer << < grid, block>> > (d_input);
    printMemTransfer_s << < grid, block >> > (d_input, size);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    free(h_input);
    return 0;
}