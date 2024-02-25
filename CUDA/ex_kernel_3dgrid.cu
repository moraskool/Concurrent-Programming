#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

// Print data within limits of data size
__global__ void print3DGridArray(int* data, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int row_offset = blockDim.x * gridDim.x * blockIdx.y;
    int gid = block_offset + row_offset + tid;

    if (gid < size)
        printf("tid : %d, gid : %d| value : %d\n", threadIdx.x, gid, data[gid]);
}

int main()
{
    int size = 64;
    int byte_size = sizeof(int) * size;

    int* h_input;
    h_input = (int*)malloc(byte_size);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        h_input[i] = (int)(rand() % 0xff);
    }

    int* d_input;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // 2 threads each in blocks x,y,z * 4 grids in x,y,z = 128 total threads
    dim3 block(2,2,2);
    dim3 grid(4,4,4);

    print3DGridArray << < grid, block >> > (d_input, size);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    free(h_input);
    return 0;
}