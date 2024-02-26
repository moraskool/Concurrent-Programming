
// cuda headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <stdlib.h> // for random init usage
#include <time.h>
#include <cstring>  // memset usage

// created files
#include "cuda_common.cuh" 

// sum the arrays on the GPU
__global__ void sumArrayGPU(int* a, int* b, int* c, int size)
{
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int row_offset = blockDim.x * gridDim.x * blockIdx.y;
    int gid = block_offset + row_offset + tid;

    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
        
}

// sum the arrays on the CPU
void sum_array_cpu(int* a, int* b, int* c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }

}

// compare arrays (move to common.h)
void compare_arrays(int* a, int* b, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            printf("Arrays are different \n");
        }
    }

    printf("Arrays are the same\n");

}


int main()
{
    int size = 1 << 25;
    int block_size = 1024;

    cudaError error;

    int byte_size = sizeof(int) * size;

    // host pointers
    int* h_a, *h_b, *gpu_result, *cpu_result;

    // allocate memory
    h_a = (int*)malloc(byte_size);
    h_b = (int*)malloc(byte_size);
    gpu_result = (int*)malloc(byte_size);
    cpu_result = (int*)malloc(byte_size);

    // initialize random data to host input
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; i++)
    {
        h_a[i] = (int)(rand() % 0xff);
    }

    for (int i = 0; i < size; i++)
    {
        h_b[i] = (int)(rand() % 0xff);
    }

    memset(gpu_result, 0, byte_size);
    memset(cpu_result, 0, byte_size);

    // CPU computation
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, cpu_result, size);
    cpu_end = clock();

    // device pointers
    int* d_a, *d_b, *d_c;
    gpuErrchk(cudaMalloc((void**)&d_a, byte_size));
    gpuErrchk(cudaMalloc((void**)&d_b, byte_size));
    gpuErrchk(cudaMalloc((void**)&d_c, byte_size));
   
    // memory transfer from host to device
    clock_t mem_htod_start, mem_htod_end;
    mem_htod_start = clock();
    gpuErrchk(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
    mem_htod_end = clock();

    // 1024 threads each in blocks x * (10^25 / 1024)+1 grids in x = 128 total threads
    dim3 block(block_size);
    dim3 grid(size/block.x + 1);
    
    // launching the kernel
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sumArrayGPU << < grid, block >> > (d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    gpu_end = clock();

    // memory transfer back to host
    clock_t mem_dtoh_start, mem_dtoh_end;
    mem_dtoh_start = clock();
    gpuErrchk(cudaMemcpy(gpu_result, d_c, byte_size, cudaMemcpyDeviceToHost));
    mem_dtoh_end = clock();

    // CPU == GPU ?
    compare_arrays(gpu_result, cpu_result, size);
    
    // free device resources
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free host resources
    free(gpu_result);
    free(cpu_result);
    free(h_b);
    free(h_a);

    return 0;
}