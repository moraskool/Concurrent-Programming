#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// Print unique id for 1D grid
__global__ void printUniqueGid(int * data)
{
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    int row_offset = blockDim.x * gridDim.x * blockIdx.y;
    int gid = block_offset + row_offset + tid;

    printf("blockIdx.x : %d, blockIdx.y : %d, threadIdx.x : %d, gid : %d| data : %d\n",
        blockIdx.x, blockIdx.y, tid, gid, data[gid]);
}


// Print unique id for 2D grid
__global__ void printUniqueGid_2d(int* data)
{
    // thread index
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // block offset 
    int num_thread_in_a_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_thread_in_a_block;

    // row offset
    int num_thread_in_a_row = num_thread_in_a_block * gridDim.x;
    int row_offset = num_thread_in_a_row * blockIdx.y;

    // global thread index
    int gid = block_offset + row_offset + tid;

    printf("blockIdx.x : %d, blockIdx.y : %d, threadIdx.x : %d, gid : %d| data : %d\n",
        blockIdx.x, blockIdx.y, tid, gid, data[gid]);

}

int main()
{
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    int h_data[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 120, 140, 150, 160 };

    int *d_data;
    cudaMalloc((void**)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(2,2);  // 2 thread blocks in x, y
    dim3 grid(2,2);

    printUniqueGid << <grid, block >> > (d_data);

    printUniqueGid_2d << <grid, block >> > (d_data);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}