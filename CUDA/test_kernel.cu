#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printHello()
{
    printf("Hello CUDA! First program. \n");
}

__global__ void printThreadIdx()
{
    printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d \n",  
        threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void printUniqueId()
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;
    printf("gid = %d \n", gid);
}

//int main()
//{
//
//    /* 
//    * ------ explanation 1 -----------
//      kernel << <nb, tb >> >(arguments)
//      nb = block , number of threads in each dim of a block
//      tb = grid ,  number of thread blocks in each dim
//      #threads = grid * block 
//      # 4 threads each in 8 grids in x dimension
//    */
//    dim3 block_1(4); 
//    dim3 grid_1(8);
//    printHello <<< grid_1, block_1>> > ();  // 4*8 = 32 threads
//
//    /*
//    * ------ explanation 2 -----------
//      kernel << <nb, tb >> >(arguments)
//      nb = 8,8
//      tb = 16/8 * 16/8
//      #threads = grid * block
//    */
//
//    int nx, ny;
//    nx = 16;
//    ny = 16;
//    dim3 block_2(8, 8);
//    dim3 grid_2(nx/block_2.x, ny/block_2.y);
//    printThreadIdx << < grid_2, block_2 >> > ();  // 32 threads
//
//    printUniqueId << <grid_1,block_1 >> >() ;
//
//    cudaDeviceSynchronize();
//
//    cudaDeviceReset();
//
//    return 0;
//}