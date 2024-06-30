// cuda headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// created files
#include "cuda_common.cuh" 

// sum the arrays on the GPU
__global__ void inclusiveScan(int* input, int* output, int size)
{
    output[0] = input[0];
    for (int i = 0; i < size; i++)
    {
        output[i] = output[i - 1] + input[i];
    }


}

int main()
{}