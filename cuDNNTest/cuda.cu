#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include "cuda.h"
__global__ void float2half_rn_kernel(int size, const float *buffIn, __half *buffOut)
{   
    const int idx = BLOCK_SIZE*blockIdx.x+threadIdx.x;
    if (idx >= size/sizeof(__half)) return;
    __half val;
    val.x = __float2half_rn(float(buffIn[idx]));
    buffOut[idx] = val;
}

void gpu_float2half_rn(int size, const float *buffIn, __half *buffOut)
{   
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float2half_rn_kernel<<<grid_size, BLOCK_SIZE>>> (size, buffIn, buffOut);
    cudaDeviceSynchronize();
}

__global__ void half2float_kernel(int size, const __half *buffIn, float *buffOut)
{   
    const int idx = BLOCK_SIZE*blockIdx.x+threadIdx.x;
    if (idx >= size/sizeof(__half)) return;
    float val;
    val = __half2float((buffIn[idx]));
    buffOut[idx] = val;
}

void gpu_half2float(int size, const __half *buffIn, float *buffOut)
{
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    half2float_kernel<<<grid_size, BLOCK_SIZE>>> (size, buffIn, buffOut);
    cudaDeviceSynchronize();
}

