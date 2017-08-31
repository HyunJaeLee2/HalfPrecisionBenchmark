#include "cuDNNTest.h"

void checkCUDA(cudaError_t error);
void checkCUDNN(cudnnStatus_t status);
__global__ void float2half_rn_kernel(int size, const float *buffIn, __half *buffOut);
void gpu_float2half_rn(int size, const float *buffIn, __half *buffOut);
__global__ void half2float_kernel(int size, const __half *buffIn, float *buffOut);
void gpu_half2float(int size, const __half *buffIn, float *buffOut);
