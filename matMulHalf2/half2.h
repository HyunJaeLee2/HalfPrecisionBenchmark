#ifndef HALF2_H
#define HALF2_H
#include "matMul.h"

__global__ void MatMulKernel_half2(Matrix_half2 A, Matrix_half2 B, Matrix_half2 C);
__global__ void float2DupHalf2_rn_kernel(int size, const float *buffIn, __half2 *buffOut);
__global__ void float2half2_rn_kernel(int size, const float *buffIn, __half2 *buffOut);
__global__ void half22float_kernel(int size, const __half2 *buffIn, float *buffOut);
void gpu_float2DupHalf2_rn(int size, const float *buffIn, __half2 *buffOut);
void gpu_float2half2_rn(int size, const float *buffIn, __half2 *buffOut);
void gpu_half22float(int size, const __half2 *buffIn, float *buffOut);
void MatMul_half2(const Matrix A, const Matrix B, Matrix C);


#endif
