#ifndef HALF_H
#define HALF_H
#include "matMul.h"

__global__ void MatMulKernel_half(Matrix_half A, Matrix_half B, Matrix_half C);
__global__ void float2half_rn_kernel(int size, const float *buffIn, __half *buffOut);
__global__ void half2float_kernel(int size, const __half *buffIn, float *buffOut);
void MatMul_half(const Matrix A, const Matrix B, Matrix C);
void gpu_half2float(int size, const __half *buffIn, float *buffOut);
void gpu_float2half_rn(int size, const float *buffIn, __half *buffOut);

#endif
