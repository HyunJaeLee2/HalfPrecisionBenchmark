#ifndef FLOAT_H 
#define FLOAT_H
#include "matMul.h"

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);
void MatMul(const Matrix A, const Matrix B, Matrix C);


#endif
