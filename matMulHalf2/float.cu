#include "float.h"

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    float Cvalue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= A.height || col >= B.width) return;

    for (int e = 0; e < A.width; ++e)
        Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);

    C.elements[row * C.width + col] = Cvalue;
}

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    int size = A.width * A.height * sizeof(float);

    cudaError_t err = cudaMalloc(&d_A.elements, size);
    //printf("CUDA malloc A: %s\n",cudaGetErrorString(err));

    err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    //printf("Copy A to device: %s\n",cudaGetErrorString(err));

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);

    err = cudaMalloc(&d_B.elements, size);
    //printf("CUDA malloc B: %s\n",cudaGetErrorString(err));

    err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    //printf("Copy B to device: %s\n",cudaGetErrorString(err));

    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);

    err = cudaMalloc(&d_C.elements, size);
    //printf("CUDA malloc C: %s\n",cudaGetErrorString(err));


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x,
            (A.height + dimBlock.y - 1) / dimBlock.y);

    long t = timer_get();
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    err = cudaDeviceSynchronize();
    fprintf(stderr,   "[normal  ]\t%9ld\n", timer_get() - t);

    //printf("Run kernel: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    //printf("Copy C off of device: %s\n",cudaGetErrorString(err));

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    // cudaFree(d_C.elements);
}

