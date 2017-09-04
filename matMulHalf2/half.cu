#include "half.h"

void MatMul_half(const Matrix A, const Matrix B, Matrix C) {
    //allocate temp mem to copy float
    Matrix d_tempA;
    d_tempA.width = A.width;
    d_tempA.height = A.height;
    int size_temp = A.width * A.height * sizeof(float);

    cudaError_t err = cudaMalloc(&d_tempA.elements, size_temp);
    //printf("CUDA malloc tempA: %s\n",cudaGetErrorString(err));

    err = cudaMemcpy(d_tempA.elements, A.elements, size_temp, cudaMemcpyHostToDevice);
    //printf("Copy A to device: %s\n",cudaGetErrorString(err));
   
    //allocate half mem
    Matrix_half d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    int size = A.width * A.height * sizeof(__half);
    
    err = cudaMalloc(&d_A.half_elements, size);
    //printf("CUDA malloc A: %s (%d)\n",cudaGetErrorString(err), size);
    
    gpu_float2half_rn(size, d_tempA.elements, d_A.half_elements);

    //allocate temp mem to copy float
    Matrix d_tempB;
    d_tempB.width = B.width;
    d_tempB.height = B.height;
    size_temp = B.width * B.height * sizeof(float);

    err = cudaMalloc(&d_tempB.elements, size_temp);
    //printf("CUDA malloc tempB: %s (%d)\n",cudaGetErrorString(err), size_temp);

    err = cudaMemcpy(d_tempB.elements, B.elements, size_temp, cudaMemcpyHostToDevice);
    //printf("Copy B to device: %s\n",cudaGetErrorString(err));
    
    //allocate half mem
    Matrix_half d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(__half);

    err = cudaMalloc(&d_B.half_elements, size);
    //printf("CUDA malloc B: %s (%d)\n",cudaGetErrorString(err), size);

    gpu_float2half_rn(size, d_tempB.elements, d_B.half_elements);
    
    
    // Allocate C in device memory
    Matrix_half d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(__half);

    err = cudaMalloc(&d_C.half_elements, size);
    //printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

    cudaFree(d_tempA.elements);
    cudaFree(d_tempB.elements);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x,
            (A.height + dimBlock.y - 1) / dimBlock.y);

    long t = timer_get();
    MatMulKernel_half<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    err = cudaDeviceSynchronize();
    fprintf(stderr,   "[half  ]\t%9ld\n", timer_get() - t);
    
    //printf("Run kernel: %s\n", cudaGetErrorString(err));
    
    //allocate temp mem to copy half
    Matrix d_tempC;
    d_tempC.width = C.width;
    d_tempC.height = C.height;
    size_temp = C.width * C.height * sizeof(float); 

    err = cudaMalloc(&d_tempC.elements, size_temp);
    //printf("CUDA malloc tempC: %s\n",cudaGetErrorString(err));

    // Read C from device memory
    gpu_half2float(size, d_C.half_elements, d_tempC.elements);

    err = cudaMemcpy(C.elements, d_tempC.elements, size_temp, cudaMemcpyDeviceToHost);
    //printf("Copy C off of device: %s\n",cudaGetErrorString(err));

    // Free device memory
    cudaFree(d_A.half_elements);
    cudaFree(d_B.half_elements);
    cudaFree(d_C.half_elements);
    cudaFree(d_tempC.elements);
}

__global__ void MatMulKernel_half(Matrix_half A, Matrix_half B, Matrix_half C) {
    __half Cvalue = __float2half(0.0);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= A.height || col >= B.width) return;

    for (int e = 0; e < A.width; ++e)
        Cvalue = __hadd( Cvalue, 
                        __hmul( (A.half_elements[row * A.width + e]), 
                                 (B.half_elements[e * B.width + col]) ) 
                         );

    C.half_elements[row * C.width + col] = Cvalue;
}

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
    //checkCudaErrors(cudaDeviceSynchronize());
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
    //checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceSynchronize();
}
