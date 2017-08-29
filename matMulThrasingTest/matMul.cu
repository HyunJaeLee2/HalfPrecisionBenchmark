#include "matMul.h"
#include <time.h>

long timer_get(void)
{
    struct timespec time;
    if ( 0 == clock_gettime( CLOCK_REALTIME, &time)) {
        //printf( "tv_sec: %ld\ntv_nsec: %ld", time.tv_sec, time.tv_nsec);
    } else {
        fprintf( stderr, "Something wrong on clock_gettime()\n");
    }   
    //return  time.tv_sec+time.tv_nsec*1e-9;
    //return  (time.tv_nsec + time.tv_sec*1000000000)/1000;
    return  (time.tv_nsec)/1000 + time.tv_sec*1000000;
    //return (k1_io_read64(0x70084040)/MPPA_FREQUENCY);
}

__global__ void MatMulKernel_half2(Matrix_half2 A, Matrix_half2 B, Matrix_half2 C) {
    __half2 Cvalue = __floats2half2_rn(0.0, 0.0);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= A.height || col >= B.width_len) return;

    for (int e = 0; e < A.width_len; ++e)
        Cvalue = __hadd2( Cvalue, 
                        __hmul2( (A.half2_elements[row * A.width_len + e]), 
                                 (B.half2_elements[e * B.width_len + col]) ) 
                         );

    C.half2_elements[row * C.width_len + col] = Cvalue;
}
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    float Cvalue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= A.height || col >= B.width) return;

    for (int e = 0; e < A.width; ++e)
        Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);

    C.elements[row * C.width + col] = Cvalue;
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

__global__ void float2DupHalf2_rn_kernel(int size, const float *buffIn, __half2 *buffOut)
{
    const int idx = (BLOCK_SIZE*blockIdx.x+threadIdx.x);
    if (idx >= size/sizeof(__half2)) return;
	
    __half2 val;
    val = __floats2half2_rn(float(buffIn[idx]), float(buffIn[idx]));
    buffOut[idx] = val;
}

__global__ void float2half2_rn_kernel(int size, const float *buffIn, __half2 *buffOut)
{
    const int idx = (BLOCK_SIZE *blockIdx.x / 2 + threadIdx.x) * 2; // divide by 2 because dimBlock is BlockSIZE / 2
    if (idx >= size/sizeof(__half2)) return;
	
    __half2 val;
    val = __floats2half2_rn(float(buffIn[idx]), float(buffIn[idx+1]));
    buffOut[idx / 2] = val;
}

void gpu_float2DupHalf2_rn(int size, const float *buffIn, __half2 *buffOut)
{
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float2DupHalf2_rn_kernel<<<grid_size, BLOCK_SIZE>>> (size, buffIn, buffOut);
    cudaDeviceSynchronize();
}


//size : length of buffIn
void gpu_float2half2_rn(int size, const float *buffIn, __half2 *buffOut)
{
    int dimBlock = BLOCK_SIZE / 2;
    int grid_size = (size + dimBlock - 1) / dimBlock;
    float2half2_rn_kernel<<<grid_size, dimBlock>>> (size, buffIn, buffOut);
    cudaDeviceSynchronize();
}

__global__ void half22float_kernel(int size, const __half2 *buffIn, float *buffOut)
{
    const int idx = BLOCK_SIZE*blockIdx.x+threadIdx.x;
    if (idx >= size / sizeof(__half2)) return;
   
    /*
    float2 val;
    val = __half22float2((buffIn[idx]));
    buffOut[idx * 2] = val.x;
    buffOut[idx * 2 + 1] = val.y;
     */
    float hi_float;
    float lo_float;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, low;}\n" : "=f"(lo_float) : "r"(buffIn[idx].x));

    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, high;}\n" : "=f"(hi_float) : "r"(buffIn[idx].x));
    buffOut[idx * 2] = lo_float;
    buffOut[idx * 2 + 1] = hi_float;
}

//size : length of buffIn
void gpu_half22float(int size, const __half2 *buffIn, float *buffOut)
{
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    half22float_kernel<<<grid_size, BLOCK_SIZE>>> (size, buffIn, buffOut);
    cudaDeviceSynchronize();
}

void MatMul_half2(const Matrix A, const Matrix B, Matrix C) {
    //allocate temp mem to copy float
    Matrix d_tempA;
    d_tempA.width = A.width;
    d_tempA.height = A.height;
    int size_temp = A.width * A.height * sizeof(float);

    cudaError_t err = cudaMalloc(&d_tempA.elements, size_temp);
    //printf("CUDA malloc tempA: %s\n",cudaGetErrorString(err));

    err = cudaMemcpy(d_tempA.elements, A.elements, size_temp, cudaMemcpyHostToDevice);
    //printf("Copy A to device: %s\n",cudaGetErrorString(err));
   
    //allocate half2 mem
    Matrix_half2 d_A;
    d_A.width = A.width;
    d_A.width_len = A.width;
    d_A.height = A.height;
    int size = d_A.width_len * d_A.height * sizeof(__half2);
    
    err = cudaMalloc(&d_A.half2_elements, size);
    //printf("CUDA malloc A: %s (%d)\n",cudaGetErrorString(err), size);
    
    gpu_float2DupHalf2_rn(size, d_tempA.elements, d_A.half2_elements);
    //gpu_float2half2_rn(size, d_tempA.elements, d_A.half2_elements);

    //allocate temp mem to copy float
    Matrix d_tempB;
    d_tempB.width = B.width;
    d_tempB.height = B.height;
    size_temp = B.width * B.height * sizeof(float);

    err = cudaMalloc(&d_tempB.elements, size_temp);
    //printf("CUDA malloc tempB: %s (%d)\n",cudaGetErrorString(err), size_temp);

    err = cudaMemcpy(d_tempB.elements, B.elements, size_temp, cudaMemcpyHostToDevice);
    //printf("Copy B to device: %s\n",cudaGetErrorString(err));
    
    //allocate half2 mem
    Matrix_half2 d_B;
    d_B.width = B.width;
    d_B.width_len = B.width;
    d_B.height = B.height;
    size = d_B.width_len * d_B.height * sizeof(__half2);

    err = cudaMalloc(&d_B.half2_elements, size);
    //printf("CUDA malloc B: %s (%d)\n",cudaGetErrorString(err), size);

    gpu_float2DupHalf2_rn(size, d_tempB.elements, d_B.half2_elements);
    
    // Allocate C in device memory
    Matrix_half2 d_C;
    d_C.width = C.width;
    d_C.width_len = C.width;
    d_C.height = C.height;
    size = d_C.width_len * d_C.height * sizeof(__half2);

    err = cudaMalloc(&d_C.half2_elements, size);
    //printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

    cudaFree(d_tempA.elements);
    cudaFree(d_tempB.elements);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_B.width_len + dimBlock.x - 1) / dimBlock.x,
            (A.height + dimBlock.y - 1) / dimBlock.y);

    long t = timer_get();
    MatMulKernel_half2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    err = cudaDeviceSynchronize();
    fprintf(stderr,   "[half2  ]\t%9ld\n", timer_get() - t);
    
    //printf("Run kernel: %s\n", cudaGetErrorString(err));
    
    //allocate temp mem to copy half2
    Matrix d_tempC;
    d_tempC.width = C.width;
    d_tempC.height = C.height;
    size_temp = C.width * C.height * sizeof(float); 

    err = cudaMalloc(&d_tempC.elements, size_temp);
    //printf("CUDA malloc tempC: %s\n",cudaGetErrorString(err));

    // Read C from device memory
    gpu_half22float(size, d_C.half2_elements, d_tempC.elements);

    err = cudaMemcpy(C.elements, d_tempC.elements, size_temp, cudaMemcpyDeviceToHost);
    //printf("Copy C off of device: %s\n",cudaGetErrorString(err));

    // Free device memory
    cudaFree(d_A.half2_elements);
    cudaFree(d_B.half2_elements);
    cudaFree(d_C.half2_elements);
    cudaFree(d_tempC.elements);
}

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

int main(int argc, char* argv[]){
    Matrix A, B, C;
    int a1, a2, b1, b2;

    a1 = atoi(argv[1]); 
    a2 = atoi(argv[2]);

    b1 = a2;
    b2 = atoi(argv[3]); 
    //a1 = a2 = b1= b2 = atoi(argv[1]);

    A.height = a1;
    A.width = a2;

    A.elements = (float*)malloc(A.width * A.height * sizeof(float));

    B.height = b1;
    B.width = b2;
    B.elements = (float*)malloc(B.width * B.height * sizeof(float));

    C.height = A.height;
    C.width = B.width;
    C.elements = (float*)malloc(C.width * C.height * sizeof(float));

    for(int i = 0; i < A.height; i++)
        for(int j = 0; j < A.width; j++)
            A.elements[i*A.width + j] = (float)(random() % 7);

    for(int i = 0; i < B.height; i++)
        for(int j = 0; j < B.width; j++)
            B.elements[i*B.width + j] = (float)(random() % 7);
    MatMul(A,B,C);
    MatMul_half(A, B, C);
    MatMul_half2(A, B, C);
/*
    if(strcmp(argv[4], "normal") == 0)
        MatMul(A,B,C);
    else if(strcmp(argv[4], "half") == 0)
        MatMul_half(A, B, C);
    else if(strcmp(argv[4], "half2") == 0)
        MatMul_half2(A, B, C);
    else{
        fprintf(stderr, "not supported\n");
        return 0;
    }
  */
    /*
    // Print up to a 10x10 portion of the three matrices
    for(int i = 0; i < min(32, A.height); i++){
        for(int j = 0; j < min(32, A.width); j++)
            printf("%d ", (int)A.elements[i*A.width + j]);
        printf("\n");
    }
    printf("\n");

    for(int i = 0; i < min(32, B.height); i++){
        for(int j = 0; j < min(32, B.width); j++)
            printf("%d ", (int)B.elements[i*B.width + j]);
        printf("\n");
    }
    printf("\n");

    for(int i = 0; i < min(32, C.height); i++){
        for(int j = 0; j < min(32, C.width); j++)
            printf("%d ", (int)C.elements[i*C.width + j]);
        printf("\n");
    }
    printf("\n");
    */
}
