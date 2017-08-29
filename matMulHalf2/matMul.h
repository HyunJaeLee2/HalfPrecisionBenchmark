#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#define BLOCK_SIZE 16
//#define FP16

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

typedef struct {
    int width;
    int height;
    __half* half_elements;
} Matrix_half;

typedef struct {
    int width;
    int height;
    int width_len;
    __half2* half2_elements;
} Matrix_half2;


