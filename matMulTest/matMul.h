#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#define BLOCK_SIZE 32
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
