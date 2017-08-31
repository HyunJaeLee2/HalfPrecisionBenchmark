#ifndef CUDNNTEST_H
#define CUDNNTEST_H

#define BLOCK_SIZE 32

#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "layer.h"


void free_layer(conv_layer *layer);
conv_layer initFirstLayerWithRandom(char *argv[]);

#endif
