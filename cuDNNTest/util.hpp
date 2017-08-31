#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include "cuDNNTest.h"

void printCUDAMem();
void print4D(const char* title, float* src, int num, int channel, int height, int width);
int convertToIndex(int n, int batch_count, int k, int in_channel, int h, int in_height, int w, int in_width) ;
void initWithRandom4D(float* src, int num, int channel, int height, int width);
long timer_get(void);
