#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include "util.hpp"

using namespace std;

void printCUDAMem()
{
    size_t free, total;
    cudaMemGetInfo(&free,&total); 
    printf("%ld KB free of total %ld KB\n",free/1024,total/1024);
}

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

void print4D(const char* title, float* src, int num, int channel, int height, int width)
{
    cout << title << endl;
    for (int n = 0; n < min(2, num); n++) {
        for (int i = 0; i < min(4, channel); i++) {
            for (int y = 0; y < min(4, height); y++) {
                for (int x = 0; x < min(4, width); x++) {
                    printf("%.4f ", src[convertToIndex(n, num, i, channel, y, height, x, width)]);
                }   
                cout << endl;
            }   
            cout << endl;
        }
    } 
}

void initWithRandom4D(float* src, int num, int channel, int height, int width)
{
    for (int n = 0; n < num; n++) {
        for (int i = 0; i < channel; i++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    src[convertToIndex(n, num, i, channel, y, height, x, width)] = 
                            static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 10);
                }   
            }   
        }
    } 
}
int convertToIndex(int n, int batch_count, int k, int in_channel, int h, int in_height, int w, int in_width) 
{
    return ((n * in_channel + k) * in_height + h) * in_width + w;
}
