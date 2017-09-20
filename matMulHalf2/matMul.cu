#include "matMul.h"
#include "float.h"
#include "half.h"
#include "half2.h"
#include <time.h>

#define PRINT_NUM 8
long timer_get(void)
{
    struct timespec time;
    if ( 0 == clock_gettime( CLOCK_REALTIME, &time)) {
        //printf( "tv_sec: %ld\ntv_nsec: %ld", time.tv_sec, time.tv_nsec);
    } else {
        fprintf( stderr, "Something wrong on clock_gettime()\n");
    }   
    return  (time.tv_nsec)/1000 + time.tv_sec*1000000;
}

int main(int argc, char* argv[]){
    Matrix A, B, C;
    int a1, a2, b1, b2;

    if(argc < 5)
    {   
        fprintf(stderr, "usage : ./test [Height of former matrix] [Width of former matrix] [Width of latter matrix] [float/half/half2]\n");
        return 0;
    }

    a1 = atoi(argv[1]); 
    a2 = atoi(argv[2]);

    b1 = a2;
    b2 = atoi(argv[3]); 

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
    
    if(strcmp(argv[4], "float") == 0)
        MatMul(A,B,C);
    else if(strcmp(argv[4], "half") == 0)
        MatMul_half(A, B, C);
    else if(strcmp(argv[4], "half2") == 0)
        MatMul_half2(A, B, C);
    else{
        fprintf(stderr, "not supported\n");
        return 0;
    }

#ifdef DEBUG
    for(int i = 0; i < min(PRINT_NUM, A.height); i++){
        for(int j = 0; j < min(PRINT_NUM, A.width); j++)
            printf("%d ", (int)A.elements[i*A.width + j]);
        printf("\n");
    }
    printf("\n");

    for(int i = 0; i < min(PRINT_NUM, B.height); i++){
        for(int j = 0; j < min(PRINT_NUM, B.width); j++)
            printf("%d ", (int)B.elements[i*B.width + j]);
        printf("\n");
    }
    printf("\n");

    for(int i = 0; i < min(PRINT_NUM, C.height); i++){
        for(int j = 0; j < min(PRINT_NUM, C.width); j++)
            printf("%d ", (int)C.elements[i*C.width + j]);
        printf("\n");
    }
    printf("\n");
#endif

    return 0;
}
