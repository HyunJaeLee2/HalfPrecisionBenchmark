#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

using namespace std;

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

void checkCUDNN(cudnnStatus_t status)
{
	if (status != CUDNN_STATUS_SUCCESS)
		cout << "[ERROR] CUDNN " << status << endl;
}

void checkCUDA(cudaError_t error)
{
	if (error != CUDA_SUCCESS)
		cout << "[ERROR] CUDA " << error << endl;
}

void print(const char* title, float* src, int filter_num, int h, int w)
{
	cout << title << endl;
	for (int i = 0; i < min(4, filter_num); i++) {
		for (int y = 0; y < min(16, h); y++) {
			for (int x = 0; x < min(16, w); x++) {
				printf("%.4f ", src[i*h * w + y * w + x]);
			}
			cout << endl;
		}
		cout << endl;
	}
}

int convertToIndex(int n, int batch_count, int k, int in_channel, int h, int in_height, int w, int in_width) 
{
    return ((n * in_channel + k) * in_height + h) * in_width + w;
}

int main(int argc, char* argv[])
{
	const int batch_count = 1;//입력 데이터 갯수, 배치사이즈
	const int padding_w = 1;//컨볼루션 패딩. 필터의 가로 세로 길이가 3이고 패딩이 1,1 이면 SAME Convolution이 된다
	const int padding_h = 1;
	const int stride_horizontal = 1;//컨볼루션 스트라이드
	const int stride_vertical = 1;
	int in_channel, in_height, in_width,filter_width, filter_height,filter_num, src_len;
	float *inData_d;//device 입력 데이터
	float *outData_d, *outData1_d;//device 출력 데이터
	float *filterData_d;//device 컨볼루션 필터 데이터
	void* workSpace;//CUDNN이 작업 중에 사용할 버퍼 메모리
    long time_accum = 0;
    size_t free, total;
    //float *inData_NCHW, *outData, *filterData;
    if(argc < 4)
    {
        fprintf(stderr, "usage : ./test [Input Height/Width] [Channel] [Filter Height/Width] [Filter Num]\n");
        return 0;
    }
   
    in_height = atoi(argv[1]);
    in_width = atoi(argv[1]);
    in_channel = atoi(argv[2]);
    filter_width = atoi(argv[3]);
    filter_height = atoi(argv[3]);
    filter_num = atoi(argv[4]);
	src_len = batch_count*filter_num*in_height*in_width;
    
    int inSize = batch_count * in_channel * in_height * in_width;
    int outSize = batch_count * filter_num * in_height * in_width;
    int filterSize = filter_num * in_channel * filter_height * filter_width;

    float *inData_NCHW = (float *)malloc(sizeof(float) * inSize);
    float *outData = (float *)malloc(sizeof(float) * outSize);
    float *filterData = (float *)malloc(sizeof(float) * filterSize);
	float *hostArray = new float[src_len];

    cudaMemGetInfo(&free,&total); 
    printf("%ld KB free of total %ld KB\n",free/1024,total/1024);

    //입력 데이터 셋팅
    for(int n = 0; n < batch_count; n++) {
        for (int k = 0; k < in_channel; k++) {
            for (int h = 0; h < in_height; h++) {
                for (int w = 0; w < in_width; w++) {
                    //inData_NCHW[convertToIndex(n, batch_count, k, in_channel, h, in_height, w, in_width)] = 0.001f; 
                    inData_NCHW[convertToIndex(n, batch_count, k, in_channel, h, in_height, w, in_width)] = 
                            static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 10);
                }
            }
        }
    }
    cout << "in_NCHW" << endl;
    for (int i = 0; i < min(4, in_channel); i++) {
        for (int y = 0; y < min(4, in_height); y++) {
            for (int x = 0; x < min(4, in_width); x++) {
                //printf("index : %d\n", convertToIndex(0, 1, i, in_channel, y, in_height, x, in_width));
                printf("%.3f ", inData_NCHW[convertToIndex(0, 1, i, in_channel, y, in_height, x, in_width)]);
            }
            cout << endl;
        }
        cout << endl;
    }
	
    //필터(가중치) 셋팅
	for(int n = 0 ; n < filter_num; n++){	
		for (int k = 0; k < in_channel; k++) {
			for (int h = 0; h < filter_height; h++) {
				for (int w = 0; w < filter_width; w++) {
					//filterData[convertToIndex(n, filter_num, k, in_channel, h, filter_height, w, filter_width) ] = 1;
					filterData[convertToIndex(n, filter_num, k, in_channel, h, filter_height, w, filter_width) ] =
                            static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) * 10);
				}
			}
		}
	}

    //GPU 메모리 할당
	checkCUDA(cudaMalloc((void**)&inData_d, inSize * sizeof(float)));
	checkCUDA(cudaMalloc((void**)&outData_d, outSize * sizeof(float)));
	checkCUDA(cudaMalloc((void**)&outData1_d, outSize * sizeof(float)));
	checkCUDA(cudaMalloc((void**)&filterData_d, filterSize * sizeof(float)));
    
    cudaMemGetInfo(&free,&total); 
    printf("%ld KB free of total %ld KB\n",free/1024,total/1024);

	//CPU 데이터를 GPU 메모리로 복사
	checkCUDA(cudaMemcpy(inData_d, inData_NCHW, inSize * sizeof(float), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(filterData_d, filterData, filterSize * sizeof(float), cudaMemcpyHostToDevice));

	//CUDNN 배열
	cudnnHandle_t cudnnHandle;// CUDNN을 사용하기 위한 핸들러
	cudnnTensorDescriptor_t inTensorDesc, outTensorDesc;//데이터 구조체 선언
	cudnnFilterDescriptor_t filterDesc;//필터 구조체 선언
	cudnnConvolutionDescriptor_t convDesc;//컨볼루션 구조체 선언 

	//할당
	checkCUDNN(cudnnCreate(&cudnnHandle));
	checkCUDNN(cudnnCreateTensorDescriptor(&inTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	//초기화
	//inData_NCHW 정보 - 구조가 [Number][Channel][Height][Width] 형태임을 알려줌
	checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_count, in_channel, in_height, in_width));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_num, in_channel, filter_height, filter_width));
	//컨볼루션의 패딩, 스트라이드, 컨볼루션 모드 등을 셋팅
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_vertical, stride_horizontal, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    
   
	int out_n, out_c, out_h, out_w;
	//입력데이터를 위에서 셋팅한 대로 컨볼루션 했을때 출력 데이터의 구조 알아내기
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
	printf("conv out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
	checkCUDNN(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

	//입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을때 가장 빠른 알고리즘이 무엇인지를 알아내기
	cudnnConvolutionFwdAlgo_t algo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
				inTensorDesc,
				filterDesc,
				convDesc,
				outTensorDesc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				0,
				&algo
				));

	//cout << "Fastest algorithm for conv0 = " << algo << endl;

	//위에서 알아낸 가장 빠른 알고리즘을 사용할 경우 계산과정에서 필요한 버퍼 데이터의 크기를 알아내기
	size_t sizeInBytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
				inTensorDesc,
				filterDesc,
				convDesc,
				outTensorDesc,
				algo,
				&sizeInBytes));

	cout << "sizeInBytes " << sizeInBytes << endl;

	//계산과정에서 버퍼 데이터가 필요한 경우가 있다면 메모리 할당
	if (sizeInBytes != 0) checkCUDA(cudaMalloc(&workSpace, sizeInBytes));

	float alpha = 1.0f;
	float beta = 0.0f;

    long t = timer_get();

    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
				&alpha,
				inTensorDesc,
				inData_d,
				filterDesc,
				filterData_d,
				convDesc,
				algo,
				workSpace,
				sizeInBytes,
				&beta,
				outTensorDesc,
				outData_d));
    
    time_accum += (timer_get() - t);

	checkCUDA(cudaMemcpy(outData, outData_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
	print("conv out", outData, filter_num, in_height, in_width);
    fprintf(stderr,   "[float  ]\t%9ld\n", time_accum);


	//메모리 해제
	checkCUDNN(cudnnDestroyTensorDescriptor(inTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outTensorDesc));

	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroy(cudnnHandle));

	checkCUDA(cudaFree(inData_d));
	checkCUDA(cudaFree(outData_d));;
	checkCUDA(cudaFree(filterData_d));
	checkCUDA(cudaThreadSynchronize());
	return 0;
}
