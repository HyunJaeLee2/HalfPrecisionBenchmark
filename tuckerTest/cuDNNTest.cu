#include "util.hpp"
#include "cuda.h"
#include "cuDNNTest.h"

#define ITER_COUNT 20

#define checkCUDNN(res) if((res)!=CUDNN_STATUS_SUCCESS) {fprintf(stderr, "CUDNN error! %d (%s:%d)\n", res, __FILE__,__LINE__);}
#define checkCUDA(res) if((res)!=(cudaError_t)CUDA_SUCCESS) {fprintf(stderr, "CUDA error! %d (%s:%d)\n", res, __FILE__,__LINE__);}
using namespace std;

const int batch_count = 1;
long time_accum = 0;
cudnnHandle_t cudnnHandle;
bool isHalfPrecision;

void free_layer(conv_layer *layer)
{
	checkCUDNN(cudnnDestroyTensorDescriptor(layer->inTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(layer->outTensorDesc));

	checkCUDNN(cudnnDestroyConvolutionDescriptor(layer->convDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(layer->filterDesc));

    if(layer->d_inData) checkCUDA(cudaFree(layer->d_inData));
    if(layer->d_outData) checkCUDA(cudaFree(layer->d_outData));;
    if(layer->d_filterData) checkCUDA(cudaFree(layer->d_filterData));

    if(layer->d_half_inData) checkCUDA(cudaFree(layer->d_half_inData));
    if(layer->d_half_outData) checkCUDA(cudaFree(layer->d_half_outData));;
    if(layer->d_half_filterData) checkCUDA(cudaFree(layer->d_half_filterData));
    
    if(layer->workSpace) checkCUDA(cudaFree(layer->workSpace));
}

conv_layer initFirstLayerWithRandom(int in_len, int in_channel, int filter_len, int filter_num, int padding, int stride)
{
	float *inData, *filterData;
    int inSize, outSize, filterSize;

    //Set to 1 if input is zero when applying tucker
    if(in_channel == 0) in_channel = 1;
    if(filter_num == 0) filter_num = 1;
    
    conv_layer layer = {0};

//    printf("in_len : %d, in_channel : %d, filter_len : %d, filter_num : %d, padding = %d, stride = %d\n", 
//            in_len, in_channel, filter_len, filter_num, padding, stride);
    
    //set padding and stride 
    if(filter_len == 1)  
        layer.padding_h = layer.padding_w = 0;  
    else
        layer.padding_h = layer.padding_w = padding;
    layer.stride_vertical = layer.stride_horizontal = stride;
   
    //set input and filter
    layer.in_height = layer.in_width = in_len; 
    layer.in_channel = in_channel;
    layer.filter_height = layer.filter_width = filter_len;
    layer.filter_num = filter_num;
    
    //Init Size 
    inSize = batch_count * layer.in_channel * layer.in_height * layer.in_width;
    filterSize = layer.filter_num * layer.in_channel * layer.filter_height * layer.filter_width;
    outSize = batch_count * layer.filter_num * layer.in_height * layer.in_width;
    layer.outSize = outSize;
    layer.inSize = inSize;
    layer.filterSize = filterSize;

    //Init Data
    inData = (float *)malloc(sizeof(float) * inSize);
    filterData = (float *)malloc(sizeof(float) * filterSize);

    initWithRandom4D(inData, batch_count, layer.in_channel, layer.in_height, layer.in_width);
    //print4D("In Data", inData, batch_count, layer.in_channel, layer.in_height, layer.in_width);

    initWithRandom4D(filterData, layer.filter_num, layer.in_channel, layer.filter_height, layer.filter_width);
    //print4D("Filter Data", filterData, layer.filter_num, layer.in_channel, layer.filter_height, layer.filter_width);
    
    checkCUDA(cudaMalloc((void**)&layer.d_inData, inSize * sizeof(float)));
	checkCUDA(cudaMalloc((void**)&layer.d_filterData, filterSize * sizeof(float)));
    checkCUDA(cudaMalloc((void**)&layer.d_outData, outSize * sizeof(float)));
    
	checkCUDA(cudaMemcpy(layer.d_inData, inData, inSize * sizeof(float), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(layer.d_filterData, filterData, filterSize * sizeof(float), cudaMemcpyHostToDevice));

	checkCUDNN(cudnnCreateTensorDescriptor(&layer.inTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&layer.outTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&layer.filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&layer.convDesc));
    
    const int convDims = 2;
    int padA[convDims] = {layer.padding_h, layer.padding_w};
    int filterStrideA[convDims] = {layer.stride_vertical, layer.stride_horizontal};
    int upscaleA[convDims] = {1, 1};
    int out_n, out_c, out_h, out_w;
    
    if(!isHalfPrecision)
    {
        checkCUDNN(cudnnSetTensor4dDescriptor(layer.inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_count, layer.in_channel, layer.in_height, layer.in_width));
        checkCUDNN(cudnnSetFilter4dDescriptor(layer.filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, layer.filter_num, layer.in_channel, layer.filter_height, layer.filter_width));
        
        cudnnSetConvolutionNdDescriptor(layer.convDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); 
        
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(layer.convDesc, layer.inTensorDesc, layer.filterDesc, &out_n, &out_c, &out_h, &out_w));
        
        checkCUDNN(cudnnSetTensor4dDescriptor(layer.outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    }
    else
    {
        checkCUDA(cudaMalloc((void**)&layer.d_half_inData, inSize * sizeof(__half)));
        checkCUDA(cudaMalloc((void**)&layer.d_half_filterData, filterSize * sizeof(__half)));
        checkCUDA(cudaMalloc((void**)&layer.d_half_outData, outSize * sizeof(__half)));

        gpu_float2half_rn(inSize, layer.d_inData, layer.d_half_inData);
        gpu_float2half_rn(filterSize, layer.d_filterData, layer.d_half_filterData);

        checkCUDNN(cudnnSetTensor4dDescriptor(layer.inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batch_count, layer.in_channel, layer.in_height, layer.in_width));
        checkCUDNN(cudnnSetFilter4dDescriptor(layer.filterDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, layer.filter_num, layer.in_channel, layer.filter_height, layer.filter_width));
        
        cudnnSetConvolutionNdDescriptor(layer.convDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF); 
        
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(layer.convDesc, layer.inTensorDesc, layer.filterDesc, &out_n, &out_c, &out_h, &out_w));
        
        checkCUDNN(cudnnSetTensor4dDescriptor(layer.outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, out_n, out_c, out_h, out_w));
    }

	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
				layer.inTensorDesc,
				layer.filterDesc,
				layer.convDesc,
				layer.outTensorDesc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				0,
				&layer.algo
				));


	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
				layer.inTensorDesc,
				layer.filterDesc,
				layer.convDesc,
				layer.outTensorDesc,
				layer.algo,
				&layer.sizeInBytes));
#ifdef DEBUG
    printf("conv out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
	cout << "Fastest layer.algorithm for conv0 = " << layer.algo << endl;
	cout << "sizeInBytes " << layer.sizeInBytes << endl;
#endif

	if (layer.sizeInBytes != 0) checkCUDA(cudaMalloc(&layer.workSpace, layer.sizeInBytes));

    free(inData);
    free(filterData);

    return layer;
}

void executeLayer(int in_len, int in_channel, int filter_len, int filter_num, int padding, int stride)
{
    conv_layer conv1 = initFirstLayerWithRandom(in_len, in_channel, filter_len, filter_num, padding, stride);
	
    float *outData = (float *)malloc(sizeof(float) * conv1.outSize);

	float alpha = 1.0f;
	float beta = 0.0f;

    long prev = timer_get();
    long time_diff;
    if(!isHalfPrecision)
    {
        checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                    &alpha,
                    conv1.inTensorDesc,
                    conv1.d_inData,
                    conv1.filterDesc,
                    conv1.d_filterData,
                    conv1.convDesc,
                    conv1.algo,
                    conv1.workSpace,
                    conv1.sizeInBytes,
                    &beta,
                    conv1.outTensorDesc,
                    conv1.d_outData));
        
        checkCUDA(cudaDeviceSynchronize());

        time_diff = (timer_get() - prev);
        time_accum += time_diff; 

        checkCUDA(cudaMemcpy(outData, conv1.d_outData, sizeof(float)* conv1.outSize, cudaMemcpyDeviceToHost));
        //print4D("conv out", outData, 1, conv1.filter_num, conv1.in_height, conv1.in_width);
#ifdef DEBUG
        fprintf(stderr,   "[float  ]\t%9ld\n", time_diff);
#endif
    }
    else
    {
        if(conv1.d_inData) {checkCUDA(cudaFree(conv1.d_inData)); conv1.d_inData = NULL;}
        if(conv1.d_outData){checkCUDA(cudaFree(conv1.d_outData)); conv1.d_outData = NULL;}
        if(conv1.d_filterData) {checkCUDA(cudaFree(conv1.d_filterData)); conv1.d_filterData = NULL;};
        
        checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                    &alpha,
                    conv1.inTensorDesc,
                    conv1.d_half_inData,
                    conv1.filterDesc,
                    conv1.d_half_filterData,
                    conv1.convDesc,
                    conv1.algo,
                    conv1.workSpace,
                    conv1.sizeInBytes,
                    &beta,
                    conv1.outTensorDesc,
                    conv1.d_half_outData));

        checkCUDA(cudaDeviceSynchronize());
        
        time_diff = (timer_get() - prev);
        time_accum += time_diff; 

        //d_outData was freed then reallocated to optimize memory
        checkCUDA(cudaMalloc((void**)&conv1.d_outData, conv1.outSize * sizeof(float)));

        gpu_half2float(conv1.outSize, conv1.d_half_outData, conv1.d_outData);
		checkCUDA(cudaMemcpy(outData, conv1.d_outData, sizeof(float)* conv1.outSize, cudaMemcpyDeviceToHost));
		//print4D("conv out", outData, 1, conv1.filter_num, conv1.in_height, conv1.in_width);
#ifdef DEBUG
		fprintf(stderr,   "[half  ]\t%9ld\n", time_diff);
#endif
    }
    free(outData);
    free_layer(&conv1);

}

int main(int argc, char* argv[])
{
    int in_len, in_channel, filter_len, filter_num;
    int padding, stride;
    int rateS, rateT;
    long time_save;
    bool applyTucker = true;

    checkCUDNN(cudnnCreate(&cudnnHandle));
    
    if(argc < 9)
    {
        fprintf(stderr, "usage : ./test [Input Height/Width] [Channel] [Filter Height/Width] [Filter Num] [padding] [stride] [Rate of S] [Rate of T] \n");
        return 0;
    }
   
    in_len =  atoi(argv[1]);
    in_channel =  atoi(argv[2]);
    filter_len =  atoi(argv[3]);
    filter_num =  atoi(argv[4]);
    padding = atoi(argv[5]);
    stride = atoi(argv[6]);
    rateS = atoi(argv[7]);
    rateT = atoi(argv[8]);

    if(filter_len == 1) applyTucker = false;
    
    //Execute Float mode
    isHalfPrecision = false;
    for(int i = 0; i < ITER_COUNT; i++){
        executeLayer(in_len, in_channel, filter_len, filter_num, padding, stride);
    }
    checkCUDA(cudaThreadSynchronize());

    time_save = time_accum;
    time_accum = 0;
    
    fprintf(stderr,   "[Float w/o tucker  ]\t%9ld\n", time_save / ITER_COUNT);
    if(applyTucker){
        for(int i = 0; i < ITER_COUNT; i++){
            executeLayer(in_len, in_channel, 1, in_channel * rateS / 100, padding, stride);
            executeLayer(in_len, in_channel * rateS / 100, filter_len, filter_num * rateT / 100 , padding, stride);
            executeLayer(in_len, filter_num * rateT / 100, 1, filter_num, padding, stride);
        }
        checkCUDA(cudaThreadSynchronize());
        fprintf(stderr,   "[Float tucker applied ]\t%9ld\n", time_accum / ITER_COUNT);
    }
    
    time_accum = 0;
    isHalfPrecision = true;

    //Execute Half mode
    for(int i = 0; i < ITER_COUNT; i++){
        executeLayer(in_len, in_channel, filter_len, filter_num, padding, stride);
    }
    checkCUDA(cudaThreadSynchronize());

    time_save = time_accum;
    time_accum = 0;
    
    fprintf(stderr,   "[Half w/o tucker  ]\t%9ld\n", time_save / ITER_COUNT);
    if(applyTucker){
        for(int i = 0; i < ITER_COUNT; i++){
            executeLayer(in_len, in_channel, 1, in_channel * rateS / 100, padding, stride);
            executeLayer(in_len, in_channel * rateS / 100, filter_len, filter_num * rateT / 100 , padding, stride);
            executeLayer(in_len, filter_num * rateT / 100, 1, filter_num, padding, stride);
        }
        checkCUDA(cudaThreadSynchronize());
        fprintf(stderr,   "[Half tucker applied ]\t%9ld\n", time_accum / ITER_COUNT);
    }

    return 0;
}
