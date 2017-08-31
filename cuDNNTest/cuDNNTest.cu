#include "util.hpp"
#include "cuda.h"
#include "cuDNNTest.h"

using namespace std;

const int batch_count = 1;
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

conv_layer initFirstLayerWithRandom(char *argv[])
{
	float *inData, *filterData;
    int inSize, outSize, filterSize;
    
    conv_layer layer = {0};

    //set padding and stride to 1 as default
    layer.padding_h = layer.padding_w = layer.stride_vertical = layer.stride_horizontal = 1;

    layer.in_height = layer.in_width = atoi(argv[1]);
    layer.in_channel = atoi(argv[2]);
    layer.filter_height = layer.filter_width = atoi(argv[3]);
    layer.filter_num = atoi(argv[4]);
    
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
    printf("conv out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);

	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
				layer.inTensorDesc,
				layer.filterDesc,
				layer.convDesc,
				layer.outTensorDesc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				0,
				&layer.algo
				));

	cout << "Fastest layer.algorithm for conv0 = " << layer.algo << endl;

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
				layer.inTensorDesc,
				layer.filterDesc,
				layer.convDesc,
				layer.outTensorDesc,
				layer.algo,
				&layer.sizeInBytes));

	cout << "sizeInBytes " << layer.sizeInBytes << endl;

	if (layer.sizeInBytes != 0) checkCUDA(cudaMalloc(&layer.workSpace, layer.sizeInBytes));

    free(inData);
    free(filterData);
    
    return layer;
}

int main(int argc, char* argv[])
{
    long time_accum = 0;
    float *outData;
	conv_layer conv1;

    if(argc < 6)
    {
        fprintf(stderr, "usage : ./test [Input Height/Width] [Channel] [Filter Height/Width] [Filter Num] [float/half]\n");
        return 0;
    }
    
    if(strcmp(argv[5], "half") == 0) 
        isHalfPrecision = true;
    else 
        isHalfPrecision = false;
    
    
    checkCUDNN(cudnnCreate(&cudnnHandle));
    
    conv1 = initFirstLayerWithRandom(argv);   
	
    outData = (float *)malloc(sizeof(float) * conv1.outSize);

	float alpha = 1.0f;
	float beta = 0.0f;
   
    long t = timer_get();
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
        time_accum += (timer_get() - t);

        checkCUDA(cudaMemcpy(outData, conv1.d_outData, sizeof(float)* conv1.outSize, cudaMemcpyDeviceToHost));
		//print("conv out", outData, conv1.filter_num, conv1.in_height, conv1.in_width);
        print4D("conv out", outData, 1, conv1.filter_num, conv1.in_height, conv1.in_width);

        fprintf(stderr,   "[float  ]\t%9ld\n", time_accum);
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
        time_accum += (timer_get() - t);

        //d_outData was freed then reallocated to optimize memory
        checkCUDA(cudaMalloc((void**)&conv1.d_outData, conv1.outSize * sizeof(float)));
        
        gpu_half2float(conv1.outSize, conv1.d_half_outData, conv1.d_outData);
		checkCUDA(cudaMemcpy(outData, conv1.d_outData, sizeof(float)* conv1.outSize, cudaMemcpyDeviceToHost));
		print4D("conv out", outData, 1, conv1.filter_num, conv1.in_height, conv1.in_width);

		fprintf(stderr,   "[half  ]\t%9ld\n", time_accum);
    }

    free_layer(&conv1);

	checkCUDA(cudaThreadSynchronize());
	return 0;
}
