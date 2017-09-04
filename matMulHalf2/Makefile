all:
	nvcc -o test -gencode arch=compute_53,code=[sm_53,compute_53] -I/usr/local/cuda-8.0/include matMul.cu float.cu half.cu half2.cu -lcudart -g 
debug:
	nvcc -o test -DDEBUG -gencode arch=compute_53,code=[sm_53,compute_53] -I/usr/local/cuda-8.0/include matMul.cu float.cu half.cu half2.cu -lcudart -g 
clean:
	rm test
