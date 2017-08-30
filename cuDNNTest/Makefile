all:
	nvcc -o test -gencode arch=compute_61,code=[sm_61,compute_61] -I/usr/local/cuda-8.0/include cuDNNTest.cu  -lcudart -lcudnn -g
clean:
	rm test
