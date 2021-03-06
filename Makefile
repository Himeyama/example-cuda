NVCC = /usr/local/cuda/bin/nvcc
 CXX = /usr/bin/g++
  CC = /usr/bin/gcc
 OPT = -ccbin $(CC) \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75


test00: test00.cu
	$(NVCC) $(OPT) $< -o $@

convolve: convolve.cu float_vector.cu
	$(NVCC) $(OPT) $^ -o $@