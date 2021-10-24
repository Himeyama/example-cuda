#include <stdio.h>
#include <cuda_runtime.h>

__global__
void hello_world(float* a){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    a[i] = i;
}

int main(void){
    int size = 256;
    int n = size * sizeof(float);
    float *a, *g;

    a = (float*)malloc(n);
    cudaMalloc((void**)&g, n);
    
    hello_world<<<1, 256>>>(g);
    cudaDeviceSynchronize();
    
    cudaMemcpy(a, g, n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++)
        printf("%f\n", a[i]);

    free(a);
    cudaFree(g);

    cudaDeviceReset();

    return 0;
}