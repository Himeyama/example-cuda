#include "float_vector.h"
#define max(a, b) ((a > b) ? a : b)

__global__
void cuda_convolve_full(float *a, float *v, float *conv, FloatVec vec_a, FloatVec vec_v){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    conv[i] = 0;
    if(i < vec_a.size + vec_v.size - 1)
        for(int j = 0; j < vec_v.size; j++)
            conv[i] += v[j] * ((i - j >= vec_a.size || i - j < 0) ? 0 : a[i - j]);
}

FloatVec convolve(FloatVec a, FloatVec v){
    FloatVec conv;
    init_FloatVec(&conv, a.size + v.size - 1);
    float* ga = fary2cuda(a);
    float* gv = fary2cuda(v);
    float* gconv = fary2cuda(conv);
    cuda_convolve_full<<<1, 256>>>(ga, gv, gconv, a, v);
    cudaDeviceSynchronize();
    cudaMemcpy(conv.data, gconv, conv.bytes, cudaMemcpyDeviceToHost);
    cudaFree(ga);
    cudaFree(gv);
    cudaFree(gconv);
    return conv;
}

FloatVec convolve_same(FloatVec a, FloatVec v){
    FloatVec conv;
    long sidx = (int)round(v.size / 2.0) - 1;
    init_FloatVec(&conv, max(a.size, v.size));
    FloatVec conv_full = convolve(a, v);
    memcpy(conv.data, conv_full.data + sidx, conv.bytes);
    free(conv_full.data);
    return conv;
}

FloatVec convolve_valid(FloatVec a, FloatVec v){
    FloatVec conv;
    long sidx = v.size - 1;
    init_FloatVec(&conv, max(a.size, v.size) - min(a.size, v.size) + 1);
    FloatVec conv_full = convolve(a, v);
    memcpy(conv.data, conv_full.data + sidx, conv.bytes);
    free(conv_full.data);
    return conv;
}


void test(){
    FloatVec a, v;
    init_FloatVec(&a, 3);
    for(int i = 0; i < a.size; i++)
        a.data[i] = i + 1; 
    init_FloatVec(&v, 3);
    v.data[0] = 0;
    v.data[1] = 1;
    v.data[2] = 0.5;
    FloatVec conv = convolve(a, v);
    printVec(conv);
    free(conv.data);

    conv = convolve_same(a, v);
    printVec(conv);
    free(conv.data);

    conv = convolve_valid(a, v);
    printVec(conv);

    free(conv.data);
    free(v.data);
    free(a.data);
}

int main(void){
    test();

    // 割り当てを破壊し状態をリセット
    cudaDeviceReset();

    return 0;
}

