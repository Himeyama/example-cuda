#define max(a, b) ((a > b) ? a : b)

#include <stdio.h>
#include <string.h>

typedef struct __FloatVec{
    float* data;
    long size;
    long bytes;
} FloatVec;

void init_FloatVec(FloatVec *a, long size){
    a->size = size;
    a->data = (float*)malloc(sizeof(float) * size);
    a->bytes = sizeof(float) * size;
    memset(a->data, 0, a->bytes);
}

void printVec(FloatVec a){
    char *tmp = (char*)malloc(a.size * 24 + 3);
    tmp[0] = '[';
    tmp[1] = '\0';
    char number[12];
    for(int i = 0; i < a.size; i++){
        if(i < a.size - 1)
            sprintf(number, "%f, ", a.data[i]);
        else
            sprintf(number, "%f]", a.data[i]);
        strcat(tmp, number);
    }
    puts(tmp);
    free(tmp);
}

float* fary2cuda(FloatVec a){
    float *g;
    cudaMalloc((void**)&g, a.bytes);
    cudaMemcpy(g, a.data, a.bytes, cudaMemcpyHostToDevice);
    return g;
}

__global__
void cuda_convolve_full(float *a, float *v, float *conv, FloatVec vec_a, FloatVec vec_v){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    conv[i] = 0;
    if(i < vec_a.size + vec_v.size - 1)
        for(int j = 0; j < vec_v.size; j++)
            conv[i] += v[j] * ((i - j >= vec_a.size || i - j < 0) ? 0 : a[i - j]);
}

FloatVec convolve_full(FloatVec a, FloatVec v){
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
    FloatVec conv_full = convolve_full(a, v);
    memcpy(conv.data, conv_full.data + sidx, conv.bytes);
    free(conv_full.data);
    return conv;
}

FloatVec convolve_valid(FloatVec a, FloatVec v){
    FloatVec conv;
    long sidx = v.size - 1;
    init_FloatVec(&conv, max(a.size, v.size) - min(a.size, v.size) + 1);
    FloatVec conv_full = convolve_full(a, v);
    memcpy(conv.data, conv_full.data + sidx, conv.bytes);
    free(conv_full.data);
    return conv;
}


int main(void){
    FloatVec a, v;
    init_FloatVec(&a, 6);
    for(int i = 0; i < a.size; i++)
        a.data[i] = i;    

    init_FloatVec(&v, 2);
    v.data[0] = 0.2;
    v.data[1] = 0.8;

    FloatVec conv = convolve_full(a, v);
    FloatVec conv_same = convolve_same(a, v);
    FloatVec conv_valid = convolve_valid(a, v);

    printVec(conv);
    printVec(conv_same);
    printVec(conv_valid);

    free(conv.data);
    free(v.data);
    free(a.data);

    // 割り当てを破壊し状態をリセット
    cudaDeviceReset();

    return 0;
}

