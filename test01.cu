#include <stdio.h>
#include <string.h>

__global__
void convolve_full(float* a, float* v, float *conv, int alen, int vlen){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    conv[i] = 0;
    if(i < alen + vlen - 1){
        for(int j = 0; j < vlen; j++){
            float ta;
            if(i - j >= alen || i - j < 0){
                ta = 0;
            }else{
                ta = a[i - j];
            }
            conv[i] += v[j] * ta;
            printf("ta[%d]: %f, %f\n", j, ta, v[j]);
        }
    }
}

typedef struct __FloatVec{
    float* data;
    long size;
    long bytes;
} FloatVec;

void init_FloatVec(FloatVec *a, long size){
    a->size = size;
    a->data = (float*)malloc(sizeof(float) * size);
    a->bytes = sizeof(float) * size;
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

int main(void){
    // 配列の大きさ
    int alen = 6;
    int vlen = 2;

    // FloatVec aa;
    // init_FloatVec(&aa, 6);
    // printVec(aa);
    // free(aa.data);



    // 配列のバイト数
    int n = alen * sizeof(float);
    int conv_bytes = sizeof(float) * (alen + vlen - 1);

    // 配列メモリーと GPU メモリーのポインタ宣言
    float *a, *ga, *v, *gv, *conv, *gconv;

    // メモリー確保
    a = (float*)malloc(n);
    cudaMalloc((void**)&ga, n);
    for(int i = 0; i < alen; i++)
        a[i] = i;
    cudaMemcpy(ga, a, n, cudaMemcpyHostToDevice);

    v = (float*)malloc(sizeof(float) * vlen);
    cudaMalloc((void**)&gv, sizeof(float) * vlen);
    v[0] = 0.2;
    v[1] = 0.8;
    cudaMemcpy(gv, v, sizeof(float) * vlen, cudaMemcpyHostToDevice);

    conv = (float*)malloc(conv_bytes);
    cudaMalloc((void**)&gconv, conv_bytes);

    // カーネル(関数)呼び出し
    convolve_full<<<1, 256>>>(ga, gv, gconv, alen, vlen);

    // 待機
    cudaDeviceSynchronize();
    
    cudaMemcpy(conv, gconv, conv_bytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < alen + vlen - 1; i++){
        printf("%f\n", conv[i]);
    }

    // メモリー開放
    free(a);
    free(v);
    free(conv);
    cudaFree(ga);
    cudaFree(gv);
    cudaFree(gconv);

    // 割り当てを破壊し状態をリセット
    cudaDeviceReset();


    return 0;
}

