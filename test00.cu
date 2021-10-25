#include <stdio.h>

__global__
void hello_world(float* a){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    a[i] = i;
}

int main(void){
    // 配列の大きさ
    int size = 256;

    // 配列のバイト数
    int n = size * sizeof(float);

    // 配列メモリーと GPU メモリーのポインタ宣言
    float *a, *g;

    // メモリー確保
    a = (float*)malloc(n);
    cudaMalloc((void**)&g, n);
    
    // カーネル(関数)呼び出し
    hello_world<<<1, 256>>>(g);

    // 待機
    cudaDeviceSynchronize();
    
    // GPU メモリーをホストメモリーにコピー
    cudaMemcpy(a, g, n, cudaMemcpyDeviceToHost);

    // 配列の中身を表示
    for(int i = 0; i < size; i++)
        printf("%f\n", a[i]);

    // メモリー開放
    free(a);
    cudaFree(g);

    // 割り当てを破壊し状態をリセット
    cudaDeviceReset();

    return 0;
}