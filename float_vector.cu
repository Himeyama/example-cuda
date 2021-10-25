#include <stdio.h>
#include "float_vector.h"

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