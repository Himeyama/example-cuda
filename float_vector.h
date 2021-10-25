/*
 *  (c) 2021 Murata Mitsuharu
 *  Licensed under the MIT License.
 *  source: https://github.com/example-cuda
*/

#ifndef FLOAT_VECTOR_H
#define FLOAT_VECTOR_H

typedef struct __FloatVec{
    float* data;
    long size;
    long bytes;
} FloatVec;

void init_FloatVec(FloatVec *a, long size);
void printVec(FloatVec a);
float* fary2cuda(FloatVec a);

#endif