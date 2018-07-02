#ifndef NORMALIZE_LAYER_H
#define NORMALIZE_LAYER_H

#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "blas.h"

#ifdef GPU
    #include "cuda.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

typedef struct {
    int batch, inputs, h, w, c;
    float *output, *delta;
    float *output_gpu, *delta_gpu;
} normalize_layer;

normalize_layer *make_normalize_layer(int w, int h, int c, int batch);
void resize_normalize_layer(const normalize_layer *l, int inputs);
void forward_normalize_layer(const normalize_layer *l, float *input);
void backward_normalize_layer(const normalize_layer *l, float *delta);

#ifdef GPU
void forward_normalize_layer_gpu(const normalize_layer *l, float *input);
void backward_normalize_layer_gpu(const normalize_layer *l, float *delta);
#endif

#endif
