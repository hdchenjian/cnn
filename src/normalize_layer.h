#ifndef NORMALIZE_LAYER_H
#define NORMALIZE_LAYER_H

#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "blas.h"

#ifdef GPU
    #include "cuda.h"
#elif defined(OPENCL)
    #include "opencl.h"
#endif

typedef struct {
    int batch, inputs, outputs, h, w, c, test;
    float *output, *delta, *norm_data;
    float *output_gpu, *delta_gpu, *norm_data_gpu;
#ifdef OPENCL
    cl_mem output_cl, delta_cl, norm_data_cl;
#endif
} normalize_layer;

normalize_layer *make_normalize_layer(int w, int h, int c, int batch, int test);
void resize_normalize_layer(const normalize_layer *l, int inputs);
void forward_normalize_layer(const normalize_layer *l, float *input);
void backward_normalize_layer(const normalize_layer *l, float *delta);

#ifdef GPU
void forward_normalize_layer_gpu(const normalize_layer *l, float *input);
void backward_normalize_layer_gpu(const normalize_layer *l, float *delta);
#elif defined(OPENCL)
void forward_normalize_layer_cl(const normalize_layer *l, cl_mem input);
#endif

#endif
