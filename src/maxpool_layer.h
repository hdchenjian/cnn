#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#ifdef GPU
#include "cuda.h"
#elif defined(OPENCL)
#include "opencl.h"
#endif

typedef struct {
    int h,w,c,stride,batch, outputs, out_h, out_w, pad, size, test;
    float *delta, *output;
    int *indexes, *indexes_gpu;
    float *output_gpu, *delta_gpu;
#ifdef OPENCL
    cl_mem output_cl, delta_cl, indexes_cl;
#endif
} maxpool_layer;

image get_maxpool_image(const maxpool_layer *layer);
maxpool_layer *make_maxpool_layer(int h, int w, int c, int size, int stride, int batch, int padding, int test);
void forward_maxpool_layer(const maxpool_layer *layer, float *in);
void backward_maxpool_layer(const maxpool_layer *layer, float *delta);

#ifdef GPU
void forward_maxpool_layer_gpu(const maxpool_layer *layer, float *in_gpu);
void backward_maxpool_layer_gpu(const maxpool_layer *layer, float *delta_gpu);
#elif defined(OPENCL)
void forward_maxpool_layer_cl(const maxpool_layer *layer, cl_mem in_gpu);
#endif

#endif

