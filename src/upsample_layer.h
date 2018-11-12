#ifndef UPSAMPLE_LAYER_H
#define UPSAMPLE_LAYER_H

#include "image.h"
#ifdef GPU
#include "cuda.h"
#elif defined(OPENCL)
#include "opencl.h"
#endif

typedef struct{
    int w, h, c, batch, out_w, out_h, out_c, stride, inputs, outputs, test;
    float scale;
    float *output, *delta;
    float *delta_gpu, *output_gpu;
#ifdef OPENCL
    cl_mem delta_cl, output_cl;
#endif
} upsample_layer;

image get_upsample_image(const upsample_layer *layer);
upsample_layer *make_upsample_layer(int batch, int w, int h, int c, int stride, int test);
void free_upsample_layer(void *input);
void forward_upsample_layer(const upsample_layer *l, float *input);
void backward_upsample_layer(const upsample_layer *l, float * delta);

#ifdef GPU
void forward_upsample_layer_gpu(const upsample_layer *l, float *input);
void backward_upsample_layer_gpu(const upsample_layer *l, float *delta);
#elif defined(OPENCL)
void forward_upsample_layer_cl(const upsample_layer *l, cl_mem input);
#endif

#endif
