#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#ifdef GPU
#include "cuda.h"
#endif

typedef struct {
    int h,w,c,stride,batch, out_h, out_w, pad, size;
    float *delta, *output;
    int *indexes, *indexes_gpu;
    float *output_gpu, *delta_gpu;
} maxpool_layer;

image get_maxpool_image(const maxpool_layer *layer);
maxpool_layer *make_maxpool_layer(int h, int w, int c, int size, int stride, int batch, int padding);
void forward_maxpool_layer(const maxpool_layer *layer, float *in);
void backward_maxpool_layer(const maxpool_layer *layer, float *in, float *delta);

#ifdef GPU
void forward_maxpool_layer_gpu(const maxpool_layer *layer, float *in_gpu);
void backward_maxpool_layer_gpu(const maxpool_layer *layer, float *delta_gpu);
#endif

#endif

