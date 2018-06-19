#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"

typedef struct {
    int h,w,c,stride,batch, out_h, out_w;
    float *delta, *output;
    float *output_gpu, *delta_gpu;
} maxpool_layer;

image get_maxpool_image(const maxpool_layer *layer, int batch);
maxpool_layer *make_maxpool_layer(int h, int w, int c, int stride, int batch);
void forward_maxpool_layer(const maxpool_layer *layer, float *in);
void backward_maxpool_layer(const maxpool_layer *layer, float *in, float *delta);

#endif

