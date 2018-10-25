#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"

#ifdef GPU
    #include "cuda.h"
#endif

typedef struct {
    int h,w,c,batch, outputs, normalize_type;
    float *delta, *output;
    float *output_gpu, *delta_gpu;
} avgpool_layer;

image get_avgpool_image(const avgpool_layer *l);
avgpool_layer *make_avgpool_layer(int batch, int w, int h, int c);
void resize_avgpool_layer(avgpool_layer *l, int w, int h);
void forward_avgpool_layer(const avgpool_layer *l, float *in);
void backward_avgpool_layer(const avgpool_layer *l, float *delta);

#ifdef GPU
void forward_avgpool_layer_gpu(const avgpool_layer *l, float *in);
void backward_avgpool_layer_gpu(const avgpool_layer *l, float *delta);
#endif

#endif

