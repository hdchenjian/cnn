#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "image.h"
#include "utils.h"

#ifdef GPU
    #include "cuda.h"
#endif

typedef struct {
    int batch, inputs, outputs, h, w, c;
    float probability, scale;
    float *rand, *output, *delta;
    float *rand_gpu, *output_gpu, *delta_gpu;
} dropout_layer;

dropout_layer *make_dropout_layer(int w, int h, int c, int batch, int inputs, float probability);
void resize_dropout_layer(dropout_layer *l, int inputs);
void forward_dropout_layer(const dropout_layer *l, float *input, int test);
void backward_dropout_layer(const dropout_layer *l, float *delta);
image get_dropout_image(const dropout_layer *layer);

#ifdef GPU
void forward_dropout_layer_gpu(const dropout_layer *l, float *input, int test);
void backward_dropout_layer_gpu(const dropout_layer *l, float *delta);
#endif

#endif
