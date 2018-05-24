#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "image.h"
#include "activations.h"

typedef struct {
    int h,w,c;
    int n;
    int size;
    int stride;
    image *kernels;
    image *kernel_updates;
    image *kernel_momentum;
    float *biases;
    float *bias_updates;
    float *bias_momentum;
    image upsampled;
    float *delta;
    float *output;

    ACTIVATION activation;
    int edge;
} convolutional_layer;

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, ACTIVATION activation);
void forward_convolutional_layer(const convolutional_layer *layer, float *in);
void backward_convolutional_layer(const convolutional_layer *layer, float *input, float *delta);
void learn_convolutional_layer(const convolutional_layer *layer, float *input);

void update_convolutional_layer(const convolutional_layer *layer, double step, double momentum, double decay);

image get_convolutional_image(const convolutional_layer *layer);
image get_convolutional_delta(const convolutional_layer *layer);

#endif

