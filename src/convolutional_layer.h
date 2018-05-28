#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "image.h"
#include "activations.h"

typedef struct {
    int h,w,c;
    int n, size;
    int stride;
    image *weights;
    image *weight_updates;
    image *weight_momentum;
    float *biases;
    float *bias_updates;
    float *bias_momentum;
    float *delta;
    float *output;

    ACTIVATION activation;
} convolutional_layer;

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, ACTIVATION activation);
void forward_convolutional_layer(const convolutional_layer *layer, float *in);
void backward_convolutional_layer(const convolutional_layer *layer, float *delta);
void learn_convolutional_layer(const convolutional_layer *layer, float *input);

void update_convolutional_layer(const convolutional_layer *layer, float learning_rate, float momentum, float decay);

image get_convolutional_image(const convolutional_layer *layer);
image get_convolutional_delta(const convolutional_layer *layer);

#endif

