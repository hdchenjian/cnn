#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdlib.h>
#include <string.h>

#include "activations.h"
#include "gemm.h"
#include "utils.h"

typedef struct {
    int h,w,c;
    int n, size;
    int stride;
    int out_h, out_w;
    float *delta;
    float *output;
    float *weights;
    float *weight_updates;
    float *weight_momentum;
    float *biases;
    float *bias_updates;
    float *bias_momentum;
    ACTIVATION activation;
} convolutional_layer;

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride,
		ACTIVATION activation, size_t *workspace_size);
void forward_convolutional_layer(const convolutional_layer *layer, float *in, float *workspace);
void backward_convolutional_layer(const convolutional_layer *layer, float *input, float *delta, float *workspace);
void update_convolutional_layer(const convolutional_layer *layer, float learning_rate, float momentum, float decay);
#endif
