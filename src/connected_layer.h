#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "activations.h"
#include "gemm.h"

typedef struct{
    int inputs;
    int outputs;
    float *output;
    float *delta;
    float *weights;
    float *weight_updates;
    float *weight_momentum;
    float *biases;
    float *bias_updates;
    float *bias_momentum;
    ACTIVATION activation;
} connected_layer;

connected_layer *make_connected_layer(int inputs, int outputs, ACTIVATION activation);
void forward_connected_layer(connected_layer *layer, float *input);
void backward_connected_layer(connected_layer *layer, float *input, float *delta);
void update_connected_layer(connected_layer *layer, float step, float momentum, float decay);
#endif
