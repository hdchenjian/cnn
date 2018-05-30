#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "activations.h"
#include "gemm.h"

typedef struct{
    int inputs, outputs, batch;
    float *output;
    float *delta;
    float *weights;
    float *weight_updates;
    float *biases;
    float *bias_updates;
    ACTIVATION activation;
} connected_layer;

connected_layer *make_connected_layer(int inputs, int outputs, int batch, ACTIVATION activation);
void forward_connected_layer(connected_layer *layer, float *input);
void backward_connected_layer(connected_layer *layer, float *input, float *delta);
void update_connected_layer(connected_layer *layer, float learning_rate, float momentum, float decay);
#endif
