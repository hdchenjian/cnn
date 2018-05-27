#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "activations.h"

typedef struct{
    int inputs;
    int outputs;

    float *weights;
    float *weight_momentum;
    float *weight_updates;

    float *biases;
    float *bias_momentum;
    float *bias_updates;

    float *output;
    float *delta;
    ACTIVATION activation;
} connected_layer;

connected_layer *make_connected_layer(int inputs, int outputs, ACTIVATION activation);

void forward_connected_layer(connected_layer *layer, float *input);
void backward_connected_layer(connected_layer *layer, float *delta);
void learn_connected_layer(connected_layer *layer, float *input);
void update_connected_layer(connected_layer *layer, float step, float momentum, float decay);


#endif

