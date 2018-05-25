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
    double *weights;
    double *biases;

    double *weight_updates;
    double *bias_updates;

    double *weight_momentum;
    double *bias_momentum;

    double *output;
    double *delta;

    ACTIVATION activation;

} connected_layer;

connected_layer *make_connected_layer(int inputs, int outputs, ACTIVATION activation);

void forward_connected_layer(connected_layer *layer, double *input);
void backward_connected_layer(connected_layer *layer, double *input, double *delta);
void learn_connected_layer(connected_layer *layer, double *input);
void update_connected_layer(connected_layer *layer, double step, double momentum, double decay);


#endif

