#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

typedef struct {
    int inputs, batch;
    float *delta;
    float *output;
} softmax_layer;

softmax_layer *make_softmax_layer(int inputs, int batch);
void forward_softmax_layer(const softmax_layer *layer, float *input);
void backward_softmax_layer(const softmax_layer *layer, float *delta);

#endif
