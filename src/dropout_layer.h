#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "network.h"

dropout_layer *make_dropout_layer(int w, int h, int c, int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer *l, float *input, network *net);
void backward_dropout_layer(dropout_layer *l, float *delta);
void resize_dropout_layer(dropout_layer *l, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer l, network net);
void backward_dropout_layer_gpu(dropout_layer l, network net);

#endif
#endif
