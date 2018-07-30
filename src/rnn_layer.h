
#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"
#include "connected_layer.h"

typedef struct{
    int inputs, outputs, batch, steps;
    float *output, *delta, *state, *prev_state;
    float *output_gpu, *delta_gpu, *state_gpu, *prev_state_gpu;
    connected_layer *input_layer, *self_layer, *output_layer;
} rnn_layer;

image get_rnn_image(const rnn_layer *layer);
rnn_layer *make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize);
void free_rnn_layer(void *input);
void forward_rnn_layer(const rnn_layer *l, float *input, int test);
void backward_rnn_layer(const rnn_layer *l, float *input, float *delta, int test);
void update_rnn_layer(const rnn_layer *l, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_rnn_layer_gpu(const rnn_layer *l, float *input, int test);
void backward_rnn_layer_gpu(const rnn_layer *l, float *input, float *delta, int test);
void update_rnn_layer_gpu(const rnn_layer *l, float learning_rate, float momentum, float decay);
void push_rnn_layer(const rnn_layer *l);
void pull_rnn_layer(const rnn_layer *l);
#endif

#endif

