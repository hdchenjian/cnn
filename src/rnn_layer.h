
#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include "activations.h"
#include "connected_layer.h"

typedef struct{
    int inputs, outputs, batch, steps;
    float *output, *delta, *state, *prev_state;
    connected_layer *input_layer, *self_layer, *output_layer;
} rnn_layer;

rnn_layer *make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam);

void forward_rnn_layer(rnn_layer *l, network net);
void backward_rnn_layer(rnn_layer *l, network net);
void update_rnn_layer(rnn_layer *l);

#ifdef GPU
void forward_rnn_layer_gpu(rnn_layer *l, network net);
void backward_rnn_layer_gpu(rnn_layer *l, network net);
void update_rnn_layer_gpu(rnn_layer *l, update_args a);
void push_rnn_layer(rnn_layer *l);
void pull_rnn_layer(rnn_layer *l);
#endif

#endif

