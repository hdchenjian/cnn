#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "activations.h"
#include "gemm.h"

#ifdef GPU
    #include "cuda.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

typedef struct{
    int inputs, outputs, batch;
    int weight_normalize, bias_term;  // weight_normalize: default no normalize, bias_term: whether use bias, default use
    float *output, *delta;
    float *weights, *weight_updates, *biases, *bias_updates;
    float *weights_gpu, *weight_updates_gpu, *biases_gpu, *bias_updates_gpu, *delta_gpu, *output_gpu;
    ACTIVATION activation;
} connected_layer;

connected_layer *make_connected_layer(int inputs, int outputs, int batch, ACTIVATION activation,
                                      int weight_normalize, int bias_term);
void forward_connected_layer(connected_layer *layer, float *input);
void backward_connected_layer(connected_layer *layer, float *input, float *delta);
void update_connected_layer(connected_layer *layer, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_connected_layer_gpu(connected_layer *layer, float *input);
void backward_connected_layer_gpu(connected_layer *layer, float *input, float *delta);
void update_connected_layer_gpu(connected_layer *layer, float learning_rate, float momentum, float decay);
void push_connected_layer(const connected_layer *layer);
void pull_connected_layer(const connected_layer *layer);
#endif
#endif
