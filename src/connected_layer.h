#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "activations.h"
#include "gemm.h"
#include "image.h"
#include "convolutional_layer.h"

#ifdef GPU
    #include "cuda.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

typedef struct{
    float bflop;
    int inputs, outputs, batch, steps, batch_normalize;
    int weight_normalize, bias_term;  // weight_normalize: default no normalize, bias_term: whether use bias, default use
    float *output, *delta;
    float *weights, *weight_updates, *biases, *bias_updates;
    float *weights_gpu, *weight_updates_gpu, *biases_gpu, *bias_updates_gpu, *delta_gpu, *output_gpu;
    ACTIVATION activation;
    float lr_mult, lr_decay_mult, bias_mult, bias_decay_mult;
    float *mean, *mean_delta, *variance, *variance_delta, *rolling_mean, *rolling_variance, *x, *x_norm, *scales, *scale_updates;
    float *mean_gpu, *mean_delta_gpu, *variance_gpu, *variance_delta_gpu, *rolling_mean_gpu, *rolling_variance_gpu, *x_gpu,
        *x_norm_gpu, *scales_gpu, *scale_updates_gpu;
} connected_layer;

connected_layer *make_connected_layer(int inputs, int outputs, int batch, int steps, ACTIVATION activation, int weight_normalize,
                                      int bias_term, float lr_mult, float lr_decay_mult, float bias_mult,
                                      float bias_decay_mult, int weight_filler, float sigma, int batch_normalize);
void free_connected_layer(void *input);
void forward_connected_layer(connected_layer *layer, float *input, int test);
void backward_connected_layer(connected_layer *layer, float *input, float *delta, int test, int keep_delta);
void update_connected_layer(connected_layer *layer, float learning_rate, float momentum, float decay);
image get_connected_image(const connected_layer *layer);

#ifdef GPU
void forward_connected_layer_gpu(connected_layer *layer, float *input, int test);
void backward_connected_layer_gpu(connected_layer *layer, float *input, float *delta, int test, int keep_delta);
void update_connected_layer_gpu(connected_layer *layer, float learning_rate, float momentum, float decay);
void push_connected_layer(const connected_layer *layer);
void pull_connected_layer(const connected_layer *layer);
#endif
#endif
