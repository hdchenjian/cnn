#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdlib.h>
#include <string.h>

#include "activations.h"
#include "gemm.h"
#include "utils.h"
#include "blas.h"

#ifdef GPU
    #include "cuda.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

typedef struct {
    int h, w, c, n, size, stride, batch, out_h, out_w;
    int batch_normalize;
    float *weights, *weight_updates, *biases, *bias_updates, *delta, *output;
    float *mean, *mean_delta, *variance, *variance_delta, *rolling_mean, *rolling_variance, *x;
    float *mean_gpu, *mean_delta_gpu, *variance_gpu, *variance_delta_gpu, *rolling_mean_gpu, *rolling_variance_gpu, *x_gpu;
    float *weights_gpu, *weight_updates_gpu, *biases_gpu, *bias_updates_gpu, *delta_gpu, *output_gpu;
    ACTIVATION activation;
} convolutional_layer;

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, int batch,
        ACTIVATION activation, size_t *workspace_size, int batch_normalize);
void forward_convolutional_layer(const convolutional_layer *layer, float *in, float *workspace, int test);
void backward_convolutional_layer(const convolutional_layer *layer, float *input, float *delta, float *workspace, int test);
void update_convolutional_layer(const convolutional_layer *layer, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_convolutional_layer_gpu(const convolutional_layer *layer, float *in, float *workspace, int test);
void backward_convolutional_layer_gpu(const convolutional_layer *layer, float *input, float *delta,
        float *workspace, int test);
void update_convolutional_layer_gpu(const convolutional_layer *layer, float learning_rate, float momentum, float decay);

void push_convolutional_layer(const convolutional_layer *layer);
void pull_convolutional_layer(const convolutional_layer *layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
#endif
#endif
