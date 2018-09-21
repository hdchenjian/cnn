#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "activations.h"
#include "gemm.h"
#include "utils.h"
#include "blas.h"
#include "image.h"

#ifdef GPU
    #include "cuda.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

typedef struct {
    int h, w, c, n, size, stride, batch, subdivisions, outputs, out_h, out_w, batch_normalize, pad;
    float bflop, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult;
    float *weights, *weight_updates, *biases, *bias_updates, *delta, *output;
    float *mean, *mean_delta, *variance, *variance_delta, *rolling_mean, *rolling_variance, *x, *x_norm, *scales, *scale_updates;
    float *mean_gpu, *mean_delta_gpu, *variance_gpu, *variance_delta_gpu, *rolling_mean_gpu, *rolling_variance_gpu, *x_gpu,
        *x_norm_gpu, *scales_gpu, *scale_updates_gpu;
    float *weights_gpu, *weight_updates_gpu, *biases_gpu, *bias_updates_gpu, *delta_gpu, *output_gpu;
    ACTIVATION activation;
    float *bottom_data, *slope, *slope_updates;
    float *bottom_data_gpu, *slope_gpu, *slope_updates_gpu;
} convolutional_layer;

image get_convolutional_image(const convolutional_layer *layer);
convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, int batch,
                                              ACTIVATION activation, size_t *workspace_size, int batch_normalize, int pad,
                                              float lr_mult, float lr_decay_mult, float bias_mult, float bias_decay_mult,
                                              int weight_filler, float sigma, int subdivisions);
void free_convolutional_layer(void *input);
void forward_convolutional_layer(const convolutional_layer *layer, float *in, float *workspace, int test);
void backward_convolutional_layer(const convolutional_layer *layer, float *input, float *delta, float *workspace, int test);
void update_convolutional_layer(const convolutional_layer *layer, float learning_rate, float momentum, float decay);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters,
                         int spatial, float *variance_delta);

void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta,
                         int batch, int filters, int spatial, float *delta);

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);

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
