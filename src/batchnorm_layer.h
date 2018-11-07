#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "gemm.h"
#include "utils.h"
#include "blas.h"
#include "image.h"

#ifdef GPU
    #include "cuda.h"
    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#elif defined(OPENCL)
    #include "opencl.h"
#endif

typedef struct {
    int h, w, c, batch, subdivisions, inputs, outputs, out_h, out_w, out_c, test;
    float *biases, *bias_updates, *delta, *output;
    float *mean, *mean_delta, *variance, *variance_delta, *rolling_mean, *rolling_variance, *x, *x_norm, *scales, *scale_updates;
    float *mean_gpu, *mean_delta_gpu, *variance_gpu, *variance_delta_gpu, *rolling_mean_gpu, *rolling_variance_gpu, *x_gpu,
        *x_norm_gpu, *scales_gpu, *scale_updates_gpu;
    float *biases_gpu, *bias_updates_gpu, *delta_gpu, *output_gpu;
#ifdef OPENCL
    cl_mem mean_cl, mean_delta_cl, variance_cl, variance_delta_cl, rolling_mean_cl, rolling_variance_cl, x_cl,
        x_norm_cl, scales_cl, scale_updates_cl;
    cl_mem biases_cl, bias_updates_cl, delta_cl, output_cl;
#endif
#ifdef CUDNN
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
#endif
} batchnorm_layer;

batchnorm_layer *make_batchnorm_layer(int batch, int subdivisions, int w, int h, int c, int test);
void forward_batchnorm_layer(const batchnorm_layer *layer, float *input, int test);
void backward_batchnorm_layer(const batchnorm_layer *layer, float*delta, int test);
void update_batchnorm_layer(const batchnorm_layer *layer, float learning_rate, float momentum, float decay);
image get_batchnorm_image(const batchnorm_layer *layer);
void free_batchnorm_layer(void *input);

#ifdef GPU
void forward_batchnorm_layer_gpu(const batchnorm_layer *layer, float *input_gpu, int test);
void backward_batchnorm_layer_gpu(const batchnorm_layer *layer, float *delta_gpu, int test);
void pull_batchnorm_layer(const batchnorm_layer *l);
void push_batchnorm_layer(const batchnorm_layer *l);
void update_batchnorm_layer_gpu(const batchnorm_layer *layer, float learning_rate, float momentum, float decay);
#elif defined(OPENCL)
void forward_batchnorm_layer_cl(const batchnorm_layer *layer, cl_mem input_cl, int test);
void push_batchnorm_layer_cl(const batchnorm_layer *l);
#endif

#endif
