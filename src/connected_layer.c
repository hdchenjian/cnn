#include "connected_layer.h"
#include <float.h>

image get_connected_image(const connected_layer *layer)
{
    int h = 1;
    int w = 1;
    int c = layer->outputs;
    return float_to_image(h,w,c,NULL);
}

connected_layer *make_connected_layer(int inputs, int outputs, int batch, ACTIVATION activation,
                                      int weight_normalize, int bias_term, float lr_mult, float lr_decay_mult,
                                      float bias_mult, float bias_decay_mult, int weight_filler, float sigma,
                                      int batch_normalize, int subdivisions, int test)
{
    connected_layer *layer = calloc(1, sizeof(connected_layer));
    layer->bflop = (2.0F * inputs * outputs) / 1000000000.0F;
    fprintf(stderr, "Connected Layer:    %d inputs, %d outputs, %f BFLOPs\n", inputs, outputs, layer->bflop);
    layer->batch_normalize = batch_normalize;
    layer->lr_mult = lr_mult;
    layer->lr_decay_mult = lr_decay_mult;
    layer->bias_mult = bias_mult;
    layer->bias_decay_mult = bias_decay_mult;

    layer->weight_normalize = weight_normalize;
    layer->bias_term = bias_term;
    layer->inputs = inputs;
    layer->outputs = outputs;
    layer->subdivisions = subdivisions;
    layer->test = test;
    layer->batch = batch;
#ifndef FORWARD_GPU
    layer->output = calloc(batch*outputs, sizeof(float));
#endif

    layer->weights = calloc(inputs*outputs, sizeof(float));  // layer->outputs is the number of rows
    if(weight_filler == 1){   // xavier
        float scale = sqrtf(2.0F / inputs);
        for(int i = 0; i < inputs*outputs; ++i) layer->weights[i] = scale*rand_uniform(-1, 1);
    } else if(weight_filler == 2){   // gaussian
        for(int i = 0; i < inputs*outputs; ++i) layer->weights[i] = rand_normal_me(0, sigma);
    } else {
        fprintf(stderr, "weight_filler not support\n");
        exit(-1);
    }
    if(0 == layer->test){    // 0: train, 1: valid
        layer->delta = calloc(batch*outputs, sizeof(float));
        layer->weight_updates = calloc(inputs*outputs, sizeof(float));
        layer->bias_updates = calloc(outputs, sizeof(float));
    }
    layer->biases = calloc(outputs, sizeof(float));
    layer->activation = activation;
    if(layer->activation == PRELU){
        if(0 == layer->test){    // 0: train, 1: valid
            layer->bottom_data = calloc(batch * outputs, sizeof(float));
            layer->slope_updates = calloc(outputs, sizeof(float));
        }
        layer->slope = calloc(outputs, sizeof(float));
        for(int i = 0; i < outputs; i++) layer->slope[i] = 0.25F;
    }

    if(layer->batch_normalize){
        layer->scales = calloc(outputs, sizeof(float));
        layer->rolling_mean = calloc(outputs, sizeof(float));
        layer->rolling_variance = calloc(outputs, sizeof(float));
        if(0 == layer->test){    // 0: train, 1: valid
            layer->scale_updates = calloc(outputs, sizeof(float));
            for(int i = 0; i < outputs; ++i){
                layer->scales[i] = 1;
            }

            layer->mean = calloc(outputs, sizeof(float));
            layer->variance = calloc(outputs, sizeof(float));
            layer->mean_delta = calloc(outputs, sizeof(float));
            layer->variance_delta = calloc(outputs, sizeof(float));
            layer->x = calloc(batch * outputs, sizeof(float));
            layer->x_norm = calloc(batch * outputs, sizeof(float));
        }
    }
#ifdef GPU
    layer->biases_gpu = cuda_make_array(layer->biases, outputs);
    layer->weights_gpu = cuda_make_array(layer->weights, outputs*inputs);
    layer->output_gpu = cuda_make_array(layer->output, outputs*batch);
    if(0 == layer->test){    // 0: train, 1: valid
        layer->bias_updates_gpu = cuda_make_array(layer->bias_updates, outputs);
        layer->weight_updates_gpu = cuda_make_array(layer->weight_updates, outputs*inputs);
        layer->delta_gpu = cuda_make_array(layer->delta, outputs*batch);
    }

    if(layer->activation == PRELU){
        if(0 == layer->test){    // 0: train, 1: valid
            layer->bottom_data_gpu = cuda_make_array(layer->bottom_data, batch * outputs);
            layer->slope_updates_gpu = cuda_make_array(layer->slope_updates, outputs);
        }
        layer->slope_gpu = cuda_make_array(layer->slope, outputs);
    }

    if(layer->batch_normalize){
        layer->scales_gpu = cuda_make_array(layer->scales, outputs);
        layer->rolling_mean_gpu = cuda_make_array(layer->rolling_mean, outputs);
        layer->rolling_variance_gpu = cuda_make_array(layer->rolling_variance, outputs);
        if(0 == layer->test){    // 0: train, 1: valid
            layer->scale_updates_gpu = cuda_make_array(layer->scale_updates, outputs);
            layer->mean_gpu = cuda_make_array(layer->mean, outputs);
            layer->variance_gpu = cuda_make_array(layer->variance, outputs);
            layer->mean_delta_gpu = cuda_make_array(layer->mean_delta, outputs);
            layer->variance_delta_gpu = cuda_make_array(layer->variance_delta, outputs);
            layer->x_gpu = cuda_make_array(layer->output, batch * outputs);
            layer->x_norm_gpu = cuda_make_array(layer->output, batch * outputs);
        }
    }
#elif defined(OPENCL)
    layer->biases_cl = cl_make_array(layer->biases, outputs);
    layer->weights_cl = cl_make_array(layer->weights, outputs*inputs);
    layer->output_cl = cl_make_array(layer->output, outputs*batch);
    if(0 == layer->test){    // 0: train, 1: valid
        layer->bias_updates_cl = cl_make_array(layer->bias_updates, outputs);
        layer->weight_updates_cl = cl_make_array(layer->weight_updates, outputs*inputs);
        layer->delta_cl = cl_make_array(layer->delta, outputs*batch);
    }
    if(layer->activation == PRELU){
        if(0 == layer->test){    // 0: train, 1: valid
            layer->bottom_data_cl = cl_make_array(layer->bottom_data, batch * outputs);
            layer->slope_updates_cl = cl_make_array(layer->slope_updates, outputs);
        }
        layer->slope_cl = cl_make_array(layer->slope, outputs);
    }

    if(batch_normalize){
        layer->scales_cl = cl_make_array(layer->scales, outputs);
        layer->rolling_mean_cl = cl_make_array(layer->rolling_mean, outputs);
        layer->rolling_variance_cl = cl_make_array(layer->rolling_variance, outputs);
        if(0 == layer->test){    // 0: train, 1: valid
            layer->scale_updates_cl = cl_make_array(layer->scale_updates, outputs);
            layer->mean_cl = cl_make_array(layer->mean, outputs);
            layer->variance_cl = cl_make_array(layer->variance, outputs);
            layer->mean_delta_cl = cl_make_array(layer->mean_delta, outputs);
            layer->variance_delta_cl = cl_make_array(layer->variance_delta, outputs);
            layer->x_cl = cl_make_array(layer->output, layer->batch * outputs);
            layer->x_norm_cl = cl_make_array(layer->output, layer->batch * outputs);
        }
    }
#endif

    return layer;
}

void free_connected_layer(void *input)
{
    connected_layer *layer = (connected_layer *)input;
    if(layer->weights) free_ptr((void *)&(layer->weights));
    if(layer->weight_updates) free_ptr((void *)&(layer->weight_updates));
    if(layer->biases) free_ptr((void *)&(layer->biases));
    if(layer->bias_updates) free_ptr((void *)&(layer->bias_updates));
    if(layer->output) free_ptr((void *)&(layer->output));
    if(layer->delta) free_ptr((void *)&(layer->delta));
    if(layer->slope) free_ptr((void *)&(layer->slope));
    if(layer->scales) free_ptr((void *)&(layer->scales));
    if(layer->scale_updates) free_ptr((void *)&(layer->scale_updates));
    if(layer->mean) free_ptr((void *)&(layer->mean));
    if(layer->mean_delta) free_ptr((void *)&(layer->mean_delta));
    if(layer->variance) free_ptr((void *)&(layer->variance));
    if(layer->variance_delta) free_ptr((void *)&(layer->variance_delta));
    if(layer->rolling_mean) free_ptr((void *)&(layer->rolling_mean));
    if(layer->rolling_variance) free_ptr((void *)&(layer->rolling_variance));
    if(layer->x) free_ptr((void *)&(layer->x));
    if(layer->x_norm) free_ptr((void *)&(layer->x_norm));
#ifdef GPU
    if(layer->weights_gpu) cuda_free(layer->weights_gpu);
    if(layer->weight_updates_gpu) cuda_free(layer->weight_updates_gpu);
    if(layer->biases_gpu) cuda_free(layer->biases_gpu);
    if(layer->bias_updates_gpu) cuda_free(layer->bias_updates_gpu);
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
    if(layer->slope_gpu) cuda_free(layer->slope_gpu);
    if(layer->scales_gpu) cuda_free(layer->scales_gpu);
    if(layer->scale_updates_gpu) cuda_free(layer->scale_updates_gpu);
    if(layer->mean_gpu) cuda_free(layer->mean_gpu);
    if(layer->mean_delta_gpu) cuda_free(layer->mean_delta_gpu);
    if(layer->variance_gpu) cuda_free(layer->variance_gpu);
    if(layer->variance_delta_gpu) cuda_free(layer->variance_delta_gpu);
    if(layer->rolling_mean_gpu) cuda_free(layer->rolling_mean_gpu);
    if(layer->rolling_variance_gpu) cuda_free(layer->rolling_variance_gpu);
    if(layer->x_gpu) cuda_free(layer->x_gpu);
    if(layer->x_norm_gpu) cuda_free(layer->x_norm_gpu);
#elif defined(OPENCL)
    if(layer->weights_cl) clReleaseMemObject(layer->weights_cl);
    if(layer->weight_updates_cl) clReleaseMemObject(layer->weight_updates_cl);
    if(layer->biases_cl) clReleaseMemObject(layer->biases_cl);
    if(layer->bias_updates_cl) clReleaseMemObject(layer->bias_updates_cl);
    if(layer->output_cl) clReleaseMemObject(layer->output_cl);
    if(layer->delta_cl) clReleaseMemObject(layer->delta_cl);
    if(layer->scales_cl) clReleaseMemObject(layer->scales_cl);
    if(layer->scale_updates_cl) clReleaseMemObject(layer->scale_updates_cl);
    if(layer->mean_cl) clReleaseMemObject(layer->mean_cl);
    if(layer->mean_delta_cl) clReleaseMemObject(layer->mean_delta_cl);
    if(layer->variance_cl) clReleaseMemObject(layer->variance_cl);
    if(layer->variance_delta_cl) clReleaseMemObject(layer->variance_delta_cl);
    if(layer->rolling_mean_cl) clReleaseMemObject(layer->rolling_mean_cl);
    if(layer->rolling_variance_cl) clReleaseMemObject(layer->rolling_variance_cl);
    if(layer->x_cl) clReleaseMemObject(layer->x_cl);
    if(layer->x_norm_cl) clReleaseMemObject(layer->x_norm_cl);
#endif
    free_ptr((void *)&layer);
}

void forward_connected_batchnorm_layer(const connected_layer *layer, int test)
{
    if(0 == test){    // 0: train, 1: valid
        memcpy(layer->x, layer->output, layer->batch * layer->outputs * sizeof(float));
        mean_cpu(layer->output, layer->batch, layer->outputs, 1, layer->mean);
        variance_cpu(layer->output, layer->mean, layer->batch, layer->outputs, 1, layer->variance);

        scal_cpu(layer->outputs, .99, layer->rolling_mean, 1);
        axpy_cpu(layer->outputs, .01, layer->mean, 1, layer->rolling_mean, 1);
        scal_cpu(layer->outputs, .99, layer->rolling_variance, 1);
        axpy_cpu(layer->outputs, .01, layer->variance, 1, layer->rolling_variance, 1);

        normalize_cpu(layer->output, layer->mean, layer->variance, layer->batch, layer->outputs, 1);
        memcpy(layer->x_norm, layer->output, layer->batch * layer->outputs * sizeof(float));
        scale_bias(layer->output, layer->scales, layer->batch, layer->outputs, 1);
    } else {
        normalize_cpu(layer->output, layer->rolling_mean, layer->rolling_variance, layer->batch, layer->outputs, 1);
        scale_bias(layer->output, layer->scales, layer->batch, layer->outputs, 1);
    }
}

void activation_prelu_connect(const connected_layer *layer, int test){
    if(0 == test){    // 0: train, 1: valid
        memcpy(layer->bottom_data, layer->output, layer->batch * layer->outputs * sizeof(float));
    }
    int count = layer->outputs * layer->batch;
    for (int i = 0; i < count; ++i) {
        layer->output[i] = fmaxf(layer->output[i], 0.0F) + layer->slope[i % layer->outputs] * fminf(layer->output[i], 0.0F);
    }
}

void forward_connected_layer(connected_layer *layer, float *input, int test)
{
    if(layer->weight_normalize && 0 == test){         // 0: train, 1: valid
        for(int i = 0; i < layer->outputs; i++){
            float sum = 1e-6;
            for(int j = 0; j < layer->inputs; j++){
                float temp = layer->weights[i * layer->inputs + j];
                sum += temp * temp;
            }
            float scale = sqrtf(sum);
            for(int j = 0; j < layer->inputs; j++){
                layer->weights[i * layer->inputs + j] /= scale;
            }
        }
    }

    float *a = input;
    float *b = layer->weights;  // layer->outputs is the number of rows
    float *c = layer->output;
    int m = layer->batch;
    int n = layer->outputs;
    int k = layer->inputs;
#if defined QML || defined INTEL_MKL || defined OPENBLAS_ARM || defined ARM_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, a, k, b, k, 0, c, n);
#else
    gemm(0, 1, m, n, k, 1, a, k, b, k, 0, c, n);
#endif

    if(layer->batch_normalize){
        forward_connected_batchnorm_layer(layer, test);
    }
    if(layer->bias_term){
        for(int b = 0; b < layer->batch; ++b){
            for(int j = 0; j < layer->outputs; ++j){
                layer->output[b*layer->outputs + j] += layer->biases[j];
            }
        }
    }

    int all_outputs = layer->outputs * layer->batch;
    if(layer->activation == PRELU){
        activation_prelu_connect(layer, test);
    } else if (layer->activation == LINEAR) {
    } else {
        for(int i = 0; i < all_outputs; ++i){
            layer->output[i] = activate(layer->output[i], layer->activation);
        }
    }
    //for(int i = 0; i < 10; ++i) printf("%d : %f\n", i, layer->output[i]);printf("\n");

    /*
    float max = -FLT_MAX, min = FLT_MAX;
    for(int i = 0; i < layer->batch * layer->outputs; ++i){
    	if(layer->output[i] > max) max = layer->output[i];
    	if(layer->output[i] < min) min = layer->output[i];
    }
    printf("forward_connected_layer max: %f, min: %f\n", max, min);*/
}

void update_connected_layer(connected_layer *layer, float learning_rate, float momentum, float decay)
{
    int batch = layer->subdivisions * layer->batch;
    for(int i = 0; i < layer->outputs; i++){
        layer->bias_updates[i] += -decay * layer->bias_decay_mult * batch * layer->biases[i];
        layer->biases[i] += learning_rate * layer->bias_mult / batch * layer->bias_updates[i];
        layer->bias_updates[i] *= momentum;
    }

    int size = layer->inputs*layer->outputs;
    for(int i = 0; i < size; i++){
        layer->weight_updates[i] += -decay * layer->lr_decay_mult * batch *layer->weights[i];
        layer->weights[i] += learning_rate * layer->lr_mult / batch * layer->weight_updates[i];
        layer->weight_updates[i] *= momentum;
    }

    if(layer->batch_normalize){
        for(int i = 0; i < layer->outputs; i++){
            layer->scales[i] += learning_rate / batch* layer->scale_updates[i];
            layer->scale_updates[i] *= momentum;
        }
    }
    if(layer->activation == PRELU){
        for(int i = 0; i < layer->outputs; i++){
            layer->slope[i] += learning_rate / batch * layer->slope_updates[i];
            layer->slope_updates[i] *= momentum;
        }
    }

}

void backward_connected_batchnorm_layer(const connected_layer *layer, int test)
{
    if(0 != test){    // 0: train, 1: valid
        fprintf(stderr, "backward_connected_batchnorm_layer: use no used!\n");
        exit(-1);
        //layer->mean = layer->rolling_mean;
        //layer->variance = layer->rolling_variance;
    }
    backward_scale_cpu(layer->x_norm, layer->delta, layer->batch, layer->outputs, 1, layer->scale_updates);
    scale_bias(layer->delta, layer->scales, layer->batch, layer->outputs, 1);

    mean_delta_cpu(layer->delta, layer->variance, layer->batch, layer->outputs, 1, layer->mean_delta);
    variance_delta_cpu(layer->x, layer->delta, layer->mean, layer->variance, layer->batch, layer->outputs,
                       1, layer->variance_delta);
    normalize_delta_cpu(layer->x, layer->mean, layer->variance, layer->mean_delta, layer->variance_delta,
                        layer->batch, layer->outputs, 1, layer->delta);
}

void backward_connected_layer(connected_layer *layer, float *input, float *delta, int test)
{
    int all_outputs = layer->outputs * layer->batch;
    if(layer->activation == PRELU){
        for (int i = 0; i < all_outputs; ++i) {
            layer->slope_updates[i] += layer->delta[i] * layer->bottom_data[i] * (layer->bottom_data[i] <= 0);
            layer->delta[i] = layer->delta[i] * ((layer->bottom_data[i] > 0) + layer->slope[i] * (layer->bottom_data[i] <= 0));
        }
    } else if (layer->activation == LINEAR) {
    } else {
        for(int i = 0; i < all_outputs; ++i){
            layer->delta[i] *= gradient(layer->output[i], layer->activation);
        }
    }

    for(int i = 0; i < layer->batch; ++i){
        for(int j = 0; j < layer->outputs; ++j){
            layer->bias_updates[j] += (layer->delta + i * layer->outputs)[j];
        }
    }
    if(layer->batch_normalize){
        backward_connected_batchnorm_layer(layer, test);
    }

    int m = layer->outputs;
    int n = layer->inputs;
    int k = layer->batch;
    float *a = layer->delta;
    float *b = input;
    float *c = layer->weight_updates;  // layer->outputs is the number of rows
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    if(delta) {
        m = layer->batch;
        n = layer->inputs;
        k = layer->outputs;
        a = layer->delta;
        b = layer->weights;
        c = delta;
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }
}

#ifdef GPU

void update_connected_layer_gpu(connected_layer *layer, float learning_rate, float momentum, float decay)
{
    int batch = layer->subdivisions * layer->batch;
    axpy_gpu(layer->outputs, -decay * layer->bias_decay_mult *batch, layer->biases_gpu, 1, layer->bias_updates_gpu, 1);
    axpy_gpu(layer->outputs, learning_rate * layer->bias_mult /batch, layer->bias_updates_gpu, 1, layer->biases_gpu, 1);
    scal_gpu(layer->outputs, momentum, layer->bias_updates_gpu, 1);

    axpy_gpu(layer->inputs*layer->outputs, -decay * layer->lr_decay_mult * batch, layer->weights_gpu, 1, layer->weight_updates_gpu, 1);
    axpy_gpu(layer->inputs*layer->outputs, learning_rate * layer->lr_mult / batch, layer->weight_updates_gpu, 1, layer->weights_gpu, 1);
    scal_gpu(layer->inputs*layer->outputs, momentum, layer->weight_updates_gpu, 1);

    if(layer->batch_normalize){
        axpy_gpu(layer->outputs, learning_rate/batch, layer->scale_updates_gpu, 1, layer->scales_gpu, 1);
        scal_gpu(layer->outputs, momentum, layer->scale_updates_gpu, 1);
    }
    if(layer->activation == PRELU){
        axpy_gpu(layer->outputs, learning_rate/batch, layer->slope_updates_gpu, 1, layer->slope_gpu, 1);
        scal_gpu(layer->outputs, momentum, layer->slope_updates_gpu, 1);
    }

}

void forward_connected_batchnorm_layer_gpu(const connected_layer *layer, int test)
{
    if(0 == test){    // 0: train, 1: valid
        copy_gpu(layer->batch * layer->outputs, layer->output_gpu, 1, layer->x_gpu, 1);
        fast_mean_gpu(layer->output_gpu, layer->batch, layer->outputs, 1, layer->mean_gpu);
        fast_variance_gpu(layer->output_gpu, layer->mean_gpu, layer->batch, layer->outputs, 1,
                          layer->variance_gpu);

        scal_gpu(layer->outputs, .99, layer->rolling_mean_gpu, 1);
        axpy_gpu(layer->outputs, .01, layer->mean_gpu, 1, layer->rolling_mean_gpu, 1);
        scal_gpu(layer->outputs, .99, layer->rolling_variance_gpu, 1);
        axpy_gpu(layer->outputs, .01, layer->variance_gpu, 1, layer->rolling_variance_gpu, 1);

        normalize_gpu(layer->output_gpu, layer->mean_gpu, layer->variance_gpu, layer->batch, layer->outputs, 1);
        copy_gpu(layer->batch * layer->outputs, layer->output_gpu, 1, layer->x_norm_gpu, 1);
        scale_bias_gpu(layer->output_gpu, layer->scales_gpu, layer->batch, layer->outputs, 1);
    } else {
        normalize_gpu(layer->output_gpu, layer->rolling_mean_gpu, layer->rolling_variance_gpu,
                      layer->batch, layer->outputs, 1);
        scale_bias_gpu(layer->output_gpu, layer->scales_gpu, layer->batch, layer->outputs, 1);
    }
}

void forward_connected_layer_gpu(connected_layer *layer, float *input, int test)
{
    if(layer->weight_normalize && 0 == test){         // 0: train, 1: valid
        weight_normalize_gpu(layer->inputs, layer->outputs, layer->weights_gpu);
    }

    int m = layer->batch;
    int n = layer->outputs;
    int k = layer->inputs;
    float *a = input;
    float *b = layer->weights_gpu;
    float *c = layer->output_gpu;
    gemm_gpu(0, 1, m, n, k, 1, a, k, b, k, 0, c, n);
    if(layer->batch_normalize){
        forward_connected_batchnorm_layer_gpu(layer, test);
    }
    if(layer->bias_term){
        add_bias_gpu(layer->output_gpu, layer->biases_gpu, layer->batch, layer->outputs, 1);
    }
    if(layer->activation == PRELU){
        if(0 == test){    // 0: train, 1: valid
            copy_gpu(layer->batch * layer->outputs, layer->output_gpu, 1, layer->bottom_data_gpu, 1);
        }
        int size = layer->batch * layer->outputs;
        int dim = 1;
        activate_prelu_array_gpu(layer->output_gpu, layer->slope_gpu, size, layer->outputs, dim);
    } else if (layer->activation == LINEAR) {
    } else {
        activate_array_gpu(layer->output_gpu, layer->outputs*layer->batch, layer->activation);
    }
}

void backward_connected_batchnorm_layer_gpu(const connected_layer *layer, int test)
{
    if(0 != test){    // 0: train, 1: valid
        fprintf(stderr, "backward_connected_batchnorm_layer_gpu: use no used!\n");
        exit(-1);
    }
    backward_scale_gpu(layer->x_norm_gpu, layer->delta_gpu, layer->batch, layer->outputs, 1,
                       layer->scale_updates_gpu);
    scale_bias_gpu(layer->delta_gpu, layer->scales_gpu, layer->batch, layer->outputs, 1);

    fast_mean_delta_gpu(layer->delta_gpu, layer->variance_gpu, layer->batch, layer->outputs, 1,
                        layer->mean_delta_gpu);
    fast_variance_delta_gpu(layer->x_gpu, layer->delta_gpu, layer->mean_gpu, layer->variance_gpu,
                            layer->batch, layer->outputs, 1, layer->variance_delta_gpu);
    normalize_delta_gpu(layer->x_gpu, layer->mean_gpu, layer->variance_gpu, layer->mean_delta_gpu,
                        layer->variance_delta_gpu, layer->batch, layer->outputs, 1, layer->delta_gpu);
}

void backward_connected_layer_gpu(connected_layer *layer, float *input, float *delta, int test)
{
    if(layer->activation == PRELU){
        int size = layer->batch * layer->outputs;
        int dim = 1;
        backward_prelu_slope_gpu(layer->delta_gpu, layer->bottom_data_gpu,
                                 layer->slope_updates_gpu, layer->outputs, dim, layer->batch);
        gradient_prelu_array_gpu(layer->delta_gpu, layer->bottom_data_gpu, layer->slope_gpu, size, layer->outputs, dim);
    } else if (layer->activation == LINEAR) {
    } else {
        gradient_array_gpu(layer->output_gpu, layer->outputs*layer->batch, layer->activation, layer->delta_gpu);
    }
    backward_bias_gpu(layer->bias_updates_gpu, layer->delta_gpu, layer->batch, layer->outputs, 1);
    if(layer->batch_normalize){
        backward_connected_batchnorm_layer_gpu(layer, test);
    }
    int m = layer->outputs;
    int n = layer->inputs;
    int k = layer->batch;
    float * a = layer->delta_gpu;
    float * b = input;
    float * c = layer->weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = layer->batch;
    n = layer->inputs;
    k = layer->outputs;
    a = layer->delta_gpu;
    b = layer->weights_gpu;
    c = delta;
    if(c) {
        gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }
}

void pull_connected_layer(const connected_layer *layer)
{
    cuda_pull_array(layer->weights_gpu, layer->weights, layer->inputs*layer->outputs);
    cuda_pull_array(layer->biases_gpu, layer->biases, layer->outputs);
    cuda_pull_array(layer->weight_updates_gpu, layer->weight_updates, layer->inputs*layer->outputs);
    cuda_pull_array(layer->bias_updates_gpu, layer->bias_updates, layer->outputs);
    if (layer->batch_normalize){
        cuda_pull_array(layer->scales_gpu, layer->scales, layer->outputs);
        cuda_pull_array(layer->rolling_mean_gpu, layer->rolling_mean, layer->outputs);
        cuda_pull_array(layer->rolling_variance_gpu, layer->rolling_variance, layer->outputs);
    }
    if(layer->activation == PRELU){
        cuda_pull_array(layer->slope_gpu, layer->slope, layer->outputs);
    }
}

void push_connected_layer(const connected_layer *layer)
{
    cuda_push_array(layer->weights_gpu, layer->weights, layer->inputs*layer->outputs);
    cuda_push_array(layer->biases_gpu, layer->biases, layer->outputs);
    if(0 == layer->test){    // 0: train, 1: valid
        cuda_push_array(layer->weight_updates_gpu, layer->weight_updates, layer->inputs*layer->outputs);
        cuda_push_array(layer->bias_updates_gpu, layer->bias_updates, layer->outputs);
    }
    if (layer->batch_normalize){
        cuda_push_array(layer->scales_gpu, layer->scales, layer->outputs);
        cuda_push_array(layer->rolling_mean_gpu, layer->rolling_mean, layer->outputs);
        cuda_push_array(layer->rolling_variance_gpu, layer->rolling_variance, layer->outputs);
    }
    if(layer->activation == PRELU){
        cuda_push_array(layer->slope_gpu, layer->slope, layer->outputs);
    }
}

#elif defined(OPENCL)
void forward_connected_batchnorm_layer_cl(const connected_layer *layer, int test)
{
    if(0 == test){    // 0: train, 1: valid
        copy_cl(layer->batch * layer->outputs, layer->output_cl, 1, layer->x_cl, 1);
        fast_mean_cl(layer->output_cl, layer->batch, layer->outputs, 1, layer->mean_cl);
        fast_variance_cl(layer->output_cl, layer->mean_cl, layer->batch, layer->outputs, 1,
                          layer->variance_cl);

        scal_cl(layer->outputs, .99, layer->rolling_mean_cl, 1);
        axpy_cl(layer->outputs, .01, layer->mean_cl, 1, layer->rolling_mean_cl, 1);
        scal_cl(layer->outputs, .99, layer->rolling_variance_cl, 1);
        axpy_cl(layer->outputs, .01, layer->variance_cl, 1, layer->rolling_variance_cl, 1);

        normalize_cl(layer->output_cl, layer->mean_cl, layer->variance_cl, layer->batch, layer->outputs, 1);
        copy_cl(layer->batch * layer->outputs, layer->output_cl, 1, layer->x_norm_cl, 1);
        scale_bias_cl(layer->output_cl, layer->scales_cl, layer->batch, layer->outputs, 1);
    } else {
        normalize_cl(layer->output_cl, layer->rolling_mean_cl, layer->rolling_variance_cl,
                      layer->batch, layer->outputs, 1);
        scale_bias_cl(layer->output_cl, layer->scales_cl, layer->batch, layer->outputs, 1);
    }
}

void forward_connected_layer_cl(connected_layer *layer, cl_mem input, int test)
{
    if(layer->weight_normalize && 0 == test){         // 0: train, 1: valid
        //weight_normalize_gpu(layer->inputs, layer->outputs, layer->weights_gpu);
    }

    int m = layer->batch;
    int n = layer->outputs;
    int k = layer->inputs;
    cl_mem a = input;
    cl_mem b = layer->weights_cl;
    cl_mem c = layer->output_cl;
    gemm_cl(0, 1, m, n, k, 1, a, 0, k, b, 0, k, 0, c, 0, n);
    if(layer->batch_normalize){
        forward_connected_batchnorm_layer_cl(layer, test);
    }
    if(layer->bias_term){
        add_bias_cl(layer->batch, 1, layer->outputs, layer->biases_cl, layer->output_cl);
    }
    if(layer->activation == PRELU){
        if(0 == layer->test){    // 0: train, 1: valid
            copy_cl(layer->batch * layer->outputs, layer->output_cl, 1, layer->bottom_data_cl, 1);
        }
        activate_prelu_array_cl(layer->output_cl, layer->slope_cl, layer->batch, layer->outputs, 1);
    } else if (layer->activation == LINEAR) {
    } else {
        activate_array_cl(layer->output_cl, layer->outputs*layer->batch, layer->activation);
    }

}

void push_connected_layer_cl(const connected_layer *layer)
{
    int size = layer->inputs*layer->outputs;
    cl_write_array(layer->weights_cl, layer->weights, size);
    cl_write_array(layer->biases_cl, layer->biases, layer->outputs);
    //cl_write_array(layer->weight_updates_cl, layer->weight_updates, size);
    //cl_write_array(layer->bias_updates_cl, layer->bias_updates, layer->outputs);
    if (layer->batch_normalize){
        cl_write_array(layer->scales_cl, layer->scales, layer->outputs);
        //cl_write_array(layer->mean_cl, layer->mean, layer->outputs);
        //cl_write_array(layer->variance_cl, layer->variance, layer->outputs);
        cl_write_array(layer->rolling_mean_cl, layer->rolling_mean, layer->outputs);
        cl_write_array(layer->rolling_variance_cl, layer->rolling_variance, layer->outputs);
    }
    if(layer->activation == PRELU){
        cl_write_array(layer->slope_cl, layer->slope, layer->outputs);
    }
}
#endif
