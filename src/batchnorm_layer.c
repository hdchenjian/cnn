#include "batchnorm_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>

image get_batchnorm_image(const batchnorm_layer *layer)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->out_c;
    return float_to_image(h,w,c,NULL);
}

batchnorm_layer *make_batchnorm_layer(int batch, int subdivisions, int w, int h, int c, int test)
{
    fprintf(stderr, "BatchNorm:          %d x %d x %d\n", w,h,c);
    batchnorm_layer *l = calloc(1, sizeof(batchnorm_layer));
    l->batch = batch;
    l->subdivisions = subdivisions;
    l->test = test;
    l->h = l->out_h = h;
    l->w = l->out_w = w;
    l->c = l->out_c = c;
#ifndef FORWARD_GPU
    l->output = calloc(h * w * c * batch, sizeof(float));
#endif

    l->inputs = w*h*c;
    l->outputs = l->inputs;
    l->scales = calloc(c, sizeof(float));
    for(int i = 0; i < c; ++i) l->scales[i] = 1;
    l->biases = calloc(c, sizeof(float));
    l->rolling_mean = calloc(c, sizeof(float));
    l->rolling_variance = calloc(c, sizeof(float));

    if(0 == l->test){    // 0: train, 1: valid
        l->delta  = calloc(h * w * c * batch, sizeof(float));
        l->scale_updates = calloc(c, sizeof(float));
        l->bias_updates = calloc(c, sizeof(float));
        l->mean = calloc(c, sizeof(float));
        l->variance = calloc(c, sizeof(float));
        l->mean_delta = calloc(c, sizeof(float));
        l->variance_delta = calloc(c, sizeof(float));
        l->x = calloc(batch * l->out_h * l->out_w * l->out_c, sizeof(float));
        l->x_norm = calloc(batch * l->out_h * l->out_w * l->out_c, sizeof(float));
    }
#ifdef GPU
    l->output_gpu = cuda_make_array(l->output, h * w * c * batch);
    l->scales_gpu = cuda_make_array(l->scales, c);
    l->biases_gpu = cuda_make_array(l->biases, c);
    l->rolling_mean_gpu = cuda_make_array(l->mean, c);
    l->rolling_variance_gpu = cuda_make_array(l->variance, c);
    if(0 == l->test){    // 0: train, 1: valid
        l->delta_gpu = cuda_make_array(l->delta, h * w * c * batch);
        l->bias_updates_gpu = cuda_make_array(l->bias_updates, c);
        l->scale_updates_gpu = cuda_make_array(l->scale_updates, c);
        l->mean_gpu = cuda_make_array(l->mean, c);
        l->variance_gpu = cuda_make_array(l->variance, c);
        l->mean_delta_gpu = cuda_make_array(l->mean, c);
        l->variance_delta_gpu = cuda_make_array(l->variance, c);
        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
    #ifdef CUDNN
    cudnnCreateTensorDescriptor(&l->normTensorDesc);
    cudnnCreateTensorDescriptor(&l->dstTensorDesc);
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    #endif
#elif defined(OPENCL)
    //l->output_cl = cl_make_share_array(l->output, batch * l->out_h * l->out_w * c);
    l->output_cl = cl_make_array(l->output, batch * l->out_h * l->out_w * c);
    l->scales_cl = cl_make_array(l->scales, c);
    l->biases_cl = cl_make_array(l->biases, c);
    l->rolling_mean_cl = cl_make_array(l->mean, c);
    l->rolling_variance_cl = cl_make_array(l->variance, c);
    if(0 == l->test){    // 0: train, 1: valid
        l->bias_updates_cl = cl_make_array(l->bias_updates, c);
        l->delta_cl = cl_make_array(l->delta, batch * l->out_h * l->out_w * c);
        l->scale_updates_cl = cl_make_array(l->scale_updates, c);

        l->mean_cl = cl_make_array(l->mean, c);
        l->mean_delta_cl = cl_make_array(l->mean, c);
        l->variance_delta_cl = cl_make_array(l->variance, c);
        l->variance_cl = cl_make_array(l->variance, c);
        l->x_cl = cl_make_array(l->output, l->batch * l->out_h * l->out_w * c);
        l->x_norm_cl = cl_make_array(l->output, l->batch * l->out_h * l->out_w * c);
    }
#endif
    return l;
}

void free_batchnorm_layer(void *input)
{
    batchnorm_layer *layer = (batchnorm_layer *)input;
    if(layer->biases) free_ptr((void *)&(layer->biases));
    if(layer->bias_updates) free_ptr((void *)&(layer->bias_updates));
    if(layer->output) free_ptr((void *)&(layer->output));
    if(layer->delta) free_ptr((void *)&(layer->delta));
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
    if(layer->biases_gpu) cuda_free(layer->biases_gpu);
    if(layer->bias_updates_gpu) cuda_free(layer->bias_updates_gpu);
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
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
#ifdef CUDNN
    cudnnDestroyTensorDescriptor(layer->normTensorDesc);
    cudnnDestroyTensorDescriptor(layer->dstTensorDesc);
#endif
#elif defined(OPENCL)
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

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_batchnorm_layer(const batchnorm_layer *layer, float *input, int test)
{
    copy_cpu(layer->outputs*layer->batch, input, 1, layer->output, 1);
    if(0 == test){    // 0: train, 1: valid
        memcpy(layer->x, layer->output, layer->batch * layer->out_h * layer->out_w * layer->c * sizeof(float));
        mean_cpu(layer->output, layer->batch, layer->c, layer->out_h*layer->out_w, layer->mean);
        variance_cpu(layer->output, layer->mean, layer->batch, layer->c, layer->out_h*layer->out_w, layer->variance);

        scal_cpu(layer->c, .99, layer->rolling_mean, 1);
        axpy_cpu(layer->c, .01, layer->mean, 1, layer->rolling_mean, 1);
        scal_cpu(layer->c, .99, layer->rolling_variance, 1);
        axpy_cpu(layer->c, .01, layer->variance, 1, layer->rolling_variance, 1);

        normalize_cpu(layer->output, layer->mean, layer->variance, layer->batch, layer->c, layer->out_h*layer->out_w);
        memcpy(layer->x_norm, layer->output, layer->batch * layer->out_h * layer->out_w * layer->c * sizeof(float));
        scale_bias(layer->output, layer->scales, layer->batch, layer->c, layer->out_h*layer->out_w);
        add_bias(layer->output, layer->biases, layer->batch, layer->out_c, layer->out_h*layer->out_w);
    } else {
        normalize_cpu(layer->output, layer->rolling_mean, layer->rolling_variance,
                      layer->batch, layer->c, layer->out_h*layer->out_w);
        scale_bias(layer->output, layer->scales, layer->batch, layer->c, layer->out_h*layer->out_w);
        add_bias(layer->output, layer->biases, layer->batch, layer->out_c, layer->out_h*layer->out_w);
    }
}

void backward_batchnorm_layer(const batchnorm_layer *layer, float* delta, int test)
{
    if(0 != test){    // 0: train, 1: valid
        fprintf(stderr, "backward_batchnorm_layer: use no used!\n");
        exit(-1);
        //layer->mean = layer->rolling_mean;
        //layer->variance = layer->rolling_variance;
    }
    backward_bias(layer->bias_updates, layer->delta, layer->batch, layer->out_c, layer->out_w*layer->out_h);
    backward_scale_cpu(layer->x_norm, layer->delta, layer->batch, layer->c, layer->out_w*layer->out_h, layer->scale_updates);
    scale_bias(layer->delta, layer->scales, layer->batch, layer->c, layer->out_h*layer->out_w);

    mean_delta_cpu(layer->delta, layer->variance, layer->batch, layer->c, layer->out_w*layer->out_h, layer->mean_delta);
    variance_delta_cpu(layer->x, layer->delta, layer->mean, layer->variance, layer->batch, layer->c,
                       layer->out_w*layer->out_h, layer->variance_delta);
    normalize_delta_cpu(layer->x, layer->mean, layer->variance, layer->mean_delta, layer->variance_delta,
                        layer->batch, layer->c, layer->out_w*layer->out_h, layer->delta);
    copy_cpu(layer->outputs*layer->batch, layer->delta, 1, delta, 1);
}

void update_batchnorm_layer(const batchnorm_layer *layer, float learning_rate, float momentum, float decay)
{
    int batch = layer->subdivisions * layer->batch;
    for(int i = 0; i < layer->c; i ++){
        layer->scales[i] += learning_rate / batch * layer->scale_updates[i];
        layer->scale_updates[i] *= momentum;
    }

    for(int i = 0; i < layer->c; i ++){
        layer->biases[i] += learning_rate / batch * layer->bias_updates[i];
        layer->bias_updates[i] *= momentum;
    }

}

#ifdef GPU

void pull_batchnorm_layer(const batchnorm_layer *l)
{
    cuda_pull_array(l->biases_gpu, l->biases, l->c);
    cuda_pull_array(l->scales_gpu, l->scales, l->c);
    cuda_pull_array(l->rolling_mean_gpu, l->rolling_mean, l->c);
    cuda_pull_array(l->rolling_variance_gpu, l->rolling_variance, l->c);
}
void push_batchnorm_layer(const batchnorm_layer *l)
{
    cuda_push_array(l->biases_gpu, l->biases, l->c);
    cuda_push_array(l->scales_gpu, l->scales, l->c);
    cuda_push_array(l->rolling_mean_gpu, l->rolling_mean, l->c);
    cuda_push_array(l->rolling_variance_gpu, l->rolling_variance, l->c);
}

void forward_batchnorm_layer_gpu(const batchnorm_layer *layer, float *input_gpu, int test)
{
    copy_gpu(layer->outputs*layer->batch, input_gpu, 1, layer->output_gpu, 1);
    if(0 == test){    // 0: train, 1: valid
        copy_gpu(layer->batch * layer->out_h * layer->out_w * layer->c, layer->output_gpu, 1, layer->x_gpu, 1);
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, &one, &zero,
                                               layer->dstTensorDesc, layer->x_gpu,
                                               layer->dstTensorDesc, layer->output_gpu,
                                               layer->normTensorDesc,
                                               layer->scales_gpu, layer->biases_gpu,
                                               .01, layer->rolling_mean_gpu, layer->rolling_variance_gpu,
                                               .00001, layer->mean_gpu, layer->variance_gpu);
#else
        fast_mean_gpu(layer->output_gpu, layer->batch, layer->c, layer->out_h*layer->out_w, layer->mean_gpu);
        fast_variance_gpu(layer->output_gpu, layer->mean_gpu, layer->batch, layer->c, layer->out_h*layer->out_w,
                          layer->variance_gpu);

        scal_gpu(layer->c, .99, layer->rolling_mean_gpu, 1);
        axpy_gpu(layer->c, .01, layer->mean_gpu, 1, layer->rolling_mean_gpu, 1);
        scal_gpu(layer->c, .99, layer->rolling_variance_gpu, 1);
        axpy_gpu(layer->c, .01, layer->variance_gpu, 1, layer->rolling_variance_gpu, 1);

        normalize_gpu(layer->output_gpu, layer->mean_gpu, layer->variance_gpu, layer->batch, layer->c,
                      layer->out_h*layer->out_w);
        copy_gpu(layer->batch * layer->out_h * layer->out_w * layer->c, layer->output_gpu, 1, layer->x_norm_gpu, 1);
        scale_bias_gpu(layer->output_gpu, layer->scales_gpu, layer->batch, layer->c, layer->out_h*layer->out_w);
        add_bias_gpu(layer->output_gpu, layer->biases_gpu, layer->batch, layer->c, layer->out_w*layer->out_h);
#endif
    } else {
        normalize_gpu(layer->output_gpu, layer->rolling_mean_gpu, layer->rolling_variance_gpu,
                      layer->batch, layer->c, layer->out_h*layer->out_w);
        scale_bias_gpu(layer->output_gpu, layer->scales_gpu, layer->batch, layer->c, layer->out_h*layer->out_w);
        add_bias_gpu(layer->output_gpu, layer->biases_gpu, layer->batch, layer->c, layer->out_w*layer->out_h);
    }
}

void backward_batchnorm_layer_gpu(const batchnorm_layer *layer, float *delta_gpu, int test)
{
    if(0 != test){    // 0: train, 1: valid
        fprintf(stderr, "backward_conv_batchnorm_layer: use no used!\n");
        exit(-1);
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(), CUDNN_BATCHNORM_SPATIAL,
                                    &one, &zero, &one, &one,
                                    layer->dstTensorDesc, layer->x_gpu,
                                    layer->dstTensorDesc, layer->delta_gpu,
                                    layer->dstTensorDesc, layer->x_norm_gpu,
                                    layer->normTensorDesc,
                                    layer->scales_gpu, layer->scale_updates_gpu, layer->bias_updates_gpu,
                                    .00001, layer->mean_gpu, layer->variance_gpu);
    copy_gpu(layer->out_h * layer->out_w * layer->c*layer->batch, layer->x_norm_gpu, 1, layer->delta_gpu, 1);
#else
    backward_bias_gpu(layer->bias_updates_gpu, layer->delta_gpu, layer->batch, layer->c, layer->out_w*layer->out_h);
    backward_scale_gpu(layer->x_norm_gpu, layer->delta_gpu, layer->batch, layer->c, layer->out_w*layer->out_h,
                       layer->scale_updates_gpu);
    scale_bias_gpu(layer->delta_gpu, layer->scales_gpu, layer->batch, layer->c, layer->out_h*layer->out_w);

    fast_mean_delta_gpu(layer->delta_gpu, layer->variance_gpu, layer->batch, layer->c, layer->out_w*layer->out_h,
                        layer->mean_delta_gpu);
    fast_variance_delta_gpu(layer->x_gpu, layer->delta_gpu, layer->mean_gpu, layer->variance_gpu,
                            layer->batch, layer->c, layer->out_w*layer->out_h, layer->variance_delta_gpu);
    normalize_delta_gpu(layer->x_gpu, layer->mean_gpu, layer->variance_gpu, layer->mean_delta_gpu,
                        layer->variance_delta_gpu, layer->batch, layer->c, layer->out_w*layer->out_h, layer->delta_gpu);
#endif
    copy_gpu(layer->outputs*layer->batch, layer->delta_gpu, 1, delta_gpu, 1);
}

void update_batchnorm_layer_gpu(const batchnorm_layer *layer, float learning_rate, float momentum, float decay)
{
    int batch = layer->subdivisions * layer->batch;
    axpy_gpu(layer->c, learning_rate  / batch, layer->bias_updates_gpu, 1, layer->biases_gpu, 1);
    scal_gpu(layer->c, momentum, layer->bias_updates_gpu, 1);

    axpy_gpu(layer->c, learning_rate / batch, layer->scale_updates_gpu, 1, layer->scales_gpu, 1);
    scal_gpu(layer->c, momentum, layer->scale_updates_gpu, 1);
}

#elif defined(OPENCL)
void forward_batchnorm_layer_cl(const batchnorm_layer *layer, cl_mem input_cl, int test)
{
    copy_cl(layer->outputs*layer->batch, input_cl, 1, layer->output_cl, 1);
    if(0 == test){    // 0: train, 1: valid
        copy_cl(layer->batch * layer->out_h * layer->out_w * layer->c, layer->output_cl, 1, layer->x_cl, 1);
        fast_mean_cl(layer->output_cl, layer->batch, layer->c, layer->out_h*layer->out_w, layer->mean_cl);
        fast_variance_cl(layer->output_cl, layer->mean_cl, layer->batch, layer->c, layer->out_h*layer->out_w,
                          layer->variance_cl);

        scal_cl(layer->c, .99, layer->rolling_mean_cl, 1);
        axpy_cl(layer->c, .01, layer->mean_cl, 1, layer->rolling_mean_cl, 1);
        scal_cl(layer->c, .99, layer->rolling_variance_cl, 1);
        axpy_cl(layer->c, .01, layer->variance_cl, 1, layer->rolling_variance_cl, 1);

        normalize_cl(layer->output_cl, layer->mean_cl, layer->variance_cl, layer->batch, layer->c,
                      layer->out_h*layer->out_w);
        copy_cl(layer->batch * layer->out_h * layer->out_w * layer->c, layer->output_cl, 1, layer->x_norm_cl, 1);
        scale_bias_cl(layer->output_cl, layer->scales_cl, layer->batch, layer->c, layer->out_h*layer->out_w);
        add_bias_cl(layer->batch, layer->out_h * layer->out_w, layer->c, layer->biases_cl, layer->output_cl);
    } else {
        normalize_cl(layer->output_cl, layer->rolling_mean_cl, layer->rolling_variance_cl,
                     layer->batch, layer->c, layer->out_h*layer->out_w);
        scale_bias_cl(layer->output_cl, layer->scales_cl, layer->batch, layer->c, layer->out_h*layer->out_w);
        add_bias_cl(layer->batch, layer->out_h * layer->out_w, layer->c, layer->biases_cl, layer->output_cl);
    }

}

void push_batchnorm_layer_cl(const batchnorm_layer *l)
{
    cl_write_array(l->biases_cl, l->biases, l->c);
    cl_write_array(l->scales_cl, l->scales, l->c);
    cl_write_array(l->rolling_mean_cl, l->rolling_mean, l->c);
    cl_write_array(l->rolling_variance_cl, l->rolling_variance, l->c);
}
#endif
