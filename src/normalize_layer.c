#include "normalize_layer.h"

normalize_layer *make_normalize_layer(int w, int h, int c, int batch, int test)
{
    int inputs = w * h * c;
    fprintf(stderr, "Normalize:          %d x %d x %d, %d inputs\n", w, h, c, inputs);
    normalize_layer *l = calloc(1, sizeof(normalize_layer));
    l->w = w;
    l->h = h;
    l->c = c;
    l->inputs = inputs;
    l->outputs = inputs;
    l->batch = batch;
    l->test = test;
#ifndef FORWARD_GPU
    l->output = calloc(inputs*batch, sizeof(float));
    l->norm_data = calloc(batch * l->w * l->h, sizeof(float));
#endif
    if(0 == l->test){    // 0: train, 1: valid
        l->delta = calloc(inputs*batch, sizeof(float));
    }
#ifdef GPU
    l->output_gpu = cuda_make_array(l->output, inputs*batch);
    l->norm_data_gpu = cuda_make_array(l->norm_data, l->w * l->h * batch);
    if(0 == l->test){    // 0: train, 1: valid
        l->delta_gpu = cuda_make_array(l->delta, inputs*batch);
    }
#elif defined(OPENCL)
    l->output_cl = cl_make_array(l->output, inputs*batch);
    l->norm_data_cl = cl_make_array(l->norm_data, l->w * l->h * batch);
    if(0 == l->test){    // 0: train, 1: valid
        l->delta_cl = cl_make_array(l->delta, inputs*batch);
    }
#endif
    return l;
} 

void forward_normalize_layer(const normalize_layer *l, float *input)
{
    memcpy(l->output, input, l->inputs * l->batch * sizeof(float));
    l2normalize_cpu(l->output, l->batch, l->c, l->w*l->h, l->norm_data);
    /*for(int j = 0; j < 100; ++j){
        printf("forward_normalize_layer %d %f %f\n", j, input[j], l->output[j]);
    }
    printf("\n");*/
}

void backward_normalize_layer(const normalize_layer *l, float *delta)
{
    //axpy_cpu(l->inputs * l->batch, 1, l->delta, 1, delta, 1);
    backward_l2normalize_cpu(l->batch, l->c, l->w*l->h, l->norm_data, l->output, l->delta, delta);
}

#ifdef GPU

void forward_normalize_layer_gpu(const normalize_layer *l, float *input)
{
    cuda_mem_copy(l->output_gpu, input, l->inputs*l->batch);
    l2normalize_gpu(l->output_gpu, l->batch, l->c, l->w*l->h, l->norm_data_gpu);
}

void backward_normalize_layer_gpu(const normalize_layer *l, float *delta)
{
    //axpy_gpu(l->inputs * l->batch, 1, l->delta_gpu, 1, delta, 1);
    backward_l2normalize_gpu(l->batch, l->c, l->w*l->h, l->norm_data_gpu, l->output_gpu, l->delta_gpu, delta);
}

#elif defined(OPENCL)
void forward_normalize_layer_cl(const normalize_layer *l, cl_mem input)
{
    copy_cl(l->batch * l->inputs, input, 1, l->output_cl, 1);
    l2normalize_cl(l->output_cl, l->batch, l->c, l->w*l->h, l->norm_data_cl);
}
#endif
