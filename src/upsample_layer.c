#include "upsample_layer.h"
#include "utils.h"

image get_upsample_image(const upsample_layer *layer)
{
    return float_to_image(layer->out_h, layer->out_w, layer->out_c, NULL);
}

upsample_layer *make_upsample_layer(int batch, int w, int h, int c, int stride, int test)
{
    upsample_layer *l = calloc(1, sizeof(upsample_layer));
    l->batch = batch;
    l->w = w;
    l->h = h;
    l->c = c;
    l->scale = 1;
    l->out_w = w*stride;
    l->out_h = h*stride;
    l->out_c = c;
    l->stride = stride;
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->w*l->h*l->c;
    l->test = test;
    if(0 == l->test){    // 0: train, 1: valid
        l->delta =  calloc(l->outputs*batch, sizeof(float));
    }
#ifndef FORWARD_GPU
    l->output = calloc(l->outputs*batch, sizeof(float));;
#endif


#ifdef GPU
    if(0 == l->test){    // 0: train, 1: valid
        l->delta_gpu =  cuda_make_array(l->delta, l->outputs*batch);
    }
    l->output_gpu = cuda_make_array(l->output, l->outputs*batch);
#elif defined(OPENCL)
    if(0 == l->test){    // 0: train, 1: valid
        l->delta_cl =  cl_make_array(l->delta, l->outputs*batch);
    }
    l->output_cl = cl_make_array(l->output, l->outputs*batch);
#endif
    fprintf(stderr, "upsample          %4d x%4d x%4d   ->  %4d x%4d x%4d, stride: %d\n", w, h, c, l->out_w, l->out_h, l->out_c, stride);
    return l;
}

void free_upsample_layer(void *input)
{
    upsample_layer *layer = (upsample_layer *)input;
    if(layer->output) free_ptr((void *)&(layer->output));
    if(layer->delta) free_ptr((void *)&(layer->delta));
#ifdef GPU
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#elif defined(OPENCL)
    if(layer->output_cl) clReleaseMemObject(layer->output_cl);
    if(layer->delta_cl) clReleaseMemObject(layer->delta_cl);
#endif
    free_ptr((void *)&layer);
}

void forward_upsample_layer(const upsample_layer *l, float *input)
{
    upsample_cpu(input, l->w, l->h, l->c, l->batch, l->stride, 1, l->scale, l->output);
}

void backward_upsample_layer(const upsample_layer *l, float * delta)
{
    //memset(delta, 0, l->h*l->w*l->c*l->batch * sizeof(float));
    upsample_cpu(delta, l->w, l->h, l->c, l->batch, l->stride, 0, l->scale, l->delta);
}

#ifdef GPU
void forward_upsample_layer_gpu(const upsample_layer *l, float *input)
{
    upsample_gpu(input, l->w, l->h, l->c, l->batch, l->stride, 1, l->scale, l->output_gpu);
}

void backward_upsample_layer_gpu(const upsample_layer *l, float *delta)
{
    //fill_gpu(l->h*l->w*l->c*l->batch, 0, delta, 1);
    upsample_gpu(delta, l->w, l->h, l->c, l->batch, l->stride, 0, l->scale, l->delta_gpu);
}

#elif defined(OPENCL)
void forward_upsample_layer_cl(const upsample_layer *l, cl_mem input){
    cl_kernel kernel = get_kernel_by_name("upsample_cl", 0);
    cl_uint i = 0;
    int forward = 1;
    cl.error = clSetKernelArg(kernel, i++, sizeof(input), (void*)&input);
    cl.error = clSetKernelArg(kernel, i++, sizeof(l->w), (void*)&l->w);
    cl.error = clSetKernelArg(kernel, i++, sizeof(l->h), (void*)&l->h);
    cl.error = clSetKernelArg(kernel, i++, sizeof(l->c), (void*)&l->c);
    cl.error = clSetKernelArg(kernel, i++, sizeof(l->batch), (void*)&l->batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(l->stride), (void*)&l->stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(forward), (void*)&forward);
    cl.error = clSetKernelArg(kernel, i++, sizeof(l->scale), (void*)&l->scale);
    cl.error = clSetKernelArg(kernel, i++, sizeof(l->output_cl), (void*)&l->output_cl);
    check_error(cl);
    const size_t global_size[] = {l->w*l->h*l->c*l->batch*l->stride*l->stride};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}
#endif
