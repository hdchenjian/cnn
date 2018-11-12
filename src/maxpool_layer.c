#include "maxpool_layer.h"
#include <stdio.h>

image get_maxpool_image(const maxpool_layer *layer)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->c;
    return float_to_image(h,w,c,NULL);
}

maxpool_layer *make_maxpool_layer(int h, int w, int c, int size, int stride, int batch, int padding, int test)
{
    fprintf(stderr, "Maxpool:            %d x %d x %d inputs, size: %d, %d stride\n", w,h,c,size,stride);
    maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
    layer->test = test;
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->size = size;
    layer->stride = stride;
    layer->batch = batch;
    layer->pad = padding;
    layer->out_w = (w + padding - size)/stride + 1;
    layer->out_h = (h + padding - size)/stride + 1;
    layer->outputs = layer->out_h * layer->out_w * c;
#ifndef FORWARD_GPU
    layer->output = calloc(batch * layer->outputs, sizeof(float));
#endif
    if(0 == layer->test){    // 0: train, 1: valid
        layer->delta = calloc(batch * layer->outputs, sizeof(float));
        layer->indexes = calloc(layer->outputs * batch, sizeof(int));
    }
#ifdef GPU
    if(0 == layer->test){    // 0: train, 1: valid
        layer->indexes_gpu = cuda_make_int_array(0, layer->outputs * batch);
        layer->delta_gpu = cuda_make_array(layer->delta, layer->outputs * batch);
    }
    layer->output_gpu = cuda_make_array(layer->output, layer->outputs * batch);
#elif defined(OPENCL)
    if(0 == layer->test){    // 0: train, 1: valid
        layer->indexes_cl = cl_make_int_array(0, layer->outputs * batch);
        layer->delta_cl =  cl_make_array(layer->delta, layer->outputs*batch);
    }
    layer->output_cl = cl_make_array(layer->output, layer->outputs*batch);
#endif
    return layer;
}

void forward_maxpool_layer(const maxpool_layer *layer, float *in)
{
    int b,i,j,k,m,n;
    int w_offset = -layer->pad / 2;
    int h_offset = -layer->pad / 2;

    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->c;

    for(b = 0; b < layer->batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < layer->size; ++n){
                        for(m = 0; m < layer->size; ++m){
                            int cur_h = h_offset + i*layer->stride + n;
                            int cur_w = w_offset + j*layer->stride + m;
                            int index = cur_w + layer->w*(cur_h + layer->h*(k + b*layer->c));
                            int valid = (cur_h >= 0 && cur_h < layer->h &&
                                         cur_w >= 0 && cur_w < layer->w);
                            float val = (valid != 0) ? in[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    layer->output[out_index] = max;
                    if(0 == layer->test){    // 0: train, 1: valid
                        layer->indexes[out_index] = max_i;
                    }
                }
            }
        }
    }

    /*
    float max = -FLT_MAX, min = FLT_MAX;
    for(int i = 0; i < layer->batch * h * w * c; ++i){
    	if(layer->output[i] > max) max = layer->output[i];
    	if(layer->output[i] < min) min = layer->output[i];
    }
    printf("forward_maxpool_layer max: %f, min: %f\n", max, min);*/
}

void backward_maxpool_layer(const maxpool_layer *layer, float *delta)
{
    int i;
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->c;
    //memset(delta, 0, layer->h*layer->w*layer->c*layer->batch * sizeof(float));
    for(i = 0; i < h*w*c*layer->batch; ++i){
        int index = layer->indexes[i];
        delta[index] += layer->delta[i];
    }
}

#if defined(OPENCL)
void forward_maxpool_layer_cl(const maxpool_layer *layer, cl_mem in_cl){
    cl_kernel kernel = get_kernel_by_name("forward_maxpool_layer_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->h), (void*)&layer->h);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->w), (void*)&layer->w);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->c), (void*)&layer->c);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->stride), (void*)&layer->stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->size), (void*)&layer->size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->pad), (void*)&layer->pad);
    cl.error = clSetKernelArg(kernel, i++, sizeof(in_cl), (void*)&in_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->output_cl), (void*)&layer->output_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->indexes_cl), (void*)&layer->indexes_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer->test), (void*)&layer->test);
    check_error(cl);
    const size_t global_size[] = {layer->out_h *layer->out_w * layer->c * layer->batch};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}
#endif
