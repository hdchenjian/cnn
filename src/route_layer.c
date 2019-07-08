#include "network.h"

image get_route_image(const route_layer *layer)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->out_c;
    return float_to_image(h,w,c,NULL);
}

route_layer *make_route_layer(int batch, int n, int *input_layers, int *input_sizes, network *net, int test)
{
    route_layer *l = calloc(1, sizeof(route_layer));
    l->batch = batch;
    l->n = n;
    l->input_layers = input_layers;
    l->input_sizes = input_sizes;
    l->test = test;
    int i;
    int outputs = 0;
    char input_layer_str[128] = {0};
    for(i = 0; i < n; ++i){
        outputs += input_sizes[i];
        sprintf(input_layer_str + i * 4, "%3d ", input_layers[i]);
    }
    image first_layer = get_network_image_layer(net, input_layers[0]);
    l->out_w = first_layer.w;
    l->out_h = first_layer.h;
    l->out_c = first_layer.c;
    for(int i = 1; i < n; ++i){
        int index = input_layers[i];
        image before_layer = get_network_image_layer(net, index);
        if(before_layer.w == first_layer.w && before_layer.h == first_layer.h){
            l->out_c += before_layer.c;
        }else{
            fprintf(stderr, "make_route_layer, input layer size not same\n");
            exit(-1);
        }
    }

    l->outputs = outputs;
    l->inputs = outputs;
    if(0 == l->test){    // 0: train, 1: valid
        l->delta =  calloc(outputs*batch, sizeof(float));
    }
#ifndef FORWARD_GPU
    l->output = calloc(outputs*batch, sizeof(float));
#if defined QML || defined INTEL_MKL || defined OPENBLAS_ARM || defined ARM_BLAS
    if(l->n == 1){
        free(l->output);
        l->output = get_network_layer_data(net, l->input_layers[0], 0, 0);
    }
#endif
#endif

#ifdef GPU
    if(0 == l->test){    // 0: train, 1: valid
        l->delta_gpu =  cuda_make_array(l->delta, outputs*batch);
    }
    l->output_gpu = cuda_make_array(l->output, outputs*batch);
#elif defined(OPENCL)
    if(0 == l->test){    // 0: train, 1: valid
        l->delta_cl =  cl_make_array(l->delta, l->outputs*batch);
    }
    //l->output_cl = cl_make_share_array(l->output, l->outputs*batch);
    l->output_cl = cl_make_array(l->output, l->outputs*batch);
    if(l->n == 1) {
        clReleaseMemObject(l->output_cl);
        l->output_cl = get_network_layer_data_cl(net, l->input_layers[0], 0);
    }
#endif
    fprintf(stderr, "Route:              %d x %d x %d -> %d x %d x %d, %d inputs, layer: %s\n",
            l->out_w, l->out_h, l->out_c, l->out_w, l->out_h, l->out_c, l->inputs, input_layer_str);
    return l;
}

void forward_route_layer(const route_layer *l, network *net)
{
#if defined QML || defined INTEL_MKL || defined OPENBLAS_ARM || defined ARM_BLAS
    if(l->n == 1) return;
#endif
    int offset = 0;
    for(int i = 0; i < l->n; ++i){
        int index = l->input_layers[i];
        float *input = get_network_layer_data(net, index, 0, 0);
        int input_size = l->input_sizes[i];
        for(int j = 0; j < l->batch; ++j){
            memcpy(l->output + offset + j*l->outputs, input + j*input_size, input_size * sizeof(float));
        }
        offset += input_size;
    }
}

void backward_route_layer(const route_layer *l, network *net)
{
    int offset = 0;
    for(int i = 0; i < l->n; ++i){
        int index = l->input_layers[i];
        float *delta = get_network_layer_data(net, index, 1, 0);
        int input_size = l->input_sizes[i];
        for(int j = 0; j < l->batch; ++j){
            //memcpy(delta + j*input_size, l->delta + offset + j*l->outputs, input_size * sizeof(float));
            axpy_cpu(input_size, 1, l->delta + offset + j*l->outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer *l, network *net)
{
    int offset = 0;
    for(int i = 0; i < l->n; ++i){
        int index = l->input_layers[i];
        float *input = get_network_layer_data(net, index, 0, 1);
        int input_size = l->input_sizes[i];
        for(int j = 0; j < l->batch; ++j){
            cuda_mem_copy(l->output_gpu + offset + j*l->outputs, input + j*input_size, input_size);
        }
        offset += input_size;
    }
}

void backward_route_layer_gpu(const route_layer *l, network *net)
{
    int offset = 0;
    for(int i = 0; i < l->n; ++i){
        int index = l->input_layers[i];
        float *delta = get_network_layer_data(net, index, 1, 1);
        int input_size = l->input_sizes[i];
        for(int j = 0; j < l->batch; ++j){
            //cuda_mem_copy(delta + j*input_size, l->delta_gpu + offset + j*l->outputs, input_size);
            axpy_gpu(input_size, 1, l->delta_gpu + offset + j*l->outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#elif defined(OPENCL)
void forward_route_layer_cl(const route_layer *l, network *net){
    //cl_compare_array(layer->output_cl, layer->output, layer->outputs*layer->batch, "route output diff: ", i);
    if(l->n == 1) return;
    int offset = 0;
    for(int i = 0; i < l->n; ++i){
        int index = l->input_layers[i];
        cl_mem input = get_network_layer_data_cl(net, index, 0);
        int input_size = l->input_sizes[i];
        for(int j = 0; j < l->batch; ++j){
            //cl_copy_array_with_offset(l->output_gpu + offset + j*l->outputs, input + j*input_size, input_size);
            cl_copy_array_with_offset(input, l->output_cl, input_size, j*input_size, offset + j*l->outputs);
        }
        offset += input_size;
    }
}
#endif
