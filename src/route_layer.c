#include "route_layer.h"

route_layer *make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    route_layer *l = calloc(1, sizeof(route_layer));
    l->batch = batch;
    l->n = n;
    l->input_layers = input_layers;
    l->input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    l->outputs = outputs;
    l->inputs = outputs;
    l->delta =  calloc(outputs*batch, sizeof(float));
    l->output = calloc(outputs*batch, sizeof(float));;

    #ifdef GPU
    l->delta_gpu =  cuda_make_array(l->delta, outputs*batch);
    l->output_gpu = cuda_make_array(l->output, outputs*batch);
    #endif
    fprintf(stderr, "Route:               layer num: %d, inputs: %d, \n", n, l->inputs);
    return l;
}

void forward_route_layer(const route_layer *l, network *net)
{
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
            axpy_cpu(input_size, 1, l->delta + offset + j*l->outputs, 1, delta + j*input_size, 1);
            memcpy(delta + j*input_size, l->delta + offset + j*l->outputs, input_size * sizeof(float));
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
            copy_gpu(input_size, input + j*input_size, 1, l->output_gpu + offset + j*l->outputs, 1);
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
            axpy_gpu(input_size, 1, l->delta_gpu + offset + j*l->outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#endif
