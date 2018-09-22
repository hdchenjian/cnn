#include "network.h"

image get_shortcut_image(const shortcut_layer *layer)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->out_c;
    return float_to_image(h,w,c,NULL);
}

shortcut_layer *make_shortcut_layer(int batch, int index, int w, int h, int c, int out_w,int out_h,int out_c,
                                    ACTIVATION activation, float prev_layer_weight, float shortcut_layer_weight)
{
    fprintf(stderr, "Shortcut:           %d x %d x %d -> %d x %d x %d, layer: %d\n", w,h,c, out_w,out_h,out_c, index);
    shortcut_layer *l = calloc(1, sizeof(shortcut_layer));
    l->batch = batch;
    l->w = w;
    l->h = h;
    l->c = c;
    l->out_w = out_w;
    l->out_h = out_h;
    l->out_c = out_c;
    l->outputs = out_w*out_h*out_c;
    l->index = index;
    l->activation = activation;
    l->prev_layer_weight = prev_layer_weight;
    l->shortcut_layer_weight = shortcut_layer_weight;

    l->delta =  calloc(l->outputs*batch, sizeof(float));
    l->output = calloc(l->outputs*batch, sizeof(float));;

    #ifdef GPU
    l->delta_gpu =  cuda_make_array(l->delta, l->outputs*batch);
    l->output_gpu = cuda_make_array(l->output, l->outputs*batch);
    #endif
    return l;
}

void forward_shortcut_layer(const shortcut_layer *l, float *input, network *net)
{
    copy_cpu(l->outputs*l->batch, input, 1, l->output, 1);
    float *shortcut_layer_output = get_network_layer_data(net, l->index, 0, 0);
    shortcut_cpu(l->batch, l->w, l->h, l->c, shortcut_layer_output, l->out_w, l->out_h, l->out_c,
                 l->prev_layer_weight, l->shortcut_layer_weight, l->output);
    activate_array(l->output, l->outputs*l->batch, l->activation);
}

void backward_shortcut_layer(const shortcut_layer *l, float *delta, network *net)
{
    gradient_array(l->output, l->outputs*l->batch, l->activation, l->delta);
    axpy_cpu(l->outputs*l->batch, l->prev_layer_weight, l->delta, 1, delta, 1);
    float *shortcut_layer_delta = get_network_layer_data(net, l->index, 1, 0);
    shortcut_cpu(l->batch, l->out_w, l->out_h, l->out_c, l->delta, l->w, l->h, l->c,
                 1, l->shortcut_layer_weight, shortcut_layer_delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const shortcut_layer *l, float *input_gpu, network *net)
{
    copy_gpu(l->outputs*l->batch, input_gpu, 1, l->output_gpu, 1);
    float *shortcut_layer_output_gpu = get_network_layer_data(net, l->index, 0, 1);
    shortcut_gpu(l->batch, l->w, l->h, l->c, shortcut_layer_output_gpu, l->out_w, l->out_h, l->out_c,
                 l->prev_layer_weight, l->shortcut_layer_weight, l->output_gpu);
    activate_array_gpu(l->output_gpu, l->outputs*l->batch, l->activation);
}

void backward_shortcut_layer_gpu(const shortcut_layer *l, float *delta_gpu, network *net)
{
    gradient_array_gpu(l->output_gpu, l->outputs*l->batch, l->activation, l->delta_gpu);
    axpy_gpu(l->outputs*l->batch, l->prev_layer_weight, l->delta_gpu, 1, delta_gpu, 1);
    float *shortcut_layer_delta_gpu = get_network_layer_data(net, l->index, 1, 1);
    shortcut_gpu(l->batch, l->out_w, l->out_h, l->out_c, l->delta_gpu, l->w, l->h, l->c,
                 1, l->shortcut_layer_weight, shortcut_layer_delta_gpu);
}
#endif
