#include "dropout_layer.h"

image get_dropout_image(const dropout_layer *layer, int batch)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->c;
    return float_to_image(h,w,c,layer->output + batch * h * w * c);
}

dropout_layer *make_dropout_layer(int w, int h, int c, int batch, int inputs, float probability,
        float *previous_layer_output, float *previous_layer_delta)
{
    fprintf(stderr, "Dropout:            p = %.2f, %4d  ->  %4d\n", probability, inputs, inputs);
    dropout_layer *l = calloc(1, sizeof(dropout_layer));
    l->out_w = w;
    l->out_h = h;
    l->c = c;
    l->probability = probability;
    l->inputs = inputs;
    l->outputs = inputs;
    l->batch = batch;
    l->rand = calloc(inputs*batch, sizeof(float));
    l->scale = 1./(1.-probability);
    l->output = previous_layer_output;  // reuse previous layer output and delta
    l->delta = previous_layer_delta;
    return l;
} 

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}

void forward_dropout_layer(dropout_layer *l, float *input, network *net)
{
    if (0 != net->test) return;  // 0: train, 1: valid, 2: test
    for(int i = 0; i < l->batch * l->inputs; ++i){
        float r = rand_uniform(0, 1);
        l->rand[i] = r;
        if(r < l->probability) input[i] = 0;
        else input[i] *= l->scale;
    }
}

void backward_dropout_layer(dropout_layer *l, float *delta)
{
    if(!delta) return;
    for(int i = 0; i < l->batch * l->inputs; ++i){
        float r = l->rand[i];
        if(r < l->probability) delta[i] = 0;
        else delta[i] *= l->scale;
    }
}
