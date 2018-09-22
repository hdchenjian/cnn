#include "dropout_layer.h"

image get_dropout_image(const dropout_layer *layer)
{
    int h = layer->h;
    int w = layer->w;
    int c = layer->c;
    return float_to_image(h,w,c,NULL);
}

dropout_layer *make_dropout_layer(int w, int h, int c, int batch, int inputs, float probability)
{
    fprintf(stderr, "Dropout:            p = %.2f, %4d  ->  %4d\n", probability, inputs, inputs);
    dropout_layer *l = calloc(1, sizeof(dropout_layer));
    l->w = w;
    l->h = h;
    l->c = c;
    l->probability = probability;
    l->inputs = inputs;
    l->outputs = inputs;
    l->batch = batch;
    l->rand = calloc(inputs*batch, sizeof(float));
    l->scale = 1.0F / (1.0F - probability);

#ifdef GPU
    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
#endif
    return l;
} 

void forward_dropout_layer(const dropout_layer *l, float *input, int test)
{
    if (0 != test) return;  // 0: train, 1: valid
    for(int i = 0; i < l->batch * l->inputs; ++i){
        float r = rand_uniform(0, 1);
        l->rand[i] = r;
        if(r < l->probability) input[i] = 0;
        else input[i] *= l->scale;
    }
}

void backward_dropout_layer(const dropout_layer *l, float *delta)
{
    if(!delta) return;
    for(int i = 0; i < l->batch * l->inputs; ++i){
        float r = l->rand[i];
        if(r < l->probability) delta[i] = 0;
        else delta[i] *= l->scale;
    }
}
