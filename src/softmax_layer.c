#include "softmax_layer.h"

softmax_layer *make_softmax_layer(int inputs)
{
    fprintf(stderr, "Softmax:            %d inputs\n", inputs);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->inputs = inputs;
    layer->output = calloc(inputs, sizeof(float));
    layer->delta = calloc(inputs, sizeof(float));
    return layer;
}

void forward_softmax_layer(const softmax_layer *layer, float *input)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < layer->inputs; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < layer->inputs; ++i){
        sum += exp(input[i]-largest);
    }

    for(i = 0; i < layer->inputs; ++i){
        layer->output[i] = exp(input[i]-largest) / sum;
        printf("forward_softmax_layer %f\n", layer->output[i]);
    }
}

void backward_softmax_layer(const softmax_layer *layer, float *delta)
{
    int i;
    for(i = 0; i < layer->inputs; ++i){
        delta[i] = layer->delta[i];
    }
}

