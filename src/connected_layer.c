#include "connected_layer.h"

connected_layer *make_connected_layer(int inputs, int outputs, ACTIVATION activation)
{
    fprintf(stderr, "Connected Layer:    %d inputs, %d outputs\n", inputs, outputs);
    connected_layer *layer = calloc(1, sizeof(connected_layer));
    layer->inputs = inputs;
    layer->outputs = outputs;
    layer->output = calloc(outputs, sizeof(float));
    layer->delta = calloc(outputs, sizeof(float));

    layer->weights = calloc(inputs*outputs, sizeof(float));  // layer->inputs is the number of rows
    float scale = 1.0F/inputs;
    for(int i = 0; i < inputs*outputs; ++i) layer->weights[i] = rand_uniform(0, 1) * scale;
    layer->weight_updates = calloc(inputs*outputs, sizeof(float));
    layer->weight_momentum = calloc(inputs*outputs, sizeof(float));

    layer->biases = calloc(outputs, sizeof(float));
    layer->bias_updates = calloc(outputs, sizeof(float));
    layer->bias_momentum = calloc(outputs, sizeof(float));
    for(int i = 0; i < outputs; ++i) layer->biases[i] = 0;
    layer->activation = activation;
    return layer;
}

void forward_connected_layer(connected_layer *layer, float *input)
{
    memcpy(layer->output, layer->biases, layer->outputs*sizeof(float));
    float *a = input;
    float *b = layer->weights;  // layer->inputs is the number of rows
    float *c = layer->output;
    int m = 1;
    int n = layer->outputs;
    int k = layer->inputs;
    gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    for(int i = 0; i < layer->outputs; ++i){
        layer->output[i] = activate(layer->output[i], layer->activation);
    }
}

void update_connected_layer(connected_layer *layer, float step, float momentum, float decay)
{
    for(int i = 0; i < layer->outputs; ++i){
        layer->bias_momentum[i] = step*(layer->bias_updates[i]) + momentum*layer->bias_momentum[i];
        layer->biases[i] += layer->bias_momentum[i];
        for(int j = 0; j < layer->inputs; ++j){
            int index = i * layer->inputs + j;  // layer->inputs is the number of rows
            layer->weight_momentum[index] = step*(layer->weight_updates[index] - decay*layer->weights[index]) +
                momentum*layer->weight_momentum[index];
            layer->weights[index] += layer->weight_momentum[index];
        }
    }
    memset(layer->bias_updates, 0, layer->outputs*sizeof(float));
    memset(layer->weight_updates, 0, layer->outputs*layer->inputs*sizeof(float));
}


void backward_connected_layer(connected_layer *layer, float *input, float *delta)
{
    for(int i = 0; i < layer->outputs; ++i){
        layer->delta[i] *= gradient(layer->output[i], layer->activation);
        layer->bias_updates[i] += layer->delta[i];
    }
    int m = layer->inputs;
    int n = layer->outputs;
    int k = 1;
    float *a = input;
    float *b = layer->delta;
    float *c = layer->weight_updates;  // layer->inputs is the number of rows
    gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);

    if(delta) {
        memset(delta, 0, layer->inputs*sizeof(float));
        m = layer->inputs;
        n = 1;
        k = layer->outputs;
        a = layer->weights;  // layer->inputs is the number of rows
        b = layer->delta;
        c = delta;
        gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
    }
}
