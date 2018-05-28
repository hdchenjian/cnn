#include "connected_layer.h"

connected_layer *make_connected_layer(int inputs, int outputs, ACTIVATION activation)
{
    fprintf(stderr, "Connected Layer:    %d inputs, %d outputs\n", inputs, outputs);
    int i;
    connected_layer *layer = calloc(1, sizeof(connected_layer));
    layer->inputs = inputs;
    layer->outputs = outputs;

    layer->output = calloc(outputs, sizeof(float*));
    layer->delta = calloc(outputs, sizeof(float*));

    layer->weight_updates = calloc(inputs*outputs, sizeof(float));
    layer->weight_momentum = calloc(inputs*outputs, sizeof(float));
    layer->weights = calloc(inputs*outputs, sizeof(float));
    float scale = 1.0F/inputs;
    for(i = 0; i < inputs*outputs; ++i)
        layer->weights[i] = rand_uniform(0, 1) * scale;

    layer->bias_updates = calloc(outputs, sizeof(float));
    layer->bias_momentum = calloc(outputs, sizeof(float));
    layer->biases = calloc(outputs, sizeof(float));
    for(i = 0; i < outputs; ++i)
        //layer->biases[i] = rand_normal()*scale + scale;
        layer->biases[i] = 0;

    layer->activation = activation;
    return layer;
}

void forward_connected_layer(connected_layer *layer, float *input)
{
    int i, j;
    for(i = 0; i < layer->outputs; ++i){
        layer->output[i] = layer->biases[i];
        for(j = 0; j < layer->inputs; ++j){
            layer->output[i] += input[j]*layer->weights[i*layer->inputs + j];
        }
        layer->output[i] = activate(layer->output[i], layer->activation);
    }
}

void learn_connected_layer(connected_layer *layer, float *input)
{
    int i, j;
    for(i = 0; i < layer->outputs; ++i){
        layer->delta[i] *= gradient(layer->output[i], layer->activation);
        layer->bias_updates[i] += layer->delta[i];
        for(j = 0; j < layer->inputs; ++j){
            layer->weight_updates[i*layer->inputs + j] += layer->delta[i]*input[j];
        }
    }
}

void update_connected_layer(connected_layer *layer, float step, float momentum, float decay)
{
    int i,j;
    for(i = 0; i < layer->outputs; ++i){
        layer->bias_momentum[i] = step*(layer->bias_updates[i]) + momentum*layer->bias_momentum[i];
        layer->biases[i] += layer->bias_momentum[i];
        for(j = 0; j < layer->inputs; ++j){
            int index = i*layer->inputs+j;
            layer->weight_momentum[index] = step*(layer->weight_updates[index] - decay*layer->weights[index]) +
            		momentum*layer->weight_momentum[index];
            layer->weights[index] += layer->weight_momentum[index];
            //layer->weights[index] = constrain(layer->weights[index], 100.);
        }
    }
    memset(layer->bias_updates, 0, layer->outputs*sizeof(float));
    memset(layer->weight_updates, 0, layer->outputs*layer->inputs*sizeof(float));
}

void backward_connected_layer(connected_layer *layer, float *delta)
{
    int i, j;

    for(j = 0; j < layer->inputs; ++j){
        delta[j] = 0;
        for(i = 0; i < layer->outputs; ++i){
            delta[j] += layer->delta[i]*layer->weights[i*layer->inputs + j];
        }
    }
}

