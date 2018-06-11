#include "connected_layer.h"

connected_layer *make_connected_layer(int inputs, int outputs, int batch, ACTIVATION activation)
{
    fprintf(stderr, "Connected Layer:    %d inputs, %d outputs\n", inputs, outputs);
    connected_layer *layer = calloc(1, sizeof(connected_layer));
    layer->inputs = inputs;
    layer->outputs = outputs;
    layer->batch = batch;
    layer->output = calloc(batch*outputs, sizeof(float));
    layer->delta = calloc(batch*outputs, sizeof(float));

    layer->weights = calloc(inputs*outputs, sizeof(float));  // layer->inputs is the number of rows
    float scale = sqrt(2.0F/inputs);
    for(int i = 0; i < inputs*outputs; ++i) layer->weights[i] = rand_normal() * scale;
    //scale = 1.0F/(inputs);
    //for(int i = 0; i < inputs*outputs; ++i) layer->weights[i] = scale*rand_uniform(0, 1);
    layer->weight_updates = calloc(inputs*outputs, sizeof(float));
    layer->biases = calloc(outputs, sizeof(float));
    layer->bias_updates = calloc(outputs, sizeof(float));
    for(int i = 0; i < outputs; ++i) layer->biases[i] = 0;
    layer->activation = activation;
    return layer;
}

void forward_connected_layer(connected_layer *layer, float *input)
{
    for(int i = 0; i < layer->batch; ++i){
        memcpy(layer->output + layer->outputs * i, layer->biases, layer->outputs*sizeof(float));
    }
    float *a = input;
    float *b = layer->weights;  // layer->inputs is the number of rows
    float *c = layer->output;
    int m = layer->batch;
    int n = layer->outputs;
    int k = layer->inputs;
    gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    int all_outputs = layer->outputs * layer->batch;
    for(int i = 0; i < all_outputs; ++i){
        layer->output[i] = activate(layer->output[i], layer->activation);
    }
}

void update_connected_layer(connected_layer *layer, float learning_rate, float momentum, float decay)
{
    for(int i = 0; i < layer->outputs; i ++){
    	layer->biases[i] += learning_rate / layer->batch * layer->bias_updates[i];
    	layer->bias_updates[i] *= momentum;
    }

    int size = layer->inputs*layer->outputs;
    for(int i = 0; i < size; i ++){
    	layer->weight_updates[i] += -decay*layer->batch*layer->weights[i];
    	layer->weights[i] += learning_rate / layer->batch * layer->weight_updates[i];
    	layer->weight_updates[i] *= momentum;
    }
}

void backward_connected_layer(connected_layer *layer, float *input, float *delta)
{
	int all_outputs = layer->outputs * layer->batch;
    for(int i = 0; i < all_outputs; ++i){
        layer->delta[i] *= gradient(layer->output[i], layer->activation);
    }
    for(int i = 0; i < layer->batch; ++i){
		for(int j = 0; j < layer->outputs; ++j){
			layer->bias_updates[j] += (layer->delta + i * layer->outputs)[j];
		}
    }
    int m = layer->inputs;
    int n = layer->outputs;
    int k = layer->batch;
    float *a = input;
    float *b = layer->delta;
    float *c = layer->weight_updates;  // layer->inputs is the number of rows
    gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);

    if(delta) {
        memset(delta, 0, layer->batch * layer->inputs*sizeof(float));
        m = layer->batch;
        n = layer->inputs;
        k = layer->outputs;
        a = layer->delta;  // layer->inputs is the number of rows
        b = layer->weights;
        c = delta;
        gemm(0,1,m,n,k,1,a,k,b,k,0,c,n);
    }
}
