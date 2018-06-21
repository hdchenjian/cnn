#include "connected_layer.h"
#include <float.h>

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

#ifdef GPU
    layer->biases_gpu = cuda_make_array(layer->biases, outputs);
    layer->bias_updates_gpu = cuda_make_array(layer->bias_updates, outputs);
    layer->weights_gpu = cuda_make_array(layer->weights, outputs*inputs);
    layer->weight_updates_gpu = cuda_make_array(layer->weight_updates, outputs*inputs);
    layer->output_gpu = cuda_make_array(layer->output, outputs*batch);
    layer->delta_gpu = cuda_make_array(layer->delta, outputs*batch);
#endif

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

    float max = -FLT_MAX, min = FLT_MAX;
    for(int i = 0; i < layer->batch * layer->outputs; ++i){
    	if(layer->output[i] > max) max = layer->output[i];
    	if(layer->output[i] < min) min = layer->output[i];
    }
    printf("forward_connected_layer max: %f, min: %f\n", max, min);
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
    /*
    for(int i = 0; i < layer->outputs*layer->batch; i++){
        printf("backward_connected_layer layer->delta: %f \n", layer->delta[i]);
    }
    printf("\n");*/

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
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    if(delta) {
        //memset(delta, 0, layer->batch * layer->inputs*sizeof(float));
        m = layer->batch;
        n = layer->inputs;
        k = layer->outputs;
        a = layer->delta;
        b = layer->weights;
        c = delta;
        gemm(0,1,m,n,k,1,a,k,b,k,0,c,n);

        /*
        for(int i = 0; i < layer->batch; ++i){
            float max = -FLT_MAX;
            float min = FLT_MAX;
            for(int j = 0; j < layer->outputs; ++j){
                //printf("backward_connected_layer  %f, \n", layer->delta[layer->outputs * i +j]);
                if(layer->delta[layer->outputs * i +j] > max) max = layer->delta[layer->outputs * i +j];
                if(layer->delta[layer->outputs * i +j] < min) min = layer->delta[layer->outputs * i +j];
            }
            printf("backward_connected_layer max: %f, min: %f\n", max, min);
        }*/
    }
}

#ifdef GPU

void pull_connected_layer(const connected_layer *layer)
{
    cuda_pull_array(layer->weights_gpu, layer->weights, layer->inputs*layer->outputs);
    cuda_pull_array(layer->biases_gpu, layer->biases, layer->outputs);
    cuda_pull_array(layer->weight_updates_gpu, layer->weight_updates, layer->inputs*layer->outputs);
    cuda_pull_array(layer->bias_updates_gpu, layer->bias_updates, layer->outputs);

}

void push_connected_layer(const connected_layer *layer)
{
    cuda_push_array(layer->weights_gpu, layer->weights, layer->inputs*layer->outputs);
    cuda_push_array(layer->biases_gpu, layer->biases, layer->outputs);
    cuda_push_array(layer->weight_updates_gpu, layer->weight_updates, layer->inputs*layer->outputs);
    cuda_push_array(layer->bias_updates_gpu, layer->bias_updates, layer->outputs);

}

void update_connected_layer_gpu(connected_layer *layer, float learning_rate, float momentum, float decay)
{
    axpy_gpu(layer->outputs, learning_rate/layer->batch, layer->bias_updates_gpu, 1, layer->biases_gpu, 1);
    scal_gpu(layer->outputs, momentum, layer->bias_updates_gpu, 1);

    axpy_gpu(layer->inputs*layer->outputs, -decay*layer->batch, layer->weights_gpu, 1, layer->weight_updates_gpu, 1);
    axpy_gpu(layer->inputs*layer->outputs, learning_rate/layer->batch, layer->weight_updates_gpu, 1, layer->weights_gpu, 1);
    scal_gpu(layer->inputs*layer->outputs, momentum, layer->weight_updates_gpu, 1);

}

void forward_connected_layer_gpu(connected_layer *layer, float *input)
{
    int m = layer->batch;
    int n = layer->outputs;
    int k = layer->inputs;
    float * a = input;
    float * b = layer->weights_gpu;
    float * c = layer->output_gpu;
    gemm_gpu(0, 0, m, n, k, 1, a, k, b, n, 0, c, n);
    add_bias_gpu(layer->output_gpu, layer->biases_gpu, layer->batch, layer->outputs, 1);
    activate_array_gpu(layer->output_gpu, layer->outputs*layer->batch, layer->activation);

    float *output_temp = (float *)calloc(layer->batch * layer->outputs, sizeof(float));
    cuda_pull_array(layer->output_gpu, output_temp, layer->outputs*layer->batch);
    float max = -FLT_MAX, min = FLT_MAX;
    for(int i = 0; i < layer->batch * layer->outputs; ++i){
    	if(output_temp[i] > max) max = output_temp[i];
    	if(output_temp[i] < min) min = output_temp[i];
    }
    printf("forward_connected_layer_gpu max: %f, min: %f\n", max, min);
}

void backward_connected_layer_gpu(connected_layer *layer, float *input, float *delta)
{
    /*
    float *delta_temp = calloc(layer->outputs*layer->batch, sizeof(float));
    cuda_pull_array(layer->delta_gpu, delta_temp, layer->outputs*layer->batch);
    for(int i = 0; i < layer->outputs*layer->batch; i++){
        printf("backward_connected_layer_gpu layer->delta_gpu: %f \n", delta_temp[i]);
    }
    printf("\n"); */

    gradient_array_gpu(layer->output_gpu, layer->outputs*layer->batch, layer->activation, layer->delta_gpu);
    backward_bias_gpu(layer->bias_updates_gpu, layer->delta_gpu, layer->batch, layer->outputs, 1);

    int m = layer->inputs;
    int n = layer->outputs;
    int k = layer->batch;
    float * a = input;
    float * b = layer->delta_gpu;
    float * c = layer->weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = layer->batch;
    n = layer->inputs;
    k = layer->outputs;

    a = layer->delta_gpu;
    b = layer->weights_gpu;
    c = delta;

    if(c) {
        gemm_gpu(0,1,m,n,k,1,a,k,b,k,0,c,n);
    }
}
#endif
