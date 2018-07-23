#include "connected_layer.h"
#include <float.h>

image get_connected_image(const connected_layer *layer)
{
    int h = 1;
    int w = 1;
    int c = layer->outputs;
    return float_to_image(h,w,c,NULL);
}

connected_layer *make_connected_layer(int inputs, int outputs, int batch, ACTIVATION activation, int weight_normalize,
                                      int bias_term, float lr_mult, float lr_decay_mult, float bias_mult,
                                      float bias_decay_mult, int weight_filler, float sigma)
{
    fprintf(stderr, "Connected Layer:    %d inputs, %d outputs, weight_normalize: %d, bias_term: %d\n",
            inputs, outputs, weight_normalize, bias_term);
    connected_layer *layer = calloc(1, sizeof(connected_layer));
    layer->lr_mult = lr_mult;
    layer->lr_decay_mult = lr_decay_mult;
    layer->bias_mult = bias_mult;
    layer->bias_decay_mult = bias_decay_mult;

    layer->weight_normalize = weight_normalize;
    layer->bias_term = bias_term;
    layer->inputs = inputs;
    layer->outputs = outputs;
    layer->batch = batch;
    layer->output = calloc(batch*outputs, sizeof(float));
    layer->delta = calloc(batch*outputs, sizeof(float));

    layer->weights = calloc(inputs*outputs, sizeof(float));  // layer->outputs is the number of rows
    if(weight_filler == 1){   // xavier
        float scale = 1.0F/(inputs);
        for(int i = 0; i < inputs*outputs; ++i) layer->weights[i] = scale*rand_uniform(-1, 1);
    } else if(weight_filler == 2){   // gaussian
        for(int i = 0; i < inputs*outputs; ++i) layer->weights[i] = rand_normal_me(0, sigma);
    } else {
        fprintf(stderr, "weight_filler not support\n");
        exit(-1);
    }
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
    if(layer->bias_term){
        for(int i = 0; i < layer->batch; ++i){
            memcpy(layer->output + layer->outputs * i, layer->biases, layer->outputs*sizeof(float));
        }
    }else{
        memset(layer->output, 0, layer->batch * layer->outputs * sizeof(float));
    }
    if(layer->weight_normalize){
        for(int i = 0; i < layer->outputs; i++){
            float sum = 1e-6;
            for(int j = 0; j < layer->inputs; j++){
                float temp = layer->weights[i * layer->inputs + j];
                sum += temp * temp;
            }
            float scale = sqrtf(sum);
            for(int j = 0; j < layer->inputs; j++){
                layer->weights[i * layer->inputs + j] /= scale;
            }
        }
    }

    float *a = input;
    float *b = layer->weights;  // layer->outputs is the number of rows
    float *c = layer->output;
    int m = layer->batch;
    int n = layer->outputs;
    int k = layer->inputs;
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
    int all_outputs = layer->outputs * layer->batch;
    for(int i = 0; i < all_outputs; ++i){
        layer->output[i] = activate(layer->output[i], layer->activation);
    }

    /*
    float max = -FLT_MAX, min = FLT_MAX;
    for(int i = 0; i < layer->batch * layer->outputs; ++i){
    	if(layer->output[i] > max) max = layer->output[i];
    	if(layer->output[i] < min) min = layer->output[i];
    }
    printf("forward_connected_layer max: %f, min: %f\n", max, min);*/
}

void update_connected_layer(connected_layer *layer, float learning_rate, float momentum, float decay)
{
    for(int i = 0; i < layer->outputs; i ++){
        layer->bias_updates[i] += -decay * layer->bias_decay_mult * layer->batch * layer->biases[i];
        layer->biases[i] += learning_rate * layer->bias_mult / layer->batch * layer->bias_updates[i];
        layer->bias_updates[i] *= momentum;
    }

    int size = layer->inputs*layer->outputs;
    for(int i = 0; i < size; i ++){
        layer->weight_updates[i] += -decay * layer->lr_decay_mult *layer->batch*layer->weights[i];
        layer->weights[i] += learning_rate * layer->lr_mult / layer->batch * layer->weight_updates[i];
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
    int m = layer->outputs;
    int n = layer->inputs;
    int k = layer->batch;
    float *a = layer->delta;
    float *b = input;
    float *c = layer->weight_updates;  // layer->outputs is the number of rows
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    if(delta) {
        m = layer->batch;
        n = layer->inputs;
        k = layer->outputs;
        a = layer->delta;
        b = layer->weights;
        c = delta;
        gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
    }
}

#ifdef GPU

void update_connected_layer_gpu(connected_layer *layer, float learning_rate, float momentum, float decay)
{
    axpy_gpu(layer->outputs, -decay * layer->bias_decay_mult *layer->batch, layer->biases_gpu, 1, layer->bias_updates_gpu, 1);
    axpy_gpu(layer->outputs, learning_rate * layer->bias_mult /layer->batch, layer->bias_updates_gpu, 1, layer->biases_gpu, 1);
    scal_gpu(layer->outputs, momentum, layer->bias_updates_gpu, 1);

    axpy_gpu(layer->inputs*layer->outputs, -decay * layer->lr_decay_mult * layer->batch, layer->weights_gpu, 1, layer->weight_updates_gpu, 1);
    axpy_gpu(layer->inputs*layer->outputs, learning_rate * layer->lr_mult / layer->batch, layer->weight_updates_gpu, 1, layer->weights_gpu, 1);
    scal_gpu(layer->inputs*layer->outputs, momentum, layer->weight_updates_gpu, 1);

}

void forward_connected_layer_gpu(connected_layer *layer, float *input)
{
    if(layer->weight_normalize){
        weight_normalize_gpu(layer->inputs, layer->outputs, layer->weights_gpu);
    }

    int m = layer->batch;
    int n = layer->outputs;
    int k = layer->inputs;
    float * a = input;
    float * b = layer->weights_gpu;
    float * c = layer->output_gpu;
    gemm_gpu(0, 1, m, n, k, 1, a, k, b, k, 0, c, n);
    if(layer->bias_term){
        add_bias_gpu(layer->output_gpu, layer->biases_gpu, layer->batch, layer->outputs, 1);
    }

    activate_array_gpu(layer->output_gpu, layer->outputs*layer->batch, layer->activation);

    /*
    char cuda_compare_error_string[128] = {0};
    sprintf(cuda_compare_error_string, "\n%s", "forward_connected_layer_gpu output");
    cuda_compare(layer->output_gpu, layer->output, layer->batch * layer->outputs, cuda_compare_error_string); */
}

void backward_connected_layer_gpu(connected_layer *layer, float *input, float *delta)
{
    gradient_array_gpu(layer->output_gpu, layer->outputs*layer->batch, layer->activation, layer->delta_gpu);
    backward_bias_gpu(layer->bias_updates_gpu, layer->delta_gpu, layer->batch, layer->outputs, 1);

    int m = layer->outputs;
    int n = layer->inputs;
    int k = layer->batch;
    float * a = layer->delta_gpu;
    float * b = input;
    float * c = layer->weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = layer->batch;
    n = layer->inputs;
    k = layer->outputs;
    a = layer->delta_gpu;
    b = layer->weights_gpu;
    c = delta;
    if(c) {
        gemm_gpu(0,0,m,n,k,1,a,k,b,n,0,c,n);
    }

    /*char cuda_compare_error_string[128] = {0};
    sprintf(cuda_compare_error_string, "\n%s", "backward_connected_layer_gpu delta");
    cuda_compare(layer->delta_gpu, layer->delta, layer->batch * layer->outputs, cuda_compare_error_string);*/
}

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

#endif
