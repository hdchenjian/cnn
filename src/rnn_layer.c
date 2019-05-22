#include "rnn_layer.h"

image get_rnn_image(const rnn_layer *layer)
{
    int h = 1;
    int w = 1;
    int c = layer->outputs;
    return float_to_image(h,w,c,NULL);
}

rnn_layer *make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize)
{
    fprintf(stderr, "RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
    rnn_layer *l = calloc(1, sizeof(rnn_layer));
    l->batch = batch;
    l->steps = steps;
    l->inputs = inputs;

    int weight_normalize = 0;
    int bias_term = 1;
    float lr_mult = 1;
    float lr_decay_mult = 1;
    float bias_mult = 1;
    float bias_decay_mult = 0;
    int weight_filler = 1;
    float sigma = 0;
    int connected_layer_batch = batch * steps;
    fprintf(stderr, "\t");
    l->input_layer = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize,
                                          bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                          sigma, batch_normalize, 1, 0);
    l->input_layer->batch = batch;

    fprintf(stderr, "\t");
    l->self_layer = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                         bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                         sigma, batch_normalize, 1, 0);
    l->self_layer->batch = batch;

    fprintf(stderr, "\t");
    l->output_layer = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                           bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                           sigma, batch_normalize, 1, 0);
    l->output_layer->batch = batch;

    l->outputs = outputs;
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;

    l->state = calloc(batch*outputs, sizeof(float));
    l->prev_state = calloc(batch*outputs, sizeof(float));
#ifdef GPU
    l->state_gpu = cuda_make_array(0, batch*outputs);
    l->prev_state_gpu = cuda_make_array(0, batch*outputs);
    l->output_gpu = l->output_layer->output_gpu;
    l->delta_gpu = l->output_layer->delta_gpu;
#endif

    return l;
}

void free_rnn_layer(void *input)
{
    rnn_layer *layer = (rnn_layer *)input;
    if(layer->state) free_ptr((void *)&(layer->state));
    if(layer->prev_state) free_ptr((void *)&(layer->prev_state));
    free_connected_layer(layer->input_layer);
    free_connected_layer(layer->self_layer);
    free_connected_layer(layer->output_layer);
#ifdef GPU
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#endif
    free_ptr((void *)&layer);
}

void update_rnn_layer(const rnn_layer *l, float learning_rate, float momentum, float decay)
{
    update_connected_layer(l->input_layer, learning_rate, momentum, decay);
    update_connected_layer(l->self_layer, learning_rate, momentum, decay);
    update_connected_layer(l->output_layer, learning_rate, momentum, decay);
}

void increment_layer(connected_layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    if(l->x) l->x += num;
    if(l->x_norm) l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    if(l->x_gpu) l->x_gpu += num;
    if(l->x_norm_gpu) l->x_norm_gpu += num;
#endif
}

void forward_rnn_layer(const rnn_layer *l, float *input, int test)
{
    if(0 == test){    // 0: train, 1: valid
        copy_cpu(l->outputs*l->batch, l->state, 1, l->prev_state, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->input_layer->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->self_layer->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->output_layer->delta, 1);
    }
    for (int i = 0; i < l->steps; ++i) {
        forward_connected_layer(l->input_layer, input, test);
        forward_connected_layer(l->self_layer, l->state, test);
        copy_cpu(l->outputs * l->batch, l->input_layer->output, 1, l->state, 1);
        axpy_cpu(l->outputs * l->batch, 1, l->self_layer->output, 1, l->state, 1);

        forward_connected_layer(l->output_layer, l->state, test);
        input += l->inputs*l->batch;
        increment_layer(l->input_layer, 1);
        increment_layer(l->self_layer, 1);
        increment_layer(l->output_layer, 1);
    }
    increment_layer(l->input_layer, -l->steps);  // restore the l->input_layer->output modify
    increment_layer(l->self_layer,  -l->steps);
    increment_layer(l->output_layer,  -l->steps);
}

void backward_rnn_layer(const rnn_layer *l, float *input, float *delta, int test)
{
    input = input + l->inputs*l->batch*l->steps;
    if(delta){
        delta = delta + l->inputs*l->batch*l->steps;
    }
    increment_layer(l->input_layer, l->steps);
    increment_layer(l->self_layer, l->steps);
    increment_layer(l->output_layer, l->steps);
    float *last_input = l->input_layer->output - l->outputs*l->batch;
    float *last_self = l->self_layer->output - l->outputs*l->batch;
    for(int i = l->steps-1; i >= 0; --i) {
        increment_layer(l->input_layer, -1);
        increment_layer(l->self_layer, -1);
        increment_layer(l->output_layer, -1);
        input -= l->inputs*l->batch;
        if(delta){
            delta -= l->inputs*l->batch;
        }
        copy_cpu(l->outputs * l->batch, l->input_layer->output, 1, l->state, 1);
        axpy_cpu(l->outputs * l->batch, 1, l->self_layer->output, 1, l->state, 1);
        backward_connected_layer(l->output_layer, l->state, l->self_layer->delta, test);

        if(i != 0) {
            copy_cpu(l->outputs * l->batch, l->input_layer->output - l->outputs*l->batch, 1, l->state, 1);
            axpy_cpu(l->outputs * l->batch, 1, l->self_layer->output - l->outputs*l->batch, 1, l->state, 1);
        }else {
            copy_cpu(l->outputs*l->batch, l->prev_state, 1, l->state, 1);
        }
        float *delta_self_layer = NULL;
        if(i == 0) delta_self_layer = 0;
        else delta_self_layer = l->self_layer->delta - l->outputs*l->batch;
        backward_connected_layer(l->self_layer, l->state, delta_self_layer, test);

        copy_cpu(l->outputs*l->batch, l->self_layer->delta, 1, l->input_layer->delta, 1);
        backward_connected_layer(l->input_layer, input, delta, test);
    }
    copy_cpu(l->outputs * l->batch, last_input, 1, l->state, 1);
    axpy_cpu(l->outputs * l->batch, 1, last_self, 1, l->state, 1);
    //printf("backward_rnn_layer %p\n", l->input_layer->output);
}

#ifdef GPU

void pull_rnn_layer(const rnn_layer *l)
{
    pull_connected_layer(l->input_layer);
    pull_connected_layer(l->self_layer);
    pull_connected_layer(l->output_layer);
}

void push_rnn_layer(const rnn_layer *l)
{
    push_connected_layer(l->input_layer);
    push_connected_layer(l->self_layer);
    push_connected_layer(l->output_layer);
}

void update_rnn_layer_gpu(const rnn_layer *l, float learning_rate, float momentum, float decay)
{
    update_connected_layer_gpu(l->input_layer, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->self_layer, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->output_layer, learning_rate, momentum, decay);
}

void forward_rnn_layer_gpu(const rnn_layer *l, float *input, int test)
{
    if(0 == test){    // 0: train, 1: valid
        copy_gpu(l->outputs*l->batch, l->state_gpu, 1, l->prev_state_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->input_layer->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->self_layer->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->output_layer->delta_gpu, 1);
    }

    for(int i = 0; i < l->steps; ++i) {
        forward_connected_layer_gpu(l->input_layer, input, test);
        forward_connected_layer_gpu(l->self_layer, l->state_gpu, test);

        copy_gpu(l->outputs * l->batch, l->input_layer->output_gpu, 1, l->state_gpu, 1);
        axpy_gpu(l->outputs * l->batch, 1, l->self_layer->output_gpu, 1, l->state_gpu, 1);

        forward_connected_layer_gpu(l->output_layer, l->state_gpu, test);
        input += l->inputs*l->batch;
        increment_layer(l->input_layer, 1);
        increment_layer(l->self_layer, 1);
        increment_layer(l->output_layer, 1);
    }
    increment_layer(l->input_layer, -l->steps);  // restore the l->input_layer->output modify
    increment_layer(l->self_layer,  -l->steps);
    increment_layer(l->output_layer,  -l->steps);
}

void backward_rnn_layer_gpu(const rnn_layer *l, float *input, float *delta, int test)
{
    input = input + l->inputs*l->batch*l->steps;
    if(delta){
        delta = delta + l->inputs*l->batch*l->steps;
    }
    increment_layer(l->input_layer, l->steps);
    increment_layer(l->self_layer, l->steps);
    increment_layer(l->output_layer, l->steps);

    float *last_input = l->input_layer->output_gpu - l->outputs*l->batch;
    float *last_self = l->self_layer->output_gpu - l->outputs*l->batch;
    for(int i = l->steps-1; i >= 0; --i) {
        increment_layer(l->input_layer, -1);
        increment_layer(l->self_layer, -1);
        increment_layer(l->output_layer, -1);
        input -= l->inputs*l->batch;
        if(delta){
            delta -= l->inputs*l->batch;
        }   
        copy_gpu(l->outputs * l->batch, l->input_layer->output_gpu, 1, l->state_gpu, 1);
        axpy_gpu(l->outputs * l->batch, 1, l->self_layer->output_gpu, 1, l->state_gpu, 1);
        backward_connected_layer_gpu(l->output_layer, l->state_gpu, l->self_layer->delta_gpu, test);

        if(i != 0) {
            copy_gpu(l->outputs * l->batch, l->input_layer->output_gpu - l->outputs*l->batch, 1, l->state_gpu, 1);
            axpy_gpu(l->outputs * l->batch, 1, l->self_layer->output_gpu - l->outputs*l->batch, 1, l->state_gpu, 1);
        }else {
            copy_gpu(l->outputs*l->batch, l->prev_state_gpu, 1, l->state_gpu, 1);
        }

        float *delta_self_layer = NULL;
        if(i == 0) delta_self_layer = 0;
        else delta_self_layer = l->self_layer->delta_gpu - l->outputs*l->batch;
        backward_connected_layer_gpu(l->self_layer, l->state_gpu, delta_self_layer, test);
        
        copy_gpu(l->outputs*l->batch, l->self_layer->delta_gpu, 1, l->input_layer->delta_gpu, 1);
        backward_connected_layer_gpu(l->input_layer, input, delta, test);
    }
    copy_gpu(l->outputs * l->batch, last_input, 1, l->state_gpu, 1);
    axpy_gpu(l->outputs * l->batch, 1, last_self, 1, l->state_gpu, 1);
}
#endif
