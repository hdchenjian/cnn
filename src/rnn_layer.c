#include "rnn_layer.h"

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
    l->input_layer = malloc(sizeof(connected_layer));
    fprintf(stderr, "\t\t");
    l->input_layer = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize, bias_term,
                                          lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                          sigma, batch_normalize);
    l->input_layer->batch = batch;

    l->self_layer = malloc(sizeof(connected_layer));
    fprintf(stderr, "\t\t");
    l->self_layer = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize, bias_term,
                                          lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                          sigma, batch_normalize);
    l->self_layer->batch = batch;

    l->output_layer = malloc(sizeof(connected_layer));
    fprintf(stderr, "\t\t");
    l->output_layer = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize, bias_term,
                                          lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                          sigma, batch_normalize);
    l->output_layer->batch = batch;

    l->outputs = outputs;
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;

    l->state = calloc(batch*outputs, sizeof(float));
    l->prev_state = calloc(batch*outputs, sizeof(float));

#ifdef GPU
    l->state_gpu = cuda_make_array(0, batch*outputs*steps);
    l->prev_state_gpu = cuda_make_array(0, batch*outputs*steps);
    l->output_gpu = l->output_layer->output_gpu;
    l->delta_gpu = l->output_layer->delta_gpu;
#endif

    return l;
}

void free_rnn_layer(void *input)
{
    rnn_layer *layer = (rnn_layer *)input;
    if(layer->state) free_ptr(layer->state);
    if(layer->prev_state) free_ptr(layer->prev_state);
    free_connected_layer(layer->input_layer);
    free_connected_layer(layer->self_layer);
    free_connected_layer(layer->output_layer);
#ifdef GPU
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#endif
    free_ptr(layer);
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
}

void backward_rnn_layer(const rnn_layer *l, float *input, float *delta, int test)
{
    increment_layer(l->input_layer, l->steps-1);
    increment_layer(l->self_layer, l->steps-1);
    increment_layer(l->output_layer, l->steps-1);
    float *last_input = l->input_layer->output_gpu;
    float *last_self = l->self_layer->output_gpu;
    for(int i = l->steps-1; i >= 0; --i) {
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
        float *delta_input_layer = NULL;
        if(delta) delta_input_layer = delta + i*l->inputs*l->batch;
        else delta_input_layer = 0;
        backward_connected_layer(l->input_layer, input + i*l->inputs*l->batch, delta_input_layer, test);

        increment_layer(l->input_layer, -1);
        increment_layer(l->self_layer, -1);
        increment_layer(l->output_layer, -1);
    }
    copy_cpu(l->outputs * l->batch, last_input, 1, l->state, 1);
    axpy_cpu(l->outputs * l->batch, 1, last_self, 1, l->state, 1);
}

#ifdef GPU

void pull_rnn_layer(layer l)
{
    pull_connected_layer(*(l.input_layer));
    pull_connected_layer(*(l.self_layer));
    pull_connected_layer(*(l.output_layer));
}

void push_rnn_layer(layer l)
{
    push_connected_layer(*(l.input_layer));
    push_connected_layer(*(l.self_layer));
    push_connected_layer(*(l.output_layer));
}

void update_rnn_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.input_layer),  a);
    update_connected_layer_gpu(*(l.self_layer),   a);
    update_connected_layer_gpu(*(l.output_layer), a);
}

void forward_rnn_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    fill_gpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, input_layer.delta_gpu, 1);

    if(net.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_gpu = net.input_gpu;
        forward_connected_layer_gpu(input_layer, s);

        s.input_gpu = l.state_gpu;
        forward_connected_layer_gpu(self_layer, s);

        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        s.input_gpu = l.state_gpu;
        forward_connected_layer_gpu(output_layer, s);

        net.input_gpu += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

void backward_rnn_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    increment_layer(&input_layer,  l.steps - 1);
    increment_layer(&self_layer,   l.steps - 1);
    increment_layer(&output_layer, l.steps - 1);
    float *last_input = input_layer.output_gpu;
    float *last_self = self_layer.output_gpu;
    for (i = l.steps-1; i >= 0; --i) {
        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = self_layer.delta_gpu;
        backward_connected_layer_gpu(output_layer, s);

        if(i != 0) {
            fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
        }else {
            copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
        }

        copy_gpu(l.outputs*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = (i > 0) ? self_layer.delta_gpu - l.outputs*l.batch : 0;
        if (i == 0) s.delta_gpu = 0;
        backward_connected_layer_gpu(self_layer, s);

        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
        else s.delta_gpu = 0;
        backward_connected_layer_gpu(input_layer, s);

        increment_layer(&input_layer,  -1);
        increment_layer(&self_layer,   -1);
        increment_layer(&output_layer, -1);
    }
    fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
    axpy_gpu(l.outputs * l.batch, 1, last_input, 1, l.state_gpu, 1);
    axpy_gpu(l.outputs * l.batch, 1, last_self, 1, l.state_gpu, 1);
}
#endif
