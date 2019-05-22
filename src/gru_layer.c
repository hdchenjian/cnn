#include "gru_layer.h"

image get_gru_image(const gru_layer *layer)
{
    int h = 1;
    int w = 1;
    int c = layer->outputs;
    return float_to_image(h,w,c,NULL);
}

gru_layer *make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize)
{
    fprintf(stderr, "GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
    gru_layer *l = calloc(1, sizeof(gru_layer));
    l->batch = batch;
    l->steps = steps;
    l->inputs = inputs;
    l->outputs = outputs;

    int weight_normalize = 0;
    int bias_term = 1;
    float lr_mult = 1;
    float lr_decay_mult = 1;
    float bias_mult = 1;
    float bias_decay_mult = 0;
    int weight_filler = 1;
    float sigma = 0;
    int connected_layer_batch = batch * steps;
    ACTIVATION activation = LINEAR;
    fprintf(stderr, "\t");
    l->uz = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->uz->batch = batch;

    fprintf(stderr, "\t");
    l->ur = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->ur->batch = batch;

    fprintf(stderr, "\t");
    l->uh = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->uh->batch = batch;
    
    fprintf(stderr, "\t");
    l->wz = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->wz->batch = batch;


    fprintf(stderr, "\t");
    l->wr = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->wr->batch = batch;

    fprintf(stderr, "\t");
    l->wh = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->wh->batch = batch;

    l->output = calloc(outputs*batch*steps, sizeof(float));
    l->delta = calloc(outputs*batch*steps, sizeof(float));
    l->state = calloc(outputs*batch, sizeof(float));
    l->prev_state = calloc(outputs*batch, sizeof(float));
    l->forgot_state = calloc(outputs*batch, sizeof(float));
    l->forgot_delta = calloc(outputs*batch, sizeof(float));

    l->r_cpu = calloc(outputs*batch, sizeof(float));
    l->z_cpu = calloc(outputs*batch, sizeof(float));
    l->h_cpu = calloc(outputs*batch, sizeof(float));

#ifdef GPU
    l->output_gpu = cuda_make_array(0, batch*outputs*steps);
    l->delta_gpu = cuda_make_array(0, batch*outputs*steps);
    l->state_gpu = cuda_make_array(0, batch*outputs);
    l->prev_state_gpu = cuda_make_array(0, batch*outputs);
    l->forgot_state_gpu = cuda_make_array(0, batch*outputs);
    l->forgot_delta_gpu = cuda_make_array(0, batch*outputs);
    l->r_gpu = cuda_make_array(0, batch*outputs);
    l->z_gpu = cuda_make_array(0, batch*outputs);
    l->h_gpu = cuda_make_array(0, batch*outputs);
#endif

    return l;
}

void free_gru_layer(void *input)
{
    gru_layer *layer = (gru_layer *)input;
    if(layer->output) free_ptr((void *)&(layer->output));
    if(layer->delta) free_ptr((void *)&(layer->delta));
    if(layer->state) free_ptr((void *)&(layer->state));
    if(layer->prev_state) free_ptr((void *)&(layer->prev_state));
    if(layer->forgot_state) free_ptr((void *)&(layer->forgot_state));
    if(layer->forgot_delta) free_ptr((void *)&(layer->forgot_delta));
    if(layer->r_cpu) free_ptr((void *)&(layer->r_cpu));
    if(layer->z_cpu) free_ptr((void *)&(layer->z_cpu));
    if(layer->h_cpu) free_ptr((void *)&(layer->h_cpu));

    free_connected_layer(layer->wr);
    free_connected_layer(layer->wz);
    free_connected_layer(layer->wh);
    free_connected_layer(layer->ur);
    free_connected_layer(layer->uz);
    free_connected_layer(layer->uh);
#ifdef GPU
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
    if(layer->state_gpu) cuda_free(layer->state_gpu);
    if(layer->prev_state_gpu) cuda_free(layer->prev_state_gpu);
    if(layer->forgot_state_gpu) cuda_free(layer->forgot_state_gpu);
    if(layer->forgot_delta_gpu) cuda_free(layer->forgot_delta_gpu);
    if(layer->r_gpu) cuda_free(layer->r_gpu);
    if(layer->z_gpu) cuda_free(layer->z_gpu);
    if(layer->h_gpu) cuda_free(layer->h_gpu);
#endif
    free_ptr((void *)&layer);
}

void update_gru_layer(const gru_layer *l, float learning_rate, float momentum, float decay)
{
    update_connected_layer(l->ur, learning_rate, momentum, decay);
    update_connected_layer(l->uz, learning_rate, momentum, decay);
    update_connected_layer(l->uh, learning_rate, momentum, decay);
    update_connected_layer(l->wr, learning_rate, momentum, decay);
    update_connected_layer(l->uz, learning_rate, momentum, decay);
    update_connected_layer(l->uh, learning_rate, momentum, decay);
}

void forward_gru_layer(gru_layer *l, float *input, int test)
{
    if(0 == test){    // 0: train, 1: valid
        copy_cpu(l->outputs*l->batch, l->state, 1, l->prev_state, 1);
    }
    for(int i = 0; i < l->steps; ++i) {
        forward_connected_layer(l->wz, l->state, test);
        forward_connected_layer(l->wr, l->state, test);
        forward_connected_layer(l->uz, input, test);
        forward_connected_layer(l->ur, input, test);
        forward_connected_layer(l->uh, input, test);
        copy_cpu(l->outputs*l->batch, l->uz->output, 1, l->z_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->wz->output, 1, l->z_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->ur->output, 1, l->r_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->wr->output, 1, l->r_cpu, 1);
        activate_array(l->z_cpu, l->outputs*l->batch, LOGISTIC);
        activate_array(l->r_cpu, l->outputs*l->batch, LOGISTIC);

        copy_cpu(l->outputs*l->batch, l->state, 1, l->forgot_state, 1);
        mul_cpu(l->outputs*l->batch, l->r_cpu, 1, l->forgot_state, 1);
        forward_connected_layer(l->wh, l->forgot_state, test);
        copy_cpu(l->outputs*l->batch, l->uh->output, 1, l->h_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->wh->output, 1, l->h_cpu, 1);
        activate_array(l->h_cpu, l->outputs*l->batch, TANH);
        //activate_array(l->h_cpu, l->outputs*l->batch, LOGISTIC);

        int N = l->outputs*l->batch;
        for(int j = 0; j < N; ++j){
            l->output[j] = l->z_cpu[j] * l->h_cpu[j] + (1 - l->z_cpu[j]) * l->state[j];
        }

        copy_cpu(l->outputs*l->batch, l->output, 1, l->state, 1);

        input += l->inputs*l->batch;
        l->output += l->outputs*l->batch;
        increment_layer(l->uz, 1);
        increment_layer(l->ur, 1);
        increment_layer(l->uh, 1);
        increment_layer(l->wz, 1);
        increment_layer(l->wr, 1);
        increment_layer(l->wh, 1);
    }
    l->output -= l->outputs*l->batch*l->steps;
    increment_layer(l->uz, -l->steps);
    increment_layer(l->ur, -l->steps);
    increment_layer(l->uh, -l->steps);
    increment_layer(l->wz, -l->steps);
    increment_layer(l->wr, -l->steps);
    increment_layer(l->wh, -l->steps);
}

void backward_gru_layer(gru_layer *l, float *input, float *delta, int test)
{
    increment_layer(l->uz, l->steps);
    increment_layer(l->ur, l->steps);
    increment_layer(l->uh, l->steps);
    increment_layer(l->wz, l->steps);
    increment_layer(l->wr, l->steps);
    increment_layer(l->wh, l->steps);

    input += l->inputs*l->batch*l->steps;
    if(delta) delta += l->inputs*l->batch*l->steps;
    l->output += l->outputs*l->batch*l->steps;
    l->delta += l->outputs*l->batch*l->steps;
    float *end_state = l->output - l->outputs*l->batch;
    for(int i = l->steps-1; i >= 0; --i) {
        increment_layer(l->uz, -1);
        increment_layer(l->ur, -1);
        increment_layer(l->uh, -1);
        increment_layer(l->wz, -1);
        increment_layer(l->wr, -1);
        increment_layer(l->wh, -1);
        input -= l->inputs*l->batch;
        if (delta) delta -= l->inputs*l->batch;
        l->output -= l->outputs*l->batch;
        l->delta -= l->outputs*l->batch;

        if(i != 0) copy_cpu(l->outputs*l->batch, l->output - l->outputs*l->batch, 1, l->state, 1);
        else copy_cpu(l->outputs*l->batch, l->prev_state, 1, l->state, 1);
        float *prev_delta = (i == 0) ? 0 : l->delta - l->outputs*l->batch;

        copy_cpu(l->outputs*l->batch, l->uz->output, 1, l->z_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->wz->output, 1, l->z_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->ur->output, 1, l->r_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->wr->output, 1, l->r_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->uh->output, 1, l->h_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->wh->output, 1, l->h_cpu, 1);
        activate_array(l->z_cpu, l->outputs*l->batch, LOGISTIC);
        activate_array(l->r_cpu, l->outputs*l->batch, LOGISTIC);
        activate_array(l->h_cpu, l->outputs*l->batch, TANH);

        weighted_delta_cpu(l->outputs*l->batch, l->state, l->h_cpu, l->z_cpu,
                           prev_delta, l->uh->delta, l->uz->delta, l->delta);
        gradient_array(l->h_cpu, l->outputs*l->batch, TANH, l->uh->delta);

        copy_cpu(l->outputs*l->batch, l->uh->delta, 1, l->wh->delta, 1);
        copy_cpu(l->outputs*l->batch, l->state, 1, l->forgot_state, 1);
        mul_cpu(l->outputs*l->batch, l->r_cpu, 1, l->forgot_state, 1);
        backward_connected_layer(l->wh, l->forgot_state, l->forgot_delta, test);

        if(prev_delta) mult_add_into_cpu(l->outputs*l->batch, l->forgot_delta, l->r_cpu, prev_delta);
        mult_add_into_cpu(l->outputs*l->batch, l->forgot_delta, l->state, l->ur->delta);
        gradient_array(l->r_cpu, l->outputs*l->batch, LOGISTIC, l->ur->delta);
        copy_cpu(l->outputs*l->batch, l->ur->delta, 1, l->wr->delta, 1);
        gradient_array(l->z_cpu, l->outputs*l->batch, LOGISTIC, l->uz->delta);
        copy_cpu(l->outputs*l->batch, l->uz->delta, 1, l->wz->delta, 1);
        backward_connected_layer(l->wr, l->state, prev_delta, test);
        backward_connected_layer(l->wz, l->state, prev_delta, test);

        backward_connected_layer(l->uh, input, delta, test);
        backward_connected_layer(l->ur, input, delta, test);
        backward_connected_layer(l->uz, input, delta, test);
    }
    copy_cpu(l->outputs*l->batch, end_state, 1, l->state, 1);
}

#ifdef GPU

void update_gru_layer_gpu(const gru_layer *l, float learning_rate, float momentum, float decay)
{
    update_connected_layer_gpu(l->ur, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->uz, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->uh, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->wr, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->uz, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->uh, learning_rate, momentum, decay);
}

void forward_gru_layer_gpu(gru_layer *l, float *input, int test)
{
    if(0 == test){    // 0: train, 1: valid
        copy_gpu(l->outputs*l->batch, l->state_gpu, 1, l->prev_state_gpu, 1);
    }
    for(int i = 0; i < l->steps; ++i) {
        forward_connected_layer_gpu(l->wz, l->state_gpu, test);
        forward_connected_layer_gpu(l->wr, l->state_gpu, test);
        forward_connected_layer_gpu(l->uz, input, test);
        forward_connected_layer_gpu(l->ur, input, test);
        forward_connected_layer_gpu(l->uh, input, test);

        copy_gpu(l->outputs*l->batch, l->uz->output_gpu, 1, l->z_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->wz->output_gpu, 1, l->z_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->ur->output_gpu, 1, l->r_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->wr->output_gpu, 1, l->r_gpu, 1);
        activate_array_gpu(l->z_gpu, l->outputs*l->batch, LOGISTIC);
        activate_array_gpu(l->r_gpu, l->outputs*l->batch, LOGISTIC);


        copy_gpu(l->outputs*l->batch, l->state_gpu, 1, l->forgot_state_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->r_gpu, 1, l->forgot_state_gpu, 1);
        forward_connected_layer_gpu(l->wh, l->forgot_state_gpu, test);
        copy_gpu(l->outputs*l->batch, l->uh->output_gpu, 1, l->h_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->wh->output_gpu, 1, l->h_gpu, 1);
        activate_array_gpu(l->h_gpu, l->outputs*l->batch, TANH);
        //activate_array(l->h_gpu, l->outputs*l->batch, LOGISTIC);

        weighted_sum_gpu(l->state_gpu, l->h_gpu, l->z_gpu, l->outputs*l->batch, l->output_gpu);
        copy_gpu(l->outputs*l->batch, l->output_gpu, 1, l->state_gpu, 1);

        input += l->inputs*l->batch;
        l->output_gpu += l->outputs*l->batch;
        increment_layer(l->uz, 1);
        increment_layer(l->ur, 1);
        increment_layer(l->uh, 1);
        increment_layer(l->wz, 1);
        increment_layer(l->wr, 1);
        increment_layer(l->wh, 1);
    }
    l->output_gpu -= l->outputs*l->batch*l->steps;
    increment_layer(l->uz, -l->steps);
    increment_layer(l->ur, -l->steps);
    increment_layer(l->uh, -l->steps);
    increment_layer(l->wz, -l->steps);
    increment_layer(l->wr, -l->steps);
    increment_layer(l->wh, -l->steps);
}

void backward_gru_layer_gpu(gru_layer *l, float *input, float *delta, int test)
{
    increment_layer(l->uz, l->steps);
    increment_layer(l->ur, l->steps);
    increment_layer(l->uh, l->steps);
    increment_layer(l->wz, l->steps);
    increment_layer(l->wr, l->steps);
    increment_layer(l->wh, l->steps);

    input += l->inputs*l->batch*l->steps;
    if(delta) delta += l->inputs*l->batch*l->steps;
    l->output_gpu += l->outputs*l->batch*l->steps;
    l->delta_gpu += l->outputs*l->batch*l->steps;
    float *end_state = l->output_gpu - l->outputs*l->batch;
    for(int i = l->steps-1; i >= 0; --i) {
        increment_layer(l->uz, -1);
        increment_layer(l->ur, -1);
        increment_layer(l->uh, -1);
        increment_layer(l->wz, -1);
        increment_layer(l->wr, -1);
        increment_layer(l->wh, -1);
        input -= l->inputs*l->batch;
        if (delta) delta -= l->inputs*l->batch;
        l->output_gpu -= l->outputs*l->batch;
        l->delta_gpu -= l->outputs*l->batch;

        if(i != 0) copy_gpu(l->outputs*l->batch, l->output_gpu - l->outputs*l->batch, 1, l->state_gpu, 1);
        else copy_gpu(l->outputs*l->batch, l->prev_state_gpu, 1, l->state_gpu, 1);
        float *prev_delta_gpu = (i == 0) ? 0 : l->delta_gpu - l->outputs*l->batch;

        copy_gpu(l->outputs*l->batch, l->uz->output_gpu, 1, l->z_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->wz->output_gpu, 1, l->z_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->ur->output_gpu, 1, l->r_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->wr->output_gpu, 1, l->r_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->uh->output_gpu, 1, l->h_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->wh->output_gpu, 1, l->h_gpu, 1);
        activate_array_gpu(l->z_gpu, l->outputs*l->batch, LOGISTIC);
        activate_array_gpu(l->r_gpu, l->outputs*l->batch, LOGISTIC);
        activate_array_gpu(l->h_gpu, l->outputs*l->batch, TANH);

        weighted_delta_gpu(l->outputs*l->batch, l->state_gpu, l->h_gpu, l->z_gpu,
                           prev_delta_gpu, l->uh->delta_gpu, l->uz->delta_gpu, l->delta_gpu);
        gradient_array_gpu(l->h_gpu, l->outputs*l->batch, TANH, l->uh->delta_gpu);

        copy_gpu(l->outputs*l->batch, l->uh->delta_gpu, 1, l->wh->delta_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->state_gpu, 1, l->forgot_state_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->r_gpu, 1, l->forgot_state_gpu, 1);
        backward_connected_layer_gpu(l->wh, l->forgot_state_gpu, l->forgot_delta_gpu, test);

        if(prev_delta_gpu) mult_add_into_gpu(l->outputs*l->batch, l->forgot_delta_gpu, l->r_gpu, prev_delta_gpu);
        mult_add_into_gpu(l->outputs*l->batch, l->forgot_delta_gpu, l->state_gpu, l->ur->delta_gpu);
        gradient_array_gpu(l->r_gpu, l->outputs*l->batch, LOGISTIC, l->ur->delta_gpu);
        copy_gpu(l->outputs*l->batch, l->ur->delta_gpu, 1, l->wr->delta_gpu, 1);
        gradient_array_gpu(l->z_gpu, l->outputs*l->batch, LOGISTIC, l->uz->delta_gpu);
        copy_gpu(l->outputs*l->batch, l->uz->delta_gpu, 1, l->wz->delta_gpu, 1);
        backward_connected_layer_gpu(l->wr, l->state_gpu, prev_delta_gpu, test);
        backward_connected_layer_gpu(l->wz, l->state_gpu, prev_delta_gpu, test);

        backward_connected_layer_gpu(l->uh, input, delta, test);
        backward_connected_layer_gpu(l->ur, input, delta, test);
        backward_connected_layer_gpu(l->uz, input, delta, test);
    }
    copy_gpu(l->outputs*l->batch, end_state, 1, l->state_gpu, 1);
}
#endif
