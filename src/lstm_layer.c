#include "lstm_layer.h"

image get_lstm_image(const lstm_layer *layer)
{
    int h = 1;
    int w = 1;
    int c = layer->outputs;
    return float_to_image(h,w,c,NULL);
}

lstm_layer *make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize)
{
    fprintf(stderr, "LSTM Layer: %d inputs, %d outputs\n", inputs, outputs);
    lstm_layer *l = calloc(1, sizeof(lstm_layer));
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
    l->uf = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->uf->batch = batch;
    fprintf(stderr, "\t");
    l->ui = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->ui->batch = batch;
    fprintf(stderr, "\t");
    l->ug = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->ug->batch = batch;
    fprintf(stderr, "\t");
    l->uo = make_connected_layer(inputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->uo->batch = batch;
    fprintf(stderr, "\t");
    l->wf = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->wf->batch = batch;
    fprintf(stderr, "\t");
    l->wi = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->wi->batch = batch;
    fprintf(stderr, "\t");
    l->wg = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->wg->batch = batch;
    fprintf(stderr, "\t");
    l->wo = make_connected_layer(outputs, outputs, connected_layer_batch, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize, 1, 0);
    l->wo->batch = batch;

    l->output = calloc(outputs*batch*steps, sizeof(float));
    l->delta = calloc(outputs*batch*steps, sizeof(float));
    l->prev_output = calloc(batch*outputs, sizeof(float));
    l->cell_cpu = calloc(batch*outputs*steps, sizeof(float));
    l->prev_cell_cpu = calloc(batch*outputs, sizeof(float));

    l->f_cpu = calloc(batch*outputs, sizeof(float));
    l->i_cpu = calloc(batch*outputs, sizeof(float));
    l->g_cpu = calloc(batch*outputs, sizeof(float));
    l->o_cpu = calloc(batch*outputs, sizeof(float));
    l->c_cpu = calloc(batch*outputs, sizeof(float));
    l->h_cpu = calloc(batch*outputs, sizeof(float));
    l->c_cpu_bak = calloc(batch*outputs, sizeof(float));
    l->h_cpu_bak = calloc(batch*outputs, sizeof(float));
    l->temp_cpu = calloc(batch*outputs, sizeof(float));
    l->temp2_cpu = calloc(batch*outputs, sizeof(float));
    l->temp3_cpu = calloc(batch*outputs, sizeof(float));

#ifdef GPU
    l->output_gpu = cuda_make_array(0, batch*outputs*steps);
    l->delta_gpu = cuda_make_array(0, batch*outputs*steps);
    l->prev_output_gpu = cuda_make_array(0, batch*outputs);
    l->cell_gpu = cuda_make_array(0, batch*outputs*steps);
    l->prev_cell_gpu = cuda_make_array(0, batch*outputs);

    l->f_gpu = cuda_make_array(0, batch*outputs);
    l->i_gpu = cuda_make_array(0, batch*outputs);
    l->g_gpu = cuda_make_array(0, batch*outputs);
    l->o_gpu = cuda_make_array(0, batch*outputs);
    l->c_gpu = cuda_make_array(0, batch*outputs);
    l->h_gpu = cuda_make_array(0, batch*outputs);
    l->c_gpu_bak = cuda_make_array(0, batch*outputs);
    l->h_gpu_bak = cuda_make_array(0, batch*outputs);
    l->temp_gpu =  cuda_make_array(0, batch*outputs);
    l->temp2_gpu = cuda_make_array(0, batch*outputs);
    l->temp3_gpu = cuda_make_array(0, batch*outputs);
#endif

    return l;
}

void free_lstm_layer(void *input)
{
    lstm_layer *layer = (lstm_layer *)input;
    if(layer->output) free_ptr((void *)&(layer->output));
    if(layer->delta) free_ptr((void *)&(layer->delta));
    if(layer->prev_output) free_ptr((void *)&(layer->prev_output));
    if(layer->cell_cpu) free_ptr((void *)&(layer->cell_cpu));
    if(layer->prev_cell_cpu) free_ptr((void *)&(layer->prev_cell_cpu));
    if(layer->f_cpu) free_ptr((void *)&(layer->f_cpu));
    if(layer->i_cpu) free_ptr((void *)&(layer->i_cpu));
    if(layer->g_cpu) free_ptr((void *)&(layer->g_cpu));
    if(layer->o_cpu) free_ptr((void *)&(layer->o_cpu));
    if(layer->c_cpu) free_ptr((void *)&(layer->c_cpu));
    if(layer->h_cpu) free_ptr((void *)&(layer->h_cpu));
    if(layer->c_cpu_bak) free_ptr((void *)&(layer->c_cpu_bak));
    if(layer->h_cpu_bak) free_ptr((void *)&(layer->h_cpu_bak));
    if(layer->temp_cpu) free_ptr((void *)&(layer->temp_cpu));
    if(layer->temp2_cpu) free_ptr((void *)&(layer->temp2_cpu));
    if(layer->temp3_cpu) free_ptr((void *)&(layer->temp3_cpu));

    free_connected_layer(layer->wf);
    free_connected_layer(layer->wi);
    free_connected_layer(layer->wg);
    free_connected_layer(layer->wo);
    free_connected_layer(layer->uf);
    free_connected_layer(layer->ui);
    free_connected_layer(layer->ug);
    free_connected_layer(layer->uo);
#ifdef GPU
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
    if(layer->prev_output_gpu) cuda_free(layer->prev_output_gpu);
    if(layer->cell_gpu) cuda_free(layer->cell_gpu);
    if(layer->prev_cell_gpu) cuda_free(layer->prev_cell_gpu);
    if(layer->f_gpu) cuda_free(layer->f_gpu);
    if(layer->i_gpu) cuda_free(layer->i_gpu);
    if(layer->g_gpu) cuda_free(layer->g_gpu);
    if(layer->o_gpu) cuda_free(layer->o_gpu);
    if(layer->c_gpu) cuda_free(layer->c_gpu);
    if(layer->h_gpu) cuda_free(layer->h_gpu);
    if(layer->c_gpu_bak) cuda_free(layer->c_gpu_bak);
    if(layer->h_gpu_bak) cuda_free(layer->h_gpu_bak);
    if(layer->temp_gpu) cuda_free(layer->temp_gpu);
    if(layer->temp2_gpu) cuda_free(layer->temp2_gpu);
    if(layer->temp3_gpu) cuda_free(layer->temp3_gpu);
#endif
    free_ptr((void *)&layer);
}

void update_lstm_layer(const lstm_layer *l, float learning_rate, float momentum, float decay)
{
    update_connected_layer(l->wf, learning_rate, momentum, decay);
    update_connected_layer(l->wi, learning_rate, momentum, decay);
    update_connected_layer(l->wg, learning_rate, momentum, decay);
    update_connected_layer(l->wo, learning_rate, momentum, decay);
    update_connected_layer(l->uf, learning_rate, momentum, decay);
    update_connected_layer(l->ui, learning_rate, momentum, decay);
    update_connected_layer(l->ug, learning_rate, momentum, decay);
    update_connected_layer(l->uo, learning_rate, momentum, decay);
}

void forward_lstm_layer(lstm_layer *l, float *input, int test)
{
    if(0 == test){    // 0: train, 1: valid
        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->c_cpu_bak, 1);
        copy_cpu(l->outputs*l->batch, l->h_cpu, 1, l->h_cpu_bak, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->wf->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->wi->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->wg->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->wo->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->uf->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->ui->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->ug->delta, 1);
        fill_cpu(l->outputs * l->batch * l->steps, 0, l->uo->delta, 1);
    }
    for(int i = 0; i < l->steps; ++i) {
        forward_connected_layer(l->wf, l->h_cpu, test);
        forward_connected_layer(l->wi, l->h_cpu, test);
        forward_connected_layer(l->wg, l->h_cpu, test);
        forward_connected_layer(l->wo, l->h_cpu, test);
        forward_connected_layer(l->uf, input, test);
        forward_connected_layer(l->ui, input, test);
        forward_connected_layer(l->ug, input, test);
        forward_connected_layer(l->uo, input, test);

        copy_cpu(l->outputs*l->batch, l->wf->output, 1, l->f_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->uf->output, 1, l->f_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->wi->output, 1, l->i_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->ui->output, 1, l->i_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->wg->output, 1, l->g_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->ug->output, 1, l->g_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->wo->output, 1, l->o_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->uo->output, 1, l->o_cpu, 1);

        activate_array(l->f_cpu, l->outputs*l->batch, LOGISTIC);
        activate_array(l->i_cpu, l->outputs*l->batch, LOGISTIC);
        activate_array(l->g_cpu, l->outputs*l->batch, TANH);
        activate_array(l->o_cpu, l->outputs*l->batch, LOGISTIC);

        copy_cpu(l->outputs*l->batch, l->i_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->g_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->f_cpu, 1, l->c_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->temp_cpu, 1, l->c_cpu, 1);

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->h_cpu, 1);
        activate_array(l->h_cpu, l->outputs*l->batch, TANH);
        mul_cpu(l->outputs*l->batch, l->o_cpu, 1, l->h_cpu, 1);

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->cell_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->h_cpu, 1, l->output, 1);

        input += l->inputs*l->batch;
        l->output += l->outputs*l->batch;
        l->cell_cpu += l->outputs*l->batch;
        increment_layer(l->wf, 1);
        increment_layer(l->wi, 1);
        increment_layer(l->wg, 1);
        increment_layer(l->wo, 1);
        increment_layer(l->uf, 1);
        increment_layer(l->ui, 1);
        increment_layer(l->ug, 1);
        increment_layer(l->uo, 1);
    }
    l->output -= l->outputs*l->batch*l->steps;
    l->cell_cpu -= l->outputs*l->batch*l->steps;

    increment_layer(l->wf, -l->steps);
    increment_layer(l->wi, -l->steps);
    increment_layer(l->wg, -l->steps);
    increment_layer(l->wo, -l->steps);
    increment_layer(l->uf, -l->steps);
    increment_layer(l->ui, -l->steps);
    increment_layer(l->ug, -l->steps);
    increment_layer(l->uo, -l->steps);
}

void backward_lstm_layer(lstm_layer *l, float *input, float *delta, int test)
{
    increment_layer(l->wf, l->steps);
    increment_layer(l->wi, l->steps);
    increment_layer(l->wg, l->steps);
    increment_layer(l->wo, l->steps);
    increment_layer(l->uf, l->steps);
    increment_layer(l->ui, l->steps);
    increment_layer(l->ug, l->steps);
    increment_layer(l->uo, l->steps);
    input += l->inputs*l->batch*l->steps;
    if(delta) delta += l->inputs*l->batch*l->steps;
    l->output += l->outputs*l->batch*l->steps;
    l->cell_cpu += l->outputs*l->batch*l->steps;
    l->delta += l->outputs*l->batch*l->steps;
    float *last_c_cpu = l->cell_cpu - l->outputs*l->batch;
    float *last_h_cpu = l->output - l->outputs*l->batch;

    for(int i = l->steps - 1; i >= 0; --i) {
        increment_layer(l->wf, -1);
        increment_layer(l->wi, -1);
        increment_layer(l->wg, -1);
        increment_layer(l->wo, -1);
        increment_layer(l->uf, -1);
        increment_layer(l->ui, -1);
        increment_layer(l->ug, -1);
        increment_layer(l->uo, -1);
        input -= l->inputs*l->batch;
        if (delta) delta -= l->inputs*l->batch;
        l->output -= l->outputs*l->batch;
        l->cell_cpu -= l->outputs*l->batch;
        l->delta -= l->outputs*l->batch;

        copy_cpu(l->outputs*l->batch, l->wf->output, 1, l->f_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->uf->output, 1, l->f_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->wi->output, 1, l->i_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->ui->output, 1, l->i_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->wg->output, 1, l->g_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->ug->output, 1, l->g_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->wo->output, 1, l->o_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, l->uo->output, 1, l->o_cpu, 1);
        activate_array(l->f_cpu, l->outputs*l->batch, LOGISTIC);
        activate_array(l->i_cpu, l->outputs*l->batch, LOGISTIC);
        activate_array(l->g_cpu, l->outputs*l->batch, TANH);
        activate_array(l->o_cpu, l->outputs*l->batch, LOGISTIC);

        if (i != 0) copy_cpu(l->outputs*l->batch, l->cell_cpu - l->outputs*l->batch, 1, l->prev_cell_cpu, 1);
        else copy_cpu(l->outputs*l->batch, l->c_cpu_bak, 1, l->prev_cell_cpu, 1);;
        if (i != 0) copy_cpu(l->outputs*l->batch, l->output - l->outputs*l->batch, 1, l->prev_output, 1);
        else copy_cpu(l->outputs*l->batch, l->h_cpu_bak, 1, l->prev_output, 1);
        copy_cpu(l->outputs*l->batch, l->cell_cpu, 1, l->c_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->output, 1, l->h_cpu, 1);
        float *dh_cpu = (i == 0) ? 0 : l->delta - l->outputs*l->batch;

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->temp_cpu, 1);
        activate_array(l->temp_cpu, l->outputs*l->batch, TANH);
        copy_cpu(l->outputs*l->batch, l->delta, 1, l->temp3_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->temp3_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->o_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->wo->delta, 1);
        backward_connected_layer(l->wo, l->prev_output, dh_cpu, test);

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->uo->delta, 1);
        backward_connected_layer(l->uo, input, delta, test);

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->temp_cpu, 1);
        activate_array(l->temp_cpu, l->outputs*l->batch, TANH);
        copy_cpu(l->outputs*l->batch, l->delta, 1, l->temp2_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->o_cpu, 1, l->temp2_cpu, 1);
        gradient_array(l->temp_cpu, l->outputs*l->batch, TANH, l->temp2_cpu);

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->i_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->g_cpu, l->outputs*l->batch, TANH, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->wg->delta, 1);
        backward_connected_layer(l->wg, l->prev_output, dh_cpu, test);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->ug->delta, 1);
        backward_connected_layer(l->ug, input, delta, test);

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->g_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->i_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->wi->delta, 1);
        backward_connected_layer(l->wi, l->prev_output, dh_cpu, test);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->ui->delta, 1);
        backward_connected_layer(l->ui, input, delta, test);

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->prev_cell_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->f_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->wf->delta, 1);
        backward_connected_layer(l->wf, l->prev_output, dh_cpu, test);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->uf->delta, 1);
        backward_connected_layer(l->uf, input, delta, test);
    }
    copy_cpu(l->outputs * l->batch, last_c_cpu, 1, l->c_cpu, 1);
    copy_cpu(l->outputs * l->batch, last_h_cpu, 1, l->h_cpu, 1);
}

#ifdef GPU
void update_lstm_layer_gpu(lstm_layer *l, float learning_rate, float momentum, float decay)
{
    update_connected_layer_gpu(l->wf, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->wi, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->wg, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->wo, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->uf, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->ui, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->ug, learning_rate, momentum, decay);
    update_connected_layer_gpu(l->uo, learning_rate, momentum, decay);
}

void forward_lstm_layer_gpu(lstm_layer *l, float *input, int test)
{
    if(0 == test){    // 0: train, 1: valid
        copy_gpu(l->outputs*l->batch, l->c_gpu, 1, l->c_gpu_bak, 1);
        copy_gpu(l->outputs*l->batch, l->h_gpu, 1, l->h_gpu_bak, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->wf->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->wi->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->wg->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->wo->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->uf->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->ui->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->ug->delta_gpu, 1);
        fill_gpu(l->outputs * l->batch * l->steps, 0, l->uo->delta_gpu, 1);
    }
    for(int i = 0; i < l->steps; ++i) {
        forward_connected_layer_gpu(l->wf, l->h_gpu, test);
        forward_connected_layer_gpu(l->wi, l->h_gpu, test);
        forward_connected_layer_gpu(l->wg, l->h_gpu, test);
        forward_connected_layer_gpu(l->wo, l->h_gpu, test);
        forward_connected_layer_gpu(l->uf, input, test);
        forward_connected_layer_gpu(l->ui, input, test);
        forward_connected_layer_gpu(l->ug, input, test);
        forward_connected_layer_gpu(l->uo, input, test);

        copy_gpu(l->outputs*l->batch, l->wf->output_gpu, 1, l->f_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->uf->output_gpu, 1, l->f_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->wi->output_gpu, 1, l->i_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->ui->output_gpu, 1, l->i_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->wg->output_gpu, 1, l->g_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->ug->output_gpu, 1, l->g_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->wo->output_gpu, 1, l->o_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->uo->output_gpu, 1, l->o_gpu, 1);

        activate_array_gpu(l->f_gpu, l->outputs*l->batch, LOGISTIC);
        activate_array_gpu(l->i_gpu, l->outputs*l->batch, LOGISTIC);
        activate_array_gpu(l->g_gpu, l->outputs*l->batch, TANH);
        activate_array_gpu(l->o_gpu, l->outputs*l->batch, LOGISTIC);

        copy_gpu(l->outputs*l->batch, l->i_gpu, 1, l->temp_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->g_gpu, 1, l->temp_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->f_gpu, 1, l->c_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->temp_gpu, 1, l->c_gpu, 1);

        copy_gpu(l->outputs*l->batch, l->c_gpu, 1, l->h_gpu, 1);
        activate_array_gpu(l->h_gpu, l->outputs*l->batch, TANH);
        mul_gpu(l->outputs*l->batch, l->o_gpu, 1, l->h_gpu, 1);

        copy_gpu(l->outputs*l->batch, l->c_gpu, 1, l->cell_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->h_gpu, 1, l->output_gpu, 1);

        input += l->inputs*l->batch;
        l->output_gpu += l->outputs*l->batch;
        l->cell_gpu += l->outputs*l->batch;
        increment_layer(l->wf, 1);
        increment_layer(l->wi, 1);
        increment_layer(l->wg, 1);
        increment_layer(l->wo, 1);
        increment_layer(l->uf, 1);
        increment_layer(l->ui, 1);
        increment_layer(l->ug, 1);
        increment_layer(l->uo, 1);
    }
    l->output_gpu -= l->outputs*l->batch*l->steps;
    l->cell_gpu -= l->outputs*l->batch*l->steps;

    increment_layer(l->wf, -l->steps);
    increment_layer(l->wi, -l->steps);
    increment_layer(l->wg, -l->steps);
    increment_layer(l->wo, -l->steps);
    increment_layer(l->uf, -l->steps);
    increment_layer(l->ui, -l->steps);
    increment_layer(l->ug, -l->steps);
    increment_layer(l->uo, -l->steps);
}

void backward_lstm_layer_gpu(lstm_layer *l, float *input, float *delta, int test)
{
    increment_layer(l->wf, l->steps);
    increment_layer(l->wi, l->steps);
    increment_layer(l->wg, l->steps);
    increment_layer(l->wo, l->steps);
    increment_layer(l->uf, l->steps);
    increment_layer(l->ui, l->steps);
    increment_layer(l->ug, l->steps);
    increment_layer(l->uo, l->steps);
    input += l->inputs*l->batch*l->steps;
    if(delta) delta += l->inputs*l->batch*l->steps;
    l->output_gpu += l->outputs*l->batch*l->steps;
    l->delta_gpu += l->outputs*l->batch*l->steps;
    l->cell_gpu += l->outputs*l->batch*l->steps;
    float *last_c_gpu = l->cell_gpu - l->outputs*l->batch;
    float *last_h_gpu = l->output_gpu - l->outputs*l->batch;

    for(int i = l->steps - 1; i >= 0; --i) {
        increment_layer(l->wf, -1);
        increment_layer(l->wi, -1);
        increment_layer(l->wg, -1);
        increment_layer(l->wo, -1);
        increment_layer(l->uf, -1);
        increment_layer(l->ui, -1);
        increment_layer(l->ug, -1);
        increment_layer(l->uo, -1);
        input -= l->inputs*l->batch;
        if (delta) delta -= l->inputs*l->batch;
        l->output_gpu -= l->outputs*l->batch;
        l->cell_gpu -= l->outputs*l->batch;
        l->delta_gpu -= l->outputs*l->batch;

        copy_gpu(l->outputs*l->batch, l->wf->output_gpu, 1, l->f_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->uf->output_gpu, 1, l->f_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->wi->output_gpu, 1, l->i_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->ui->output_gpu, 1, l->i_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->wg->output_gpu, 1, l->g_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->ug->output_gpu, 1, l->g_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->wo->output_gpu, 1, l->o_gpu, 1);
        axpy_gpu(l->outputs*l->batch, 1, l->uo->output_gpu, 1, l->o_gpu, 1);
        activate_array_gpu(l->f_gpu, l->outputs*l->batch, LOGISTIC);
        activate_array_gpu(l->i_gpu, l->outputs*l->batch, LOGISTIC);
        activate_array_gpu(l->g_gpu, l->outputs*l->batch, TANH);
        activate_array_gpu(l->o_gpu, l->outputs*l->batch, LOGISTIC);

        if (i != 0) copy_gpu(l->outputs*l->batch, l->cell_gpu - l->outputs*l->batch, 1, l->prev_cell_gpu, 1);
        else copy_gpu(l->outputs*l->batch, l->c_gpu_bak, 1, l->prev_cell_gpu, 1);;
        if (i != 0) copy_gpu(l->outputs*l->batch, l->output_gpu - l->outputs*l->batch, 1, l->prev_output_gpu, 1);
        else copy_gpu(l->outputs*l->batch, l->h_gpu_bak, 1, l->prev_output_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->cell_gpu, 1, l->c_gpu, 1);
        copy_gpu(l->outputs*l->batch, l->output_gpu, 1, l->h_gpu, 1);
        float *dh_gpu = (i == 0) ? 0 : l->delta_gpu - l->outputs*l->batch;

        copy_gpu(l->outputs*l->batch, l->c_gpu, 1, l->temp_gpu, 1);
        activate_array_gpu(l->temp_gpu, l->outputs*l->batch, TANH);
        copy_gpu(l->outputs*l->batch, l->delta_gpu, 1, l->temp3_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->temp3_gpu, 1, l->temp_gpu, 1);
        gradient_array_gpu(l->o_gpu, l->outputs*l->batch, LOGISTIC, l->temp_gpu);
        copy_gpu(l->outputs*l->batch, l->temp_gpu, 1, l->wo->delta_gpu, 1);
        backward_connected_layer_gpu(l->wo, l->prev_output_gpu, dh_gpu, test);
        copy_gpu(l->outputs*l->batch, l->temp_gpu, 1, l->uo->delta_gpu, 1);
        backward_connected_layer_gpu(l->uo, input, delta, test);

        copy_gpu(l->outputs*l->batch, l->c_gpu, 1, l->temp_gpu, 1);
        activate_array_gpu(l->temp_gpu, l->outputs*l->batch, TANH);
        copy_gpu(l->outputs*l->batch, l->delta_gpu, 1, l->temp2_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->o_gpu, 1, l->temp2_gpu, 1);
        gradient_array_gpu(l->temp_gpu, l->outputs*l->batch, TANH, l->temp2_gpu);

        copy_gpu(l->outputs*l->batch, l->temp2_gpu, 1, l->temp_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->i_gpu, 1, l->temp_gpu, 1);
        gradient_array_gpu(l->g_gpu, l->outputs*l->batch, TANH, l->temp_gpu);
        copy_gpu(l->outputs*l->batch, l->temp_gpu, 1, l->wg->delta_gpu, 1);
        backward_connected_layer_gpu(l->wg, l->prev_output_gpu, dh_gpu, test);
        copy_gpu(l->outputs*l->batch, l->temp_gpu, 1, l->ug->delta_gpu, 1);
        backward_connected_layer_gpu(l->ug, input, delta, test);

        copy_gpu(l->outputs*l->batch, l->temp2_gpu, 1, l->temp_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->g_gpu, 1, l->temp_gpu, 1);
        gradient_array_gpu(l->i_gpu, l->outputs*l->batch, LOGISTIC, l->temp_gpu);
        copy_gpu(l->outputs*l->batch, l->temp_gpu, 1, l->wi->delta_gpu, 1);
        backward_connected_layer_gpu(l->wi, l->prev_output_gpu, dh_gpu, test);
        copy_gpu(l->outputs*l->batch, l->temp_gpu, 1, l->ui->delta_gpu, 1);
        backward_connected_layer_gpu(l->ui, input, delta, test);

        copy_gpu(l->outputs*l->batch, l->temp2_gpu, 1, l->temp_gpu, 1);
        mul_gpu(l->outputs*l->batch, l->prev_cell_gpu, 1, l->temp_gpu, 1);
        gradient_array_gpu(l->f_gpu, l->outputs*l->batch, LOGISTIC, l->temp_gpu);
        copy_gpu(l->outputs*l->batch, l->temp_gpu, 1, l->wf->delta_gpu, 1);
        backward_connected_layer_gpu(l->wf, l->prev_output_gpu, dh_gpu, test);
        copy_gpu(l->outputs*l->batch, l->temp_gpu, 1, l->uf->delta_gpu, 1);
        backward_connected_layer_gpu(l->uf, input, delta, test);
    }
    copy_gpu(l->outputs * l->batch, last_c_gpu, 1, l->c_gpu, 1);
    copy_gpu(l->outputs * l->batch, last_h_gpu, 1, l->h_gpu, 1);
}
#endif
