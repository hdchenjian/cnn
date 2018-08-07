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
    l->uf = make_connected_layer(inputs, outputs, connected_layer_batch, steps, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize);
    l->uf->batch = batch;
    fprintf(stderr, "\t");
    l->ui = make_connected_layer(inputs, outputs, connected_layer_batch, steps, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize);
    l->ui->batch = batch;
    fprintf(stderr, "\t");
    l->ug = make_connected_layer(inputs, outputs, connected_layer_batch, steps, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize);
    l->ug->batch = batch;
    fprintf(stderr, "\t");
    l->uo = make_connected_layer(inputs, outputs, connected_layer_batch, steps, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize);
    l->uo->batch = batch;
    fprintf(stderr, "\t");
    l->wf = make_connected_layer(outputs, outputs, connected_layer_batch, steps, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize);
    l->wf->batch = batch;
    fprintf(stderr, "\t");
    l->wi = make_connected_layer(outputs, outputs, connected_layer_batch, steps, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize);
    l->wi->batch = batch;
    fprintf(stderr, "\t");
    l->wg = make_connected_layer(outputs, outputs, connected_layer_batch, steps, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize);
    l->wg->batch = batch;
    fprintf(stderr, "\t");
    l->wo = make_connected_layer(outputs, outputs, connected_layer_batch, steps, activation, weight_normalize,
                                 bias_term, lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                 sigma, batch_normalize);
    l->wo->batch = batch;

    l->outputs = outputs;
    l->output = calloc(outputs*batch*steps, sizeof(float));
    l->delta = calloc(outputs*batch*steps, sizeof(float));

    l->state = calloc(outputs*batch, sizeof(float));
    l->prev_state = calloc(batch*outputs, sizeof(float));
    l->cell_cpu = calloc(batch*outputs, sizeof(float));
    l->prev_cell_cpu = calloc(batch*outputs*steps, sizeof(float));

    l->f_cpu = calloc(batch*outputs, sizeof(float));
    l->i_cpu = calloc(batch*outputs, sizeof(float));
    l->g_cpu = calloc(batch*outputs, sizeof(float));
    l->o_cpu = calloc(batch*outputs, sizeof(float));
    l->c_cpu = calloc(batch*outputs, sizeof(float));
    l->h_cpu = calloc(batch*outputs, sizeof(float));
    l->temp_cpu = calloc(batch*outputs, sizeof(float));
    l->temp2_cpu = calloc(batch*outputs, sizeof(float));
    l->temp3_cpu = calloc(batch*outputs, sizeof(float));
    l->dc_cpu = calloc(batch*outputs, sizeof(float));
    l->dh_cpu = calloc(batch*outputs, sizeof(float));

#ifdef GPU
    l->output_gpu = cuda_make_array(0, batch*outputs*steps);
    l->delta_gpu = cuda_make_array(0, batch*outputs*steps);

    l->state_gpu = cuda_make_array(0, batch*outputs);
    l->prev_state_gpu = cuda_make_array(0, batch*outputs);
    l->cell_gpu = cuda_make_array(0, batch*outputs*steps);
    l->prev_cell_gpu = cuda_make_array(0, batch*outputs);

    l->f_gpu = cuda_make_array(0, batch*outputs);
    l->i_gpu = cuda_make_array(0, batch*outputs);
    l->g_gpu = cuda_make_array(0, batch*outputs);
    l->o_gpu = cuda_make_array(0, batch*outputs);
    l->c_gpu = cuda_make_array(0, batch*outputs);
    l->h_gpu = cuda_make_array(0, batch*outputs);
    l->temp_gpu =  cuda_make_array(0, batch*outputs);
    l->temp2_gpu = cuda_make_array(0, batch*outputs);
    l->temp3_gpu = cuda_make_array(0, batch*outputs);
    l->dc_gpu = cuda_make_array(0, batch*outputs);
    l->dh_gpu = cuda_make_array(0, batch*outputs);
#endif

    return l;
}

void free_lstm_layer(void *input)
{
    lstm_layer *layer = (lstm_layer *)input;
    if(layer->output) free_ptr(layer->output);
    if(layer->delta) free_ptr(layer->delta);
    if(layer->state) free_ptr(layer->state);
    if(layer->prev_state) free_ptr(layer->prev_state);
    if(layer->cell_cpu) free_ptr(layer->cell_cpu);
    if(layer->prev_cell_cpu) free_ptr(layer->prev_cell_cpu);
    if(layer->f_cpu) free_ptr(layer->f_cpu);
    if(layer->i_cpu) free_ptr(layer->i_cpu);
    if(layer->g_cpu) free_ptr(layer->g_cpu);
    if(layer->o_cpu) free_ptr(layer->o_cpu);
    if(layer->c_cpu) free_ptr(layer->c_cpu);
    if(layer->h_cpu) free_ptr(layer->h_cpu);
    if(layer->temp_cpu) free_ptr(layer->temp_cpu);
    if(layer->temp2_cpu) free_ptr(layer->temp2_cpu);
    if(layer->temp3_cpu) free_ptr(layer->temp3_cpu);
    if(layer->dc_cpu) free_ptr(layer->dc_cpu);
    if(layer->dh_cpu) free_ptr(layer->dh_cpu);

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
#endif
    free_ptr(layer);
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
        copy_cpu(l->outputs*l->batch, l->cell_cpu, 1, l->c_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->output, 1, l->h_cpu, 1);

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->temp_cpu, 1);
        activate_array(l->temp_cpu, l->outputs*l->batch, TANH);
        copy_cpu(l->outputs*l->batch, l->delta, 1, l->temp3_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->temp3_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->o_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->wo->delta, 1);
        l->dh_cpu = (i == 0) ? 0 : l->delta - l->outputs*l->batch;
        if (i != 0) copy_cpu(l->outputs*l->batch, l->output - l->outputs*l->batch, 1, l->prev_state, 1);
        backward_connected_layer(l->wo, l->prev_state, l->dh_cpu, test);

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->uo->delta, 1);
        backward_connected_layer(l->uo, input, delta, test);

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->temp_cpu, 1);
        activate_array(l->temp_cpu, l->outputs*l->batch, TANH);
        copy_cpu(l->outputs*l->batch, l->delta, 1, l->temp2_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->o_cpu, 1, l->temp2_cpu, 1);
        gradient_array(l->temp_cpu, l->outputs*l->batch, TANH, l->temp2_cpu);
        axpy_cpu(l->outputs*l->batch, 1, l->dc_cpu, 1, l->temp2_cpu, 1);

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->i_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->g_cpu, l->outputs*l->batch, TANH, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->wg->delta, 1);
        backward_connected_layer(l->wg, l->prev_state, l->dh_cpu, test);

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->ug->delta, 1);
        backward_connected_layer(l->ug, input, delta, test);

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->g_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->i_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->wi->delta, 1);
        backward_connected_layer(l->wi, l->prev_state, l->dh_cpu, test);

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->ui->delta, 1);
        backward_connected_layer(l->ui, input, delta, test);

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->prev_cell_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->f_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->wf->delta, 1);
        backward_connected_layer(l->wf, l->prev_state, l->dh_cpu, test);

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->uf->delta, 1);
        backward_connected_layer(l->uf, input, delta, test);

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);
        mul_cpu(l->outputs*l->batch, l->f_cpu, 1, l->temp_cpu, 1);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->dc_cpu, 1);
    }
}

#ifdef GPU
void update_lstm_layer_gpu(lstm_layer *l, update_args a)
{
    update_connected_layer_gpu(*(l.wf), a);
    update_connected_layer_gpu(*(l.wi), a);
    update_connected_layer_gpu(*(l.wg), a);
    update_connected_layer_gpu(*(l.wo), a);
    update_connected_layer_gpu(*(l.uf), a);
    update_connected_layer_gpu(*(l.ui), a);
    update_connected_layer_gpu(*(l.ug), a);
    update_connected_layer_gpu(*(l.uo), a);
}

void forward_lstm_layer_gpu(lstm_layer *l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    fill_gpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);

    fill_gpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
    if (state.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_gpu = l.h_gpu;
        forward_connected_layer_gpu(wf, s);
        forward_connected_layer_gpu(wi, s);
        forward_connected_layer_gpu(wg, s);
        forward_connected_layer_gpu(wo, s);

        s.input_gpu = state.input_gpu;
        forward_connected_layer_gpu(uf, s);
        forward_connected_layer_gpu(ui, s);
        forward_connected_layer_gpu(ug, s);
        forward_connected_layer_gpu(uo, s);

        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);

        copy_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.h_gpu, 1);
        activate_array_gpu(l.h_gpu, l.outputs*l.batch, TANH);
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.cell_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.h_gpu, 1, l.output_gpu, 1);

        state.input_gpu += l.inputs*l.batch;
        l.output_gpu    += l.outputs*l.batch;
        l.cell_gpu      += l.outputs*l.batch;

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}

void backward_lstm_layer_gpu(lstm_layer *l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input_gpu += l.inputs*l.batch*(l.steps - 1);
    if (state.delta_gpu) state.delta_gpu += l.inputs*l.batch*(l.steps - 1);

    l.output_gpu += l.outputs*l.batch*(l.steps - 1);
    l.cell_gpu += l.outputs*l.batch*(l.steps - 1);
    l.delta_gpu += l.outputs*l.batch*(l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) copy_gpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, 1, l.prev_cell_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.cell_gpu, 1, l.c_gpu, 1);
        if (i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);

        l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, TANH);
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);

        copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);
        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);

        copy_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);

        gradient_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH, l.temp2_gpu);
        axpy_gpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);
        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, TANH);
        mul_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.o_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wo, s);

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(uo, s);

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.g_gpu, l.outputs*l.batch, TANH, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wg, s);

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(ug, s);

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.i_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wi, s);

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(ui, s);

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.f_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wf, s);

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(uf, s);

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, l.dc_gpu, 1);

        state.input_gpu -= l.inputs*l.batch;
        if (state.delta_gpu) state.delta_gpu -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.cell_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}
#endif
