#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

typedef struct{
    int inputs, outputs, batch, steps;
    float *output, *delta, *cell_cpu, *prev_cell_cpu, *prev_output;
    float *f_cpu, *i_cpu, *g_cpu, *o_cpu, *c_cpu, *c_cpu_bak, *h_cpu, *h_cpu_bak;
    float *temp_cpu, *temp2_cpu, *temp3_cpu;

    float *output_gpu, *delta_gpu, *cell_gpu, *prev_cell_gpu, *prev_output_gpu;
    float *f_gpu, *i_gpu, *g_gpu, *o_gpu, *c_gpu, *c_gpu_bak, *h_gpu, *h_gpu_bak;
    float *temp_gpu, *temp2_gpu, *temp3_gpu;
    connected_layer *wf, *wi, *wg, *wo, *uf, *ui, *ug, *uo;
} lstm_layer;

void increment_layer(connected_layer *l, int steps);

image get_lstm_image(const lstm_layer *layer);
lstm_layer *make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);
void free_lstm_layer(void *input);
void forward_lstm_layer(lstm_layer *l, float *input, int test);
void backward_lstm_layer(lstm_layer *l, float *input, float *delta, int test);
void update_lstm_layer(const lstm_layer *l, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_lstm_layer_gpu(lstm_layer *l, float *input, int test);
void backward_lstm_layer_gpu(lstm_layer *l, float *input, float *delta, int test);
void update_lstm_layer_gpu(lstm_layer *l, float learning_rate, float momentum, float decay); 
#endif
#endif
