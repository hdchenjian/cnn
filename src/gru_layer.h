#ifndef GRU_LAYER_H
#define GRU_LAYER_H

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
    float *output, *delta, *state, *prev_state, *forgot_state, *forgot_delta, *prev_delta, *r_cpu, *z_cpu, *h_cpu;

    float *output_gpu, *delta_gpu, *state_gpu, *prev_state_gpu, *forgot_state_gpu, *forgot_delta_gpu, *prev_delta_gpu, *r_gpu, *z_gpu, *h_gpu;
    connected_layer *wr, *wz, *wh, *ur, *uz, *uh;
} gru_layer;

void increment_layer(connected_layer *l, int steps);

image get_gru_image(const gru_layer *layer);
gru_layer *make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);
void free_gru_layer(void *input);
void forward_gru_layer(gru_layer *l, float *input, int test);
void backward_gru_layer(gru_layer *l, float *input, float *delta, int test);
void update_gru_layer(const gru_layer *l, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_gru_layer_gpu(gru_layer *l, float *input, int test);
void backward_gru_layer_gpu(gru_layer *l, float *input, float *delta, int test);
void update_gru_layer_gpu(const gru_layer *l, float learning_rate, float momentum, float decay); 
#endif

#endif

