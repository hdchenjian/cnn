#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "data.h"
#include "utils.h"

#include "convolutional_layer.h"
#include "connected_layer.h"
#include "maxpool_layer.h"
#include "dropout_layer.h"
#include "normalize_layer.h"
#include "avgpool_layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define SECRET_NUM -1234

#ifdef GPU
    #include "cuda.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

enum COST_TYPE{
    SSE, MASKED, L1, SEG, SMOOTH
};

enum learning_rate_policy{
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
};

enum LAYER_TYPE{
    CONVOLUTIONAL,
    CONNECTED,
    ROUTE,
    SHORTCUT,
    MAXPOOL,
    AVGPOOL,
    NORMALIZE,
    DROPOUT,
    SOFTMAX,
    COST,
};

typedef struct {
    int w, h, c, out_w, out_h, out_c, index, batch, outputs;
    float prev_layer_weight, shortcut_layer_weight;
    ACTIVATION activation;
    float *delta, *output;
    float *output_gpu, *delta_gpu;
} shortcut_layer;

typedef struct {
    int batch, n, inputs, outputs, out_w, out_h, out_c;
    int *input_layers, *input_sizes;
    float *delta, *output;
    float *output_gpu, *delta_gpu;
} route_layer;


typedef struct {
    int is_last_layer;   // 1: is last layer, 0: not
    int inputs, batch;
    float label_specific_margin_bias;
    int margin_scale;
    float *delta, *output;
    float *delta_gpu, *output_gpu;
    float *input_backup, *input_backup_gpu;  // for AM-softmax
    float *loss, *loss_gpu;
    float *cost;
} softmax_layer;

typedef struct {
    int batch,inputs, outputs;
    float scale;
    float *delta, *output;
    float *output_gpu, *delta_gpu;
    enum COST_TYPE cost_type;
    float *cost;
} cost_layer;

typedef struct {
    int gpu_index;
    int n;                  // the size of network
    int max_batches, max_epoch; // max iteration times of batch
    size_t seen;    // the number of image processed
    int batch;   // the number of batch processed
    int epoch;
    int batch_train;   // the number of batch trained
    int w, h, c;  // net input data dimension
    int test;    // 0: train, 1: valid, 2: test
    int classes;    // train data classes
    float *truth;  // train data label
    int *truth_label_index;

    int correct_num;  // train correct number
    int correct_num_count;  // all trained data size, train accuracy = correct_num / correct_num_count
    float *workspace;  // for convolutional_layer image reorder
    float *workspace_gpu;  // for convolutional_layer image reorder
    size_t workspace_size;
    float loss;
    float hue, saturation, exposure;  // random_distort_image

    enum learning_rate_policy policy;
    int learning_rate_poly_power;  // for POLY learning_rate_policy
    float learning_rate_init;
    float learning_rate;
    float momentum;
    float decay;
    float *scales; // for STEP STEPS learning_rate_policy
    int   *steps;
    int num_steps;

    void **layers;
    enum LAYER_TYPE *layers_type;
#ifdef GPU
    float *input_gpu;  // train data
    float *truth_gpu;  // train data truth
    int *truth_label_index_gpu;
    int *is_not_max_gpu; // for counting correct rate in forward_softmax_layer_gpu
#endif
}network;

image get_avgpool_image(const avgpool_layer *l);
void forward_cost_layer(const cost_layer *l, float *input, network *net);
void backward_cost_layer(const cost_layer *l, float *delta);
void forward_softmax_layer(softmax_layer *layer, float *input, network *net);
void backward_softmax_layer(const softmax_layer *layer, float *delta);
void forward_route_layer(const route_layer *l, network *net);
void backward_route_layer(const route_layer *l, network *net);
image get_route_image(const route_layer *layer);
void forward_shortcut_layer(const shortcut_layer *l, float *input, network *net);
void backward_shortcut_layer(const shortcut_layer *l, float *delta, network *net);
image get_shortcut_image(const shortcut_layer *layer);

#ifdef GPU
void forward_cost_layer_gpu(const cost_layer *l, float *input, network *net);
void backward_cost_layer_gpu(const cost_layer *l, float *delta);
void forward_softmax_layer_gpu(softmax_layer *layer, float *input_gpu, network *net);
void backward_softmax_layer_gpu(const softmax_layer *layer, float *delta_gpu);
void forward_route_layer_gpu(const route_layer *l, network *net);
void backward_route_layer_gpu(const route_layer *l, network *net);
void forward_shortcut_layer_gpu(const shortcut_layer *l, float *input_gpu, network *net);
void backward_shortcut_layer_gpu(const shortcut_layer *l, float *delta_gpu, network *net);
#endif

float *get_network_layer_data(network *net, int i, int data_type, int is_gpu);

network *make_network(int n);
void free_network(network *net);
void train_network_batch(network *net, batch b);
void valid_network(network *net, batch b);
int get_network_output_size_layer(network *net, int i);
image get_network_image_layer(network *net, int i);
float update_current_learning_rate(network * net);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
#endif

