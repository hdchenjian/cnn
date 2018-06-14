#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "data.h"
#include "utils.h"

#include "convolutional_layer.h"
#include "connected_layer.h"
#include "maxpool_layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define GPU
#define SECRET_NUM -1234
extern int gpu_index;

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

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
    MAXPOOL,
    AVGPOOL,
    DROPOUT,
    SOFTMAX,
    COST,
};

typedef struct {
    int h,w,c,batch;
    float *delta;
    float *output;
    enum LAYER_TYPE type;
} avgpool_layer;

typedef struct {
    int batch, inputs, outputs, out_h, out_w, c;
    float probability, scale;
    float *rand;
    float *output, *delta;
} dropout_layer;

typedef struct {
	int is_last_layer;   // 1: is last layer, 0: not
    int inputs, batch;
    float *delta;
    float *output;
    float * loss;
    float *cost;
} softmax_layer;

typedef struct {
    int batch,inputs, outputs;
    float scale;
    float *delta;
    float *output;
    enum COST_TYPE cost_type;
    float *cost;
} cost_layer;

typedef struct {
    int n;                  // the size of network
    int max_batches; // max iteration times of batch
    size_t seen;    // the number of image processed
    int batch;   // the number of batch processed
    int epoch;
    int batch_train;   // the number of batch trained
    int w, h, c;  // net input data dimension
    int test;    // 0: train, 1: valid, 2: test
    int classes;    // train data classes
    float *truth;  // train data label

    int correct_num;  // train correct number
    int correct_num_count;  // all trained data size, train accuracy = correct_num / correct_num_count
    float *workspace;  // for convolutional_layer image reorder
    size_t workspace_size;
    float loss;
    float hue, saturation, exposure;  // random_distort_image

    enum learning_rate_policy policy;
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
#endif

}network;

void forward_avgpool_layer(const avgpool_layer *l, float *in);
void backward_avgpool_layer(const avgpool_layer *l, float *delta);
void forward_cost_layer(const cost_layer *l, float *input, network *net);
void backward_cost_layer(const cost_layer *l, float *delta);
void forward_softmax_layer(const softmax_layer *layer, float *input, network *net);
void backward_softmax_layer(const softmax_layer *layer, float *delta);
void forward_dropout_layer(dropout_layer *l, float *input, network *net);
void backward_dropout_layer(dropout_layer *l, float *delta);
image get_dropout_image(const dropout_layer *layer, int batch);

float *get_network_layer_data(network *net, int i, int data_type);

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

