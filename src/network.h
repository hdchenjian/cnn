#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "data.h"
#include "utils.h"

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "lstm_layer.h"
#include "gru_layer.h"
#include "maxpool_layer.h"
#include "upsample_layer.h"
#include "dropout_layer.h"
#include "normalize_layer.h"
#include "avgpool_layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include <pthread.h>
#include <time.h>

#ifdef GPU
#include "cuda.h"
#elif defined(OPENCL)
#include "opencl.h"
#endif

enum COST_TYPE{
    SSE, MASKED, L1, SEG, SMOOTH
};

enum learning_rate_policy{
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
};

enum LAYER_TYPE{
    CONVOLUTIONAL,
    BATCHNORM,
    CONNECTED,
    RNN,
    LSTM,
    GRU,
    ROUTE,
    SHORTCUT,
    MAXPOOL,
    AVGPOOL,
    NORMALIZE,
    DROPOUT,
    SOFTMAX,
    COST,
    UPSAMPLE,
    YOLO,
};

typedef struct {
    int w, h, c, out_w, out_h, out_c, index, batch, outputs, test;
    float prev_layer_weight, shortcut_layer_weight;
    ACTIVATION activation;
    float *delta, *output;
    float *output_gpu, *delta_gpu;
#ifdef OPENCL
    cl_mem output_cl, delta_cl;
#endif
} shortcut_layer;

typedef struct {
    int batch, n, inputs, outputs, out_w, out_h, out_c, test;
    int *input_layers, *input_sizes;
    float *delta, *output;
    float *output_gpu, *delta_gpu;
#ifdef OPENCL
    cl_mem output_cl, delta_cl;
#endif
} route_layer;


typedef struct {
    int is_last_layer;   // 1: is last layer, 0: not
    int inputs, outputs, batch;
    float label_specific_margin_bias;
    int margin_scale;
    float *delta, *output, *delta_gpu, *output_gpu;
    float *input_backup, *input_backup_gpu;  // for AM-softmax
    float *loss, *loss_gpu, *cost;
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
    int output_layer; // when output_layer == 1, get output from 1th layer
    int gpu_index;
    int n;                  // the size of network
    int max_batches, max_epoch; // max iteration times of batch
    size_t seen;    // the number of image processed
    int time_steps, inputs;  // for rnn layer, the inputs num of network
    int epoch;
    int batch_train;   // the number of batch trained
    int w, h, c, batch, subdivisions;  // net input data dimension
    int test;    // 0: train, 1: valid
    int classes;    // train data classes
    int *truth_label_index, *truth_label_index_gpu;
    float *input, *truth, *input_gpu, *truth_gpu;
    int *is_not_max_gpu; // for counting correct rate in forward_softmax_layer_gpu
    float *workspace, *workspace_gpu;  // for convolutional_layer image reorder
#ifdef OPENCL
    cl_mem workspace_cl, input_cl, truth_cl, is_not_max_cl, truth_label_index_cl;
#endif
    size_t workspace_size;
    int max_boxes;  // a image contain max_boxes groud truth box

    int correct_num;  // train correct number
    int accuracy_count, accuracy_count_max;  // all trained data size, train accuracy = correct_num / accuracy_count
    float loss;
    float hue, saturation, exposure, jitter;  // random_distort_image
    float mean_value, scale;   // use when load image
    int flip;

    enum learning_rate_policy policy;
    int learning_rate_poly_power;  // for POLY learning_rate_policy
    float learning_rate_init, learning_rate, momentum, decay;
    float *scales; // for STEP STEPS learning_rate_policy
    int *steps;
    int num_steps;

    void **layers;
    enum LAYER_TYPE *layers_type;
} network;

typedef struct {
    int h,w,c, out_h, out_w, out_c, n, batch, total, classes, inputs, outputs, truths, max_boxes, layer_index;
    int *mask;
    float ignore_thresh, truth_thresh;
    float *biases, *bias_updates, *delta, *output, *input_cpu;
    float *output_gpu, *delta_gpu;
#ifdef OPENCL
    cl_mem output_cl, delta_cl;
#endif
} yolo_layer;

image get_yolo_image(const yolo_layer *layer);
yolo_layer *make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int layer_index);
void free_yolo_layer(void *input);
void forward_yolo_layer(const yolo_layer *l, network *net, float *input, int test);
void backward_yolo_layer(const yolo_layer *l, float *delta);
int yolo_num_detections(const yolo_layer *l, float thresh);
int get_yolo_detections(yolo_layer *l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
#ifdef GPU
void forward_yolo_layer_gpu(const yolo_layer *l, network *net, float *input, int test);
void backward_yolo_layer_gpu(const yolo_layer *l, float *delta);
#elif defined(OPENCL)
void forward_yolo_layer_cl(const yolo_layer *l, network *net, cl_mem input_cl, int test);
#endif

cost_layer *make_cost_layer(int batch, int inputs, enum COST_TYPE cost_type, float scale);
void forward_cost_layer(const cost_layer *l, float *input, network *net);
void backward_cost_layer(const cost_layer *l, float *delta);
#ifdef GPU
void forward_cost_layer_gpu(const cost_layer *l, float *input, network *net);
void backward_cost_layer_gpu(const cost_layer *l, float *delta);
#endif

softmax_layer *make_softmax_layer(int inputs, int batch, int is_last_layer, float label_specific_margin_bias, int margin_scale);
void forward_softmax_layer(softmax_layer *layer, float *input, network *net);
void backward_softmax_layer(const softmax_layer *layer, float *delta);
#ifdef GPU
void forward_softmax_layer_gpu(softmax_layer *layer, float *input_gpu, network *net);
void backward_softmax_layer_gpu(const softmax_layer *layer, float *delta_gpu);
#endif

image get_route_image(const route_layer *layer);
route_layer *make_route_layer(int batch, int n, int *input_layers, int *input_size, network *net, int test);
void forward_route_layer(const route_layer *l, network *net);
void backward_route_layer(const route_layer *l, network *net);
#ifdef GPU
void forward_route_layer_gpu(const route_layer *l, network *net);
void backward_route_layer_gpu(const route_layer *l, network *net);
#elif defined(OPENCL)
void forward_route_layer_cl(const route_layer *l, network *net);
#endif

image get_shortcut_image(const shortcut_layer *layer);
shortcut_layer *make_shortcut_layer(int batch, int index, int w, int h, int c, int out_w,int out_h,int out_c,
                                    ACTIVATION activation, float prev_layer_weight, float shortcut_layer_weight, int test);
void forward_shortcut_layer(const shortcut_layer *l, float *input, network *net);
void backward_shortcut_layer(const shortcut_layer *l, float *delta, network *net);
#ifdef GPU
void forward_shortcut_layer_gpu(const shortcut_layer *l, float *input_gpu, network *net);
void backward_shortcut_layer_gpu(const shortcut_layer *l, float *delta_gpu, network *net);
#elif defined(OPENCL)
void forward_shortcut_layer_cl(const shortcut_layer *l, cl_mem input_cl, network *net);
#endif

float *get_network_layer_data(network *net, int i, int data_type, int is_gpu);
void reset_rnn_state(network *net, int b);
network *make_network(int n);
network *load_network(const char *cfg, const char *weights, int test);
void free_network(network *net);
void free_network_weight_bias_cpu(network *net);
void train_network(network *net, float *input, int *truth_label_index);
void train_network_detect(network *net, batch_detect d);
void valid_network(network *net, float *input, int *truth_label_index);
void forward_network_test(network *net, float *input);
int get_network_output_size_layer(network *net, int i);
image get_network_image_layer(network *net, int i);
float update_current_learning_rate(network * net);
void save_weights(network *net, char *filename);
void load_weights(network *net, const char *filename);
detection *get_network_boxes(network *net, int w, int h, float thresh, int *map, int relative, int *num);

#ifdef OPENCL
cl_mem get_network_layer_data_cl(network *net, int i, int data_type);
void forward_network_cl(network *net, cl_mem input);
#endif
#endif

