#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "data.h"
#include "utils.h"

#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "softmax_layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

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

#ifndef __cplusplus
    #ifdef OPENCV
    #include "opencv2/highgui/highgui_c.h"
    #include "opencv2/imgproc/imgproc_c.h"
    #include "opencv2/core/version.hpp"
    #if CV_MAJOR_VERSION == 3
    #include "opencv2/videoio/videoio_c.h"
    #include "opencv2/imgcodecs/imgcodecs_c.h"
    #endif
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
    MAXPOOL,
	AVGPOOL,
    SOFTMAX,
	COST,
};

typedef struct {
    int h,w,c;
    int batch;
    int inputs;
    int stride;
    int out_h, out_w, out_c;
    int outputs;
    float *delta;
    float *output;
    enum LAYER_TYPE type;
} avgpool_layer;

typedef struct {
	float scale;
    int batch;
    int inputs;
    int outputs;
    float *delta;
    float *output;
    enum COST_TYPE cost_type;
    float *cost;
} cost_layer;

struct network{
    int n;                  // the size of network
    int max_batches; // max iteration times of batch
    int seen;    // the number of image processed
    int batch;
    int w, h, c;
    int test;    // 0: train, 1: test
    int classes;    // train data classes
    float *truth;  // train data label

    enum learning_rate_policy policy;
    float learning_rate;
    float momentum;
    float decay;

    float *scales; // for STEP STEPS learning_rate_policy
    int   *steps;
    int num_steps;

    void **layers;
    enum LAYER_TYPE *layers_type;
    int inputs;
    int outputs;
    float *output;
};
void forward_cost_layer(const cost_layer *l, float *input, struct network *net);
void forward_avgpool_layer(const avgpool_layer *l, float *in);
void backward_avgpool_layer(const avgpool_layer *l, float *delta);
void backward_cost_layer(const cost_layer *l, float *delta);


struct network *make_network(int n);
void forward_network(struct network *net, float *input);
void learn_network(struct network *net, float *input);
void update_network(struct network *net, double step);
void train_network_batch(struct network *net, batch b);
int get_network_output_size_layer(struct network *net, int i);
int get_network_output_size(struct network *net);
image get_network_image_layer(struct network *net, int i);
float *get_network_output(struct network *net);

#endif

