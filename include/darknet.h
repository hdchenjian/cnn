#ifndef DARKNET_API
#define DARKNET_API
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

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

struct tree{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
};

enum ACTIVATION{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;

enum BINARY_ACTIVATION {
    MULT, ADD, SUB, DIV
};

enum COST_TYPE{
    SSE, MASKED, L1, SEG, SMOOTH
};

struct update_args{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
};

enum learning_rate_policy{
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
};

struct network{
    int n;           // the size of network
    int batch;
    int max_batches; // max iteration times of batch
    int *seen;    // the number of image processed
    float epoch;
    float *output;
    enum learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;

    float *scales; // for STEP STEPS learning_rate_policy
    int   *steps;
    int num_steps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;

    int gpu_index;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
#endif

};

struct convolutional_layer{
    enum ACTIVATION activation;
    enum COST_TYPE cost_type;
    void (*forward)   (struct network);
    void (*backward)  (struct network);
    void (*update)    (struct update_args);
    void (*forward_gpu)   (struct network);
    void (*backward_gpu)  (struct network);
    void (*update_gpu)    (struct update_args);
    int batch_normalize;
};

struct image{
    int w;
    int h;
    int c;
    float *data;
};

struct box{
    float x, y, w, h;
};

struct matrix_darknet{
    int rows, cols;
    float **vals;
};

struct data{
    int w, h;
    struct matrix_darknet X;
    struct matrix_darknet y;
    int shallow;
    int *num_boxes;
    struct box **boxes;
};

struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    struct data *d;
    struct image *im;
    struct image *resized;
};

struct network *load_network(char *cfg, char *weights, int clear);
struct load_args get_base_args(struct network *net);

void free_data(struct data d);

struct node{
    void *val;
    struct node *next;
    struct node *prev;
};

struct list{
    int size;
    struct node *front;
    struct node *back;
};

void **list_to_array(struct list *l);
void free_list(struct list *l);

pthread_t load_data(struct load_args args);
struct list *read_data_cfg(char *filename);
struct list *read_cfg(char *filename);
unsigned char *read_file(char *filename);

void forward_network(struct network *net);
void backward_network(struct network *net);
void update_network(struct network *net);
int get_current_batch(struct network *net);
float get_current_learning_rate(struct network *net);
void free_network(struct network *net);
void set_batch_network(struct network *net, int b);

void change_leaves(struct tree *t, char *leaf_list);

void save_image_png(struct image im, const char *name);
void get_next_batch(struct data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(struct image im);
void normalize_image(struct image p);

float train_network_sgd(struct network *net, struct data d, int n);
void rgbgr_image(struct image im);
struct data copy_data(struct data d);


float *network_accuracies(struct network *net, struct data d, int n);
float train_network_datum(struct network *net);
struct image make_random_image(int w, int h, int c);

char *option_find_str(struct list *l, char *key, char *def);
int option_find_int(struct list *l, char *key, int def);

struct network *parse_network_cfg(char *filename);
void save_weights(struct network *net, char *filename);
void load_weights(struct network *net, char *filename);
void save_weights_upto(struct network *net, char *filename, int cutoff);
void load_weights_upto(struct network *net, char *filename, int start, int cutoff);

struct matrix_darknet network_predict_data(struct network *net, struct data test);
struct image **load_alphabet();
struct image get_network_image(struct network *net);
float *network_predict(struct network *net, float *input);

float *network_predict_image(struct network *net, struct image im);
int num_boxes(struct network *net);

char **get_labels(char *filename);

struct matrix_darknet make_matrix(int rows, int cols);

struct image load_image_color(char *filename, int w, int h);
struct image crop_image(struct image im, int dx, int dy, int w, int h);
struct image center_crop_image(struct image im, int w, int h);
struct image resize_min(struct image im, int min);
struct image letterbox_image(struct image im, int w, int h);

void free_image(struct image m);
float train_network(struct network *net, struct data d);
pthread_t load_data_in_thread(struct load_args args);
void load_data_blocking(struct load_args args);
struct list *get_paths(char *filename);

void top_k(float *a, int n, int k, int *index);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);

void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);

#endif
