#ifndef DATA_H
#define DATA_H

#include "image.h"

typedef struct{
    int n;  // number of image
    int h, w, c;
    float *data;
    int *truth_label_index;
} batch;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

typedef struct{
    int w, h;
    matrix X;
    matrix y;
} batch_detect;

batch random_batch(char **paths, int batch_size, char **labels, int classes, int train_set_size, int w, int h, int c,
                   float hue, float saturation, float exposure, int flip, float mean_value, float scale, int test);
void free_batch(batch *b);
char **get_labels(char *filename);
char **get_labels_and_num(char *filename, int *num);
struct list *get_paths(char *filename);
batch *load_csv_image_to_memory(char *filename, int batch_size, char **labels, int classes, int train_set_size,
                                int *batch_num_return, int w, int h, int c, float hue, float saturation, float exposure, int test);
batch *load_image_to_memory(char **paths, int batch_size, char **labels, int classes, int train_set_size,
                            int *batch_num_return, int w, int h, int c, float hue, float saturation, float exposure,
                            int flip, float mean_value, float scale, int test);

void free_matrix(matrix m);
matrix make_matrix(int rows, int cols);
batch_detect load_data_detection(int n, char **paths, int train_set_size, int w, int h, int boxes, int classes,
                                 float jitter, float hue, float saturation, float exposure, int test);
image load_data_detection_valid(char *path, int w, int h, int *image_w, int *image_h);
void free_batch_detect(batch_detect d);
#endif
