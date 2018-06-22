#ifndef DATA_H
#define DATA_H

#include "image.h"

typedef struct{
    int n;  // number of image
    int h;
    int w;
    int c;
    float *data;
    float *truth;
} batch;

batch random_batch(char **paths, int batch_size, char **labels, int classes, int train_set_size,
        int w, int h, int c, float hue, float saturation, float exposure);
void free_batch(batch b);
char **get_labels(char *filename);
struct list *get_paths(char *filename);
batch *load_csv_image_to_memory(char *filename, int batch_size, char **labels, int classes, int train_set_size,
        int *batch_num_return, int w, int h, int c, float hue, float saturation, float exposure);
batch *load_image_to_memory(char **paths, int batch_size, char **labels, int classes, int train_set_size,
        int *batch_num_return, int w, int h, int c, float hue, float saturation, float exposure);
#endif