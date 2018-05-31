#ifndef DATA_H
#define DATA_H

#include "image.h"

typedef struct{
    int n;  // number of image
    image **images;
    float **truth;
} batch;

batch get_all_data(char *filename, char **labels, int k);
batch random_batch(char **paths, int batch_size, char **labels, int classes, int train_set_size);
batch get_batch(char *filename, int curr, int total, char **labels, int k);
void free_batch(batch b);
char **get_labels(char *filename);
struct list *get_paths(char *filename);
batch *load_csv_image_to_memory(char *filename, int batch_size, char **labels, int classes, int train_set_size);


#endif
