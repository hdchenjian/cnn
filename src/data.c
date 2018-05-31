#include "data.h"
#include "list.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

batch make_batch(int batch_size, int classes)
{
    batch b;
    b.n = batch_size;
    b.images = calloc(batch_size, sizeof(image *));
    b.truth = calloc(batch_size, sizeof(float *));
    for(int i =0 ; i < batch_size; ++i) b.truth[i] = calloc(classes, sizeof(float));
    return b;
}

struct list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    struct list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

void free_batch(batch b)
{
    int i;
    for(i = 0; i < b.n; ++i){
        free_image(*b.images[i]);
        free(b.truth[i]);
    }
    free(b.images);
    free(b.truth);
}


void fill_truth(char *path, char **labels, int classes, float *truth)
{
    for(int i = 0; i < classes; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
        }
    }
}

void load_csv_images(char *filename, char **labels, int classes, int train_set_size, image **image_all, float **truth_all)
{
    FILE *fp = fopen(filename, "r");
    if(!fp) file_error(filename);
    int fields = 0;  // the number of pixel per image
    int w, h;
    int n = 0;
    char *line;
    while((line = fgetl(fp)) && (n < train_set_size)){
        char class = line[0];
        if(0 == fields){
            fields = count_fields(line);
            w = sqrt(fields);
            h = sqrt(fields);
        }
        float *value = parse_fields(line, fields);
        image_all[n] = calloc(1, sizeof(image));
        image_all[n]->h = h;
        image_all[n]->w = w;
        image_all[n]->c = 1;
        image_all[n]->data = value + 1;
        normalize_array(image_all[n]->data, image_all[n]->h*image_all[n]->w*image_all[n]->c);
        truth_all[n] = calloc(classes, sizeof(float));
        char name[16] = {0};
        sprintf(name, "%c.png", class);
        fill_truth(name, labels, classes, truth_all[n]);
        free(line);
        n += 1;
    }
    fclose(fp);
}

batch *load_csv_image_to_memory(char *filename, int batch_size, char **labels, int classes, int train_set_size)
{
    image **image_all = calloc(train_set_size, sizeof(image *));
    float **truth_all = calloc(train_set_size, sizeof(float *));
    load_csv_images(filename, labels, classes, train_set_size, image_all, truth_all);

    /* random the train image index */
    int train_set_size_real = 0;
    if(train_set_size % batch_size == 0) {
        train_set_size_real = train_set_size;
    } else {
        train_set_size_real = (train_set_size / batch_size + 1) * batch_size;
    }
    int *index = calloc(train_set_size_real, sizeof(int));
    for(int i = 0; i < train_set_size_real; i++) {
        if(i < train_set_size) {
            index[i] = i;
        } else {
            index[i] = rand() % train_set_size;
        }
    }
    for(int i = 0; i < train_set_size_real; i++) {
        int index_random = rand() % train_set_size_real;
        int temp = index[i];
        index[i] = index[index_random];
        index[index_random] = temp;
    }

    int batch_num = train_set_size_real / batch_size;
    batch *train_data = calloc(batch_num, sizeof(batch));
    for(int i = 0; i < batch_num; i++){
        train_data[i].n = batch_size;
        train_data[i].images = calloc(batch_num, sizeof(batch *));
        train_data[i].truth = calloc(batch_num, sizeof(float *));
        for(int j = 0; j < batch_num; j++){
        	train_data[i].images[j] = image_all[i * batch_size + j];
        	train_data[i].truth[j] = truth_all[i * batch_size + j];
        }
    } 
    free(image_all);
    free(truth_all);
    return train_data;
}

batch random_batch(char **paths, int batch_size, char **labels, int classes, int train_set_size)
{
    batch b = make_batch(batch_size, classes);
    for(int i = 0; i < batch_size; ++i){
        int index = rand() % train_set_size;
        *(b.images[i]) = load_image_me(paths[index]);
        normalize_array(b.images[i]->data, b.images[i]->h*b.images[i]->w*b.images[i]->c);

        fill_truth(paths[index], labels, classes, b.truth[i]);
        //printf("%s %f\n", paths[index], *b.truth[i]);
    }
    return b;
}

char **get_labels(char *filename)
{
    struct list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}
