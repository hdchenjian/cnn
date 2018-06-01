#include "data.h"
#include "list.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    free(b.data);
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

void load_csv_images(char *filename, char **labels, int classes, int train_set_size,
		int image_size, float *image_all, float *truth_all)
{
    FILE *fp = fopen(filename, "r");
    if(!fp) file_error(filename);
    int fields = 0;  // the number of pixel per image
    int n = 0;
    char *line;
    while((line = fgetl(fp)) && (n < train_set_size)){
        char class = line[0];
        if(0 == fields){
            fields = count_fields(line);
            //w = sqrt(fields);
            //h = sqrt(fields);
        }
        float *value = parse_fields(line, fields);
        memcpy(image_all + n * image_size, value + 1, image_size * sizeof(float));
        normalize_array(image_all + n * image_size, image_size);
        char name[16] = {0};
        sprintf(name, "%c.png", class);
        fill_truth(name, labels, classes, truth_all + n * classes);
        free(line);
        free(value);
        n += 1;
    }
    fclose(fp);
}

batch *load_csv_image_to_memory(char *filename, int batch_size, char **labels, int classes, int train_set_size,
		int *batch_num_return, int w, int h, int c)
{
	int image_size = h * w * c;
    float *image_all = calloc(train_set_size * image_size, sizeof(float));
    float *truth_all = calloc(train_set_size * classes, sizeof(float));
    load_csv_images(filename, labels, classes, train_set_size, image_size, image_all, truth_all);

    /* random the train image index */
    int train_set_size_real = 0;
    if(train_set_size % batch_size == 0) {
        train_set_size_real = train_set_size;
    } else {
        train_set_size_real = (train_set_size / batch_size + 1) * batch_size;
    }
    /*
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
    }*/

    int batch_num = train_set_size_real / batch_size;
    int less_image = batch_num * batch_size - train_set_size;
    *batch_num_return = batch_num;
    batch *train_data = calloc(batch_num, sizeof(batch));
    for(int i = 0; i < batch_num; i++){
        train_data[i].n = batch_size;
        train_data[i].w = w;
        train_data[i].h = h;
        train_data[i].c = c;
        if(i == batch_num - 1) {
            train_data[i].data = image_all + (i * batch_size - less_image) * image_size;
            train_data[i].truth = truth_all + (i * batch_size - less_image) * classes;
        } else {
            train_data[i].data = image_all + i * batch_size * image_size;
            train_data[i].truth = truth_all + i * batch_size * classes;
        }
    } 
    return train_data;
}

batch random_batch(char **paths, int batch_size, char **labels, int classes, int train_set_size, int w, int h, int c)
{
	int image_size = h * w * c;
    batch b;
    b.w = w;
    b.h = h;
    b.c = c;
    b.n = batch_size;
    b.data = calloc(batch_size * image_size, sizeof(image));
    b.truth = calloc(batch_size * classes, sizeof(float));

    for(int i = 0; i < batch_size; ++i){
        int index = rand() % train_set_size;
        image img = load_image_me(paths[index]);
        memcpy(b.data + i * image_size, img.data, image_size * sizeof(float));
        free_image(img);
        normalize_array(b.data + i * image_size, image_size);
        fill_truth(paths[index], labels, classes, b.truth + i * classes);
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
