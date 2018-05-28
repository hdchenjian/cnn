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
    b.images = calloc(batch_size, sizeof(image));
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
        free_image(b.images[i]);
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

batch *load_csv_image_to_memory(char *filename, int batch_size, char **labels, int classes, int train_set_size)
{
    FILE *fp = fopen(filename, "r");
    if(!fp) file_error(filename);
    batch *train_data = calloc(train_set_size, sizeof(batch));
    int fields = 0;  // the number of pixel per image
    int w, h;
    int n = 0;
    char *line;
    while((line = fgetl(fp)) && (n < train_set_size)){
        train_data[n].n = batch_size;
        char class = line[0];
        if(0 == fields){
            fields = count_fields(line);
            w = sqrt(fields);
            h = sqrt(fields);
        }
        float *value = parse_fields(line, fields);
        image *im = calloc(1, sizeof(image));
        im->h = h;
        im->w = w;
        im->c = 1;
        im->data = value + 1;
        normalize_array(im->data, im->h*im->w*im->c);
        train_data[n].images = im;
        train_data[n].truth = calloc(batch_size, sizeof(float *));
        for(int i =0 ; i < batch_size; ++i){
            train_data[n].truth[i] = calloc(classes, sizeof(float));
            char name[16] = {0};
            sprintf(name, "%c.png", class);
            fill_truth(name, labels, classes, train_data[n].truth[i]);
        }
        free(line);
        n += 1;
    }
    fclose(fp);
    return train_data;
}

batch random_batch(char **paths, int batch_size, char **labels, int classes, int train_set_size)
{
    batch b = make_batch(batch_size, classes);
    for(int i = 0; i < batch_size; ++i){
        int index = rand() % train_set_size;
        b.images[i] = load_image_me(paths[index]);
        normalize_array(b.images[i].data, b.images[i].h*b.images[i].w*b.images[i].c);

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
