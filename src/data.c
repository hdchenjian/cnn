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
    if(classes < 3) classes = 1;
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

batch random_batch(char **paths, int batch_size, char **labels, int classes, int train_set_size)
{
    batch b = make_batch(batch_size, 1);
    for(int i = 0; i < batch_size; ++i){
        int index = rand() % train_set_size;
        b.images[i] = load_image_me(paths[index]);
        //scale_image(b.images[i], 1./255.);
        z_normalize_image(b.images[i]);
        fill_truth(paths[index], labels, classes, b.truth[i]);
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
