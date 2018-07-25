#include "data.h"
#include "list.h"
#include "utils.h"
#include "image.h"

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

void free_batch(batch *b)
{
    free(b->data);
    b->data = NULL;
    free(b->truth_label_index);
    b->truth_label_index = NULL;
}


void fill_truth(char *path, char **labels, int classes, int *truth_label_index)
{
    for(int i = 0; i < classes; ++i){
        if(strstr(path, labels[i])){
            *truth_label_index = i;
            break;
        }
    }
}

void load_csv_images(char *filename, char **labels, int classes, int train_set_size, int image_size, float *image_all,
                     int *truth_lable_all, float hue, float saturation, float exposure, int test)
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
            w = sqrtf(fields);
            h = sqrtf(fields);
        }
        float *value = parse_fields(line, fields);
        memcpy(image_all + n * image_size, value + 1, image_size * sizeof(float));
        image crop;
        crop.w = w;
        crop.h = h;
        crop.c = 1;
        crop.data = image_all + n * image_size;
        if(crop.c == 3 && test == 0) {      // 0: train, 1: valid, 2: test
            random_distort_image(crop, hue, saturation, exposure);
        }
        normalize_array(image_all + n * image_size, image_size);
        /*float max = -FLT_MAX, min = FLT_MAX;
        for(int i = 0; i < crop.w * crop.h * crop.c; ++i){
            if(crop.data[i] > max) max = crop.data[i];
            if(crop.data[i] < min) min = crop.data[i];
        }
        printf("input image max: %f, min: %f\n", max, min);
        exit(-1);*/

        char name[16] = {0};
        sprintf(name, "%c.png", class);
        fill_truth(name, labels, classes, truth_lable_all + n);
        free(line);
        free(value);
        n += 1;
    }
    fclose(fp);
}

int *get_random_index(int train_set_size, int train_set_size_real)
{
    /* random the train image index */
    int *index = calloc(train_set_size_real, sizeof(int));
    for(int i = 0; i < train_set_size_real; i++) {
        if(i < train_set_size) {
            index[i] = i;
        } else {
            index[i] = rand() % train_set_size;
        }
    }
    for(int j = 0; j < 5; j++){
        for(int i = 0; i < train_set_size_real; i++) {
            int index_random = rand() % train_set_size_real;
            int temp = index[i];
            index[i] = index[index_random];
            index[index_random] = temp;
        }
    }
    return index;
}

batch *load_csv_image_to_memory(char *filename, int batch_size, char **labels, int classes, int train_set_size,
                                int *batch_num_return, int w, int h, int c, float hue, float saturation,
                                float exposure, int test)
{
    int image_size = h * w * c;
    float *image_all = calloc(train_set_size * image_size, sizeof(float));
    int *truth_lable_all = calloc(train_set_size, sizeof(int));
    load_csv_images(filename, labels, classes, train_set_size, image_size, image_all, truth_lable_all,
                    hue, saturation, exposure, test);

    int train_set_size_real = 0;
    if(train_set_size % batch_size == 0) {
        train_set_size_real = train_set_size;
    } else {
        train_set_size_real = (train_set_size / batch_size + 1) * batch_size;
    }
    int *index = get_random_index(train_set_size, train_set_size_real);

    int batch_num = train_set_size_real / batch_size;
    *batch_num_return = batch_num;
    batch *train_data = calloc(batch_num, sizeof(batch));
    for(int i = 0; i < batch_num; i++){
        train_data[i].n = batch_size;
        train_data[i].w = w;
        train_data[i].h = h;
        train_data[i].c = c;
        train_data[i].data = calloc(batch_size * image_size, sizeof(float));
        train_data[i].truth_label_index = calloc(batch_size, sizeof(int));
        for(int j = 0; j < batch_size; ++j){
            int image_index = 0;
            if(test == 0) {      // 0: train, 1: valid, 2: test
                image_index = index[i * batch_size + j];
            } else {
                image_index = i * batch_size + j;
            }
            memcpy(train_data[i].data + j * image_size, image_all + image_size * image_index, image_size * sizeof(float));
            train_data[i].truth_label_index[j] = truth_lable_all[image_index];
        }
    }
    free_ptr(index);
    free_ptr(image_all);
    free_ptr(truth_lable_all);
    return train_data;
}

batch *load_image_to_memory(char **paths, int batch_size, char **labels, int classes, int train_set_size,
                            int *batch_num_return, int w, int h, int c, float hue, float saturation, float exposure,
                            int flip, float mean_value, float scale, int test)
{
    double time = what_time_is_it_now();
    int train_set_size_real = 0;
    if(train_set_size % batch_size == 0) {
        train_set_size_real = train_set_size;
    } else {
        train_set_size_real = (train_set_size / batch_size + 1) * batch_size;
    }
    int *index = get_random_index(train_set_size, train_set_size_real);
    int image_size = h * w * c;
    int batch_num = train_set_size_real / batch_size;
    *batch_num_return = batch_num;
    batch *train_data = calloc(batch_num, sizeof(batch));
    for(int i = 0; i < batch_num; i++){
        train_data[i].n = batch_size;
        train_data[i].w = w;
        train_data[i].h = h;
        train_data[i].c = c;
        train_data[i].data = calloc(batch_size * image_size, sizeof(float));
        train_data[i].truth_label_index = calloc(batch_size, sizeof(int));
        for(int j = 0; j < batch_size; ++j){
            char *image_path = NULL;
            if(test == 0) {      // 0: train, 1: valid, 2: test
                image_path = paths[index[i * batch_size + j]];
            } else {
                image_path = paths[i * batch_size + j];
            }
            image img = load_image(image_path, w, h, c);
            if(test == 0) {      // 0: train, 1: valid, 2: test
                if(flip){
                    if(rand() % 2){
                        flip_image(img);
                    }
                }
                random_distort_image(img, hue, saturation, exposure);
            }
            if(mean_value > 0.001){
                for(int k = 0; k < image_size; ++k){
                    // load_image_stb divide 255.0F
                    img.data[k] = (img.data[k] * 255.0F - mean_value) * scale;
                }
            }

            memcpy(train_data[i].data + j * image_size, img.data, image_size * sizeof(float));
            free_image(img);
            //normalize_array(b.data + i * image_size, image_size);
            fill_truth(image_path, labels, classes, train_data[i].truth_label_index + j);
        }
    }
    free_ptr(index);
    printf("load_image_to_memory done: use memory: %f MB, spend %f s\n",
           batch_num * batch_size * (image_size + classes ) * sizeof(float) / 1024.0 / 1024.0,
           what_time_is_it_now() - time);
    return train_data;
}

batch random_batch(char **paths, int batch_size, char **labels, int classes, int train_set_size, int w, int h, int c,
                   float hue, float saturation, float exposure, int flip, float mean_value, float scale, int test)
{
    static int test_index = 0;
    int image_size = h * w * c;
    batch b;
    b.w = w;
    b.h = h;
    b.c = c;
    b.n = batch_size;
    b.data = calloc(batch_size * image_size, sizeof(float));
    //b.truth = calloc(batch_size * classes, sizeof(float));
    b.truth_label_index = calloc(batch_size, sizeof(int));

    #pragma omp parallel for
    for(int i = 0; i < batch_size; ++i){
        int index = 0;
        image img;
        if(test == 0) {      // 0: train, 1: valid, 2: test
            index = rand() % train_set_size;
            img = load_image(paths[index], w, h, c);
            if(flip){
                if(rand() % 2){
                    flip_image(img);
                }
            }
            random_distort_image(img, hue, saturation, exposure);
        } else {
            index = test_index;
            //printf("index: %d %s %f %f\n", index, paths[index], mean_value, scale);
            img = load_image(paths[index], w, h, c);
            test_index += 1;
        }
        //save_image_png(img, "input.jpg");
        //flip_image(img);
        if(mean_value > 0.001){
            for(int k = 0; k < image_size; ++k){
                // load_image_stb divide 255.0F
                img.data[k] = (img.data[k] * 255.0F - mean_value) * scale;
            }
        }
        memcpy(b.data + i * image_size, img.data, image_size * sizeof(float));
        /*float max = -FLT_MAX, min = FLT_MAX;
        for(int i = 0; i < img.w * img.h * img.c; ++i){
            if(img.data[i] > max) max = img.data[i];
            if(img.data[i] < min) min = img.data[i];
        }
        printf("input image max: %f, min: %f\n", max, min);
        */
        free_image(img);
        fill_truth(paths[index], labels, classes, b.truth_label_index + i);
        //printf("index: %d %s %d %f %f\n", index, paths[index], b.truth_label_index[i], mean_value, scale);
        //exit(-1);
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
