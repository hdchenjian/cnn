#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "blas.h"

typedef struct {
    int h;
    int w;
    int c;
    float *data;
} image;

float get_color(int c, int x, int max);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_bbox(image a, box bbox, int w, float r, float g, float b);
void write_label(image a, int r, int c, image *characters, char *string, float *rgb);
image image_distance(image a, image b);
void scale_image(image m, float s);
image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect);
image random_crop_image(image im, int w, int h);
image random_augment_image(image im, float angle, float aspect, int low, int high, int w, int h);
void letterbox_image_into(image im, int w, int h, image boxed);
image letterbox_image(image im, int w, int h);
image resize_max(image im, int max);
void translate_image(image m, float s);
void embed_image(image source, image dest, int dx, int dy);
void place_image(image im, int w, int h, int dx, int dy, image canvas);
void saturate_image(image im, float sat);
void exposure_image(image im, float sat);
void distort_image(image im, float hue, float sat, float val);
void saturate_exposure_image(image im, float sat, float exposure);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
void yuv_to_rgb(image im);
void rgb_to_yuv(image im);


image collapse_image_layers(image source, int border);
image collapse_images_horz(image *ims, int n);
image collapse_images_vert(image *ims, int n);

void show_image_normalized(image im, const char *name);
void show_images(image *ims, int n, char *window);
void show_image_layers(image p, char *name);
void show_image_collapsed(image p, char *name);
image make_image(int w, int h, int c);
image copy_image(image p);
void print_image(image m);
void free_image(image m);
image load_image_color(char *filename, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image resize_image(image im, int w, int h);
image load_image(char *filename, int w, int h, int c);

void save_image_png(image im, const char *name);
image float_to_image(int h, int w, int c, float *data);
image make_random_kernel(int size, int c, float scale);
void show_image(image p, const char *name);
void back_convolve(image in_delta, image kernel, int stride, int channel, image out_delta);
void kernel_update(image m, image update, int stride, int channel, image out_delta);
void convolve(image m, image kernel, int stride, int channel, image out);
void zero_image(image m);
void two_d_convolve(image m, int mc, image kernel, int kc, int stride, image out, int oc);
float get_pixel(image m, int x, int y, int c);
void set_pixel(image m, int x, int y, int c, float val);
float avg_image_layer(image m, int l);
void normalize_image(image p);
void random_distort_image(image im, float hue, float saturation, float exposure);

image make_empty_image(int w, int h, int c);
void copy_image_into(image src, image dest);

image get_image_layer(image m, int l);
void flip_image(image a);
void fill_image(image m, float s);
#endif

