#include "maxpool_layer.h"
#include <stdio.h>

image get_maxpool_image(const maxpool_layer *layer)
{
    int h = (layer->h-1)/layer->stride + 1;
    int w = (layer->w-1)/layer->stride + 1;
    int c = layer->c;
    return float_to_image(h,w,c,layer->output);
}

image get_maxpool_delta(const maxpool_layer *layer)
{
    int h = (layer->h-1)/layer->stride + 1;
    int w = (layer->w-1)/layer->stride + 1;
    int c = layer->c;
    return float_to_image(h,w,c,layer->delta);
}

maxpool_layer *make_maxpool_layer(int h, int w, int c, int stride)
{
    fprintf(stderr, "Maxpool:            %d x %d x %d image, %d stride\n", h,w,c,stride);
    maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->stride = stride;
    layer->output = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * c, sizeof(float));
    layer->delta = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * c, sizeof(float));
    return layer;
}

void forward_maxpool_layer(const maxpool_layer *layer, float *in)
{
    image input = float_to_image(layer->h, layer->w, layer->c, in);
    image output = get_maxpool_image(layer);
    int i,j,k;
    for(i = 0; i < output.h*output.w*output.c; ++i) output.data[i] = -FLT_MAX;
    float max = 0.0F;
        float min = 0.0F;
    for(k = 0; k < input.c; ++k){
        for(i = 0; i < input.h; ++i){
            for(j = 0; j < input.w; ++j){
                float val = get_pixel(input, i, j, k);
                float cur = get_pixel(output, i/layer->stride, j/layer->stride, k);
                if(val > cur) set_pixel(output, i/layer->stride, j/layer->stride, k, val);
                if(val > max) max = val;
                            if(val < min) min = val;
            }
        }
    }
    printf("forward_maxpool_layer %f %f\n", max, min);

}

void backward_maxpool_layer(const maxpool_layer *layer, float *in, float *delta)
{
    image input = float_to_image(layer->h, layer->w, layer->c, in);
    image input_delta = float_to_image(layer->h, layer->w, layer->c, delta);
    image output_delta = get_maxpool_delta(layer);
    image output = get_maxpool_image(layer);
    int i,j,k;
    for(k = 0; k < input.c; ++k){
        for(i = 0; i < input.h; ++i){
            for(j = 0; j < input.w; ++j){
                float val = get_pixel(input, i, j, k);
                float cur = get_pixel(output, i/layer->stride, j/layer->stride, k);
                float d = get_pixel(output_delta, i/layer->stride, j/layer->stride, k);
                if(val == cur) {
                    set_pixel(input_delta, i, j, k, d);
                } else {
                    set_pixel(input_delta, i, j, k, 0);
                }
            }
        }
    }
}

