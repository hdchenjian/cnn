#include "maxpool_layer.h"
#include <stdio.h>

image get_maxpool_image(const maxpool_layer *layer, int batch)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->c;
    return float_to_image(h,w,c,layer->output + batch * h * w * c);
}

image get_maxpool_delta(const maxpool_layer *layer, int batch)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->c;
    return float_to_image(h,w,c,layer->delta + batch * h * w * c);
}

maxpool_layer *make_maxpool_layer(int h, int w, int c, int size, int stride, int batch, int padding)
{
    fprintf(stderr, "Maxpool:            %d x %d x %d inputs, size: %d, %d stride\n", h,w,c,size,stride);
    maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->size = size;
    layer->stride = stride;
    layer->batch = batch;
    layer->pad = padding;
    layer->out_w = (w + 2*padding)/stride;
    layer->out_h = (h + 2*padding)/stride;
    layer->output = calloc(batch * layer->out_h * layer->out_w * c, sizeof(float));
    layer->delta = calloc(batch * layer->out_h * layer->out_w * c, sizeof(float));
    layer->indexes = calloc(layer->out_h * layer->out_w * layer->c * batch, sizeof(int));
#ifdef GPU
    layer->indexes_gpu = cuda_make_int_array(0, layer->out_h * layer->out_w * layer->c * batch);
    layer->output_gpu = cuda_make_array(layer->output, layer->out_h * layer->out_w * layer->c * batch);
    layer->delta_gpu = cuda_make_array(layer->delta, layer->out_h * layer->out_w * layer->c * batch);
#endif
    return layer;
}

void forward_maxpool_layer(const maxpool_layer *layer, float *in)
{
    int b,i,j,k,m,n;
    int w_offset = -layer->pad;
    int h_offset = -layer->pad;

    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->c;

    for(b = 0; b < layer->batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < layer->size; ++n){
                        for(m = 0; m < layer->size; ++m){
                            int cur_h = h_offset + i*layer->stride + n;
                            int cur_w = w_offset + j*layer->stride + m;
                            int index = cur_w + layer->w*(cur_h + layer->h*(k + b*layer->c));
                            int valid = (cur_h >= 0 && cur_h < layer->h &&
                                         cur_w >= 0 && cur_w < layer->w);
                            float val = (valid != 0) ? in[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    layer->output[out_index] = max;
                    layer->indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer *layer, float *in, float *delta)
{
    int i;
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->c;
    for(i = 0; i < h*w*c*layer->batch; ++i){
        int index = layer->indexes[i];
        delta[index] += layer->delta[i];
    }
}
