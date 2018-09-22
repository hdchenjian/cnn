#include "avgpool_layer.h"

image get_avgpool_image(const avgpool_layer *layer)
{
    int h = 1;
    int w = 1;
    int c = layer->c;
    return float_to_image(h,w,c,NULL);
}

avgpool_layer *make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Avgpool:            %d x %d x %d image -> 1 x 1 x %d image\n", w,h,c, c);
    avgpool_layer *l = calloc(1, sizeof(avgpool_layer));
    l->batch = batch;
    l->h = h;
    l->w = w;
    l->c = c;
    l->outputs = c;
    l->output = calloc(l->outputs * batch, sizeof(float));
    l->delta = calloc(l->outputs * batch, sizeof(float));
    #ifdef GPU
    l->output_gpu  = cuda_make_array(l->output, l->outputs * batch);
    l->delta_gpu   = cuda_make_array(l->delta, l->outputs * batch);
    #endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
}

void forward_avgpool_layer(const avgpool_layer *l, float *in)
{
    for(int b = 0; b < l->batch; ++b){
        for(int k = 0; k < l->c; ++k){
            int out_index = k + b*l->c;
            float sum = 0;
            for(int i = 0; i < l->h*l->w; ++i){
                int in_index = i + l->h*l->w*(k + b*l->c);
                sum += in[in_index];
            }
            l->output[out_index] = sum / (l->h*l->w);
            //printf("forward_avgpool_layer: %f\n", l->output[k]);
        }
    }
}

void backward_avgpool_layer(const avgpool_layer *l, float *delta)
{
    for(int b = 0; b < l->batch; ++b){
        for(int k = 0; k < l->c; ++k){
            int out_index = k + b*l->c;
            float temp = l->delta[out_index];
            //printf("backward_avgpool_layer: %f\n", temp);
            for(int i = 0; i < l->h*l->w; ++i){
                int in_index = i + l->h*l->w*(k + b*l->c);
                delta[in_index] += temp / (l->h*l->w);
            }
        }
    }
}
