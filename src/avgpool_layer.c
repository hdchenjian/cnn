#include "avgpool_layer.h"

avgpool_layer *make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Avgpool:            %d x %d x %d image -> 1 x 1 x %d image\n", w,h,c, c);
    avgpool_layer *l = calloc(1, sizeof(avgpool_layer));
    l->batch = batch;
    l->type = AVGPOOL;
    l->h = h;
    l->w = w;
    l->c = c;
    l->out_w = 1;
    l->out_h = 1;
    l->out_c = c;
    l->outputs = l->out_c;
    l->inputs = h*w*c;
    int output_size = l->outputs * batch;
    l->output =  calloc(output_size, sizeof(float));
    l->delta =   calloc(output_size, sizeof(float));
    #ifdef GPU
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta, output_size);
    #endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer(const avgpool_layer *l, float *in)
{
    int b,i,k;
    float max = 0.0F;
            float min = 0.0F;
    for(b = 0; b < l->batch; ++b){
        for(k = 0; k < l->c; ++k){
            int out_index = k + b*l->c;
            l->output[out_index] = 0;
            for(i = 0; i < l->h*l->w; ++i){
                int in_index = i + l->h*l->w*(k + b*l->c);
                l->output[out_index] += in[in_index];
            }
            l->output[out_index] /= l->h*l->w;
            if(l->output[out_index] > max) max = l->output[out_index];
            if(l->output[out_index] < min) min = l->output[out_index];
            printf("forward_avgpool_layer %f   ", l->output[out_index]);
        }
    }
    printf("forward_avgpool_layer %f %f\n", max, min);


}

void backward_avgpool_layer(const avgpool_layer *l, float *delta)
{
    int b,i,k;

    for(b = 0; b < l->batch; ++b){
        for(k = 0; k < l->c; ++k){
            int out_index = k + b*l->c;
            for(i = 0; i < l->h*l->w; ++i){
                int in_index = i + l->h*l->w*(k + b*l->c);
                delta[in_index] += l->delta[out_index] / (l->h*l->w);
            }
        }
    }
}

