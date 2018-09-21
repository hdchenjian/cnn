#include "upsample_layer.h"
#include "utils.h"

image get_upsample_image(const upsample_layer *layer)
{
    return float_to_image(layer->out_h, layer->out_w, layer->out_c, NULL);
}

upsample_layer *make_upsample_layer(int batch, int w, int h, int c, int stride)
{
    upsample_layer *l = calloc(1, sizeof(upsample_layer));
    l->batch = batch;
    l->w = w;
    l->h = h;
    l->c = c;
    l->scale = 1;
    l->out_w = w*stride;
    l->out_h = h*stride;
    l->out_c = c;
    l->stride = stride;
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->w*l->h*l->c;
    l->delta =  calloc(l->outputs*batch, sizeof(float));
    l->output = calloc(l->outputs*batch, sizeof(float));;

    #ifdef GPU
    l->delta_gpu =  cuda_make_array(l->delta, l->outputs*batch);
    l->output_gpu = cuda_make_array(l->output, l->outputs*batch);
    #endif
    fprintf(stderr, "upsample          %4d x%4d x%4d   ->  %4d x%4d x%4d, stride: %d\n", w, h, c, l->out_w, l->out_h, l->out_c, stride);
    return l;
}

void free_upsample_layer(void *input)
{
    upsample_layer *layer = (upsample_layer *)input;
    if(layer->output) free_ptr(layer->output);
    if(layer->delta) free_ptr(layer->delta);
#ifdef GPU
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#endif
    free_ptr(layer);
}

void forward_upsample_layer(const upsample_layer *l, float *input)
{
    upsample_cpu(input, l->w, l->h, l->c, l->batch, l->stride, 1, l->scale, l->output);
}

void backward_upsample_layer(const upsample_layer *l, float * delta)
{
    //memset(delta, 0, l->h*l->w*l->c*l->batch * sizeof(float));
    upsample_cpu(delta, l->w, l->h, l->c, l->batch, l->stride, 0, l->scale, l->delta);
}

#ifdef GPU
void forward_upsample_layer_gpu(const upsample_layer *l, float *input)
{
    upsample_gpu(l->output_gpu, l->out_w, l->out_h, l->c, l->batch, l->stride, 0, l->scale, input);
}

void backward_upsample_layer_gpu(const upsample_layer *l, float *delta)
{
    //fill_gpu(l->h*l->w*l->c*l->batch, 0, delta, 1);
    upsample_gpu(delta, l->w, l->h, l->c, l->batch, l->stride, 0, l->scale, l->delta_gpu);
}
#endif
