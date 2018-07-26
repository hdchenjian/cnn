#include "convolutional_layer.h"
#include <float.h>

image get_convolutional_image(const convolutional_layer *layer)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->n;
    return float_to_image(h,w,c,NULL);
}

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, int batch,
                                              ACTIVATION activation, size_t *workspace_size, int batch_normalize, int pad,
                                              float lr_mult, float lr_decay_mult, float bias_mult, float bias_decay_mult,
                                              int weight_filler, float sigma)
{
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
    layer->lr_mult = lr_mult;
    layer->lr_decay_mult = lr_decay_mult;
    layer->bias_mult = bias_mult;
    layer->bias_decay_mult = bias_decay_mult;
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->size = size;
    layer->stride = stride;
    layer->batch = batch;
    layer->weights = calloc(c*n*size*size, sizeof(float));
    if(weight_filler == 1){   // xavier
        float scale = sqrtf(2.0F/(size*size*c));
        for(int i = 0; i < c*n*size*size; ++i){
            layer->weights[i] = scale*rand_uniform(-1, 1);
            //if(i < 5) printf("%d %f\n", i, layer->weights[i]);
        }
        //scale = sqrtf(2./(size*size*c));
        //for(int i = 0; i < c*n*size*size; ++i) layer->weights[i] = scale*rand_normal();

    } else if(weight_filler == 2){   // gaussian
        for(int i = 0; i < c*n*size*size; ++i) layer->weights[i] = rand_normal_me(0, sigma);
    } else {
        fprintf(stderr, "weight_filler not support\n");
        exit(-1);
    }

    layer->weight_updates = calloc(c*n*size*size, sizeof(float));
    layer->biases = calloc(n, sizeof(float));
    layer->bias_updates = calloc(n, sizeof(float));
    layer->out_h = (layer->h-1)/layer->stride + 1;
    layer->out_w = (layer->w-1)/layer->stride + 1;
    // 2.0F: multiplication add
    layer->bflop = (2.0F * layer->size*layer->size*layer->c * layer->n * layer->out_h*layer->out_w) / 1000000000.0F;
    fprintf(
        stderr,
        "Convolutional:      %d x %d x %d inputs, %d weights size %d stride %d -> %d x %d x %d outputs %5.3f BFLOPs\n",
        w,h,c, n, size, stride, layer->out_w, layer->out_h, n, layer->bflop);
    layer->output = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
    layer->delta  = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
    layer->activation = activation;
    if(layer->activation == PRELU){
        layer->bottom_data = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
        layer->slope = calloc(n, sizeof(float));
        for(int i = 0; i < n; i++){
            layer->slope[i] = 0.25F;
        }
        layer->slope_updates = calloc(n, sizeof(float));
    }

    layer->batch_normalize = batch_normalize;
    layer->pad = pad;
    if(batch_normalize){
        layer->scales = calloc(n, sizeof(float));
        layer->scale_updates = calloc(n, sizeof(float));
        for(int i = 0; i < n; ++i){
            layer->scales[i] = 1;
        }

        layer->mean = calloc(n, sizeof(float));
        layer->variance = calloc(n, sizeof(float));

        layer->mean_delta = calloc(n, sizeof(float));
        layer->variance_delta = calloc(n, sizeof(float));

        layer->rolling_mean = calloc(n, sizeof(float));
        layer->rolling_variance = calloc(n, sizeof(float));
        layer->x = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
        layer->x_norm = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
    }
    size_t workspace_size_local = (size_t)(layer->out_h*layer->out_w*size*size*c*sizeof(float));
    if (workspace_size_local > *workspace_size) *workspace_size = workspace_size_local;

#ifdef GPU
    layer->weights_gpu = cuda_make_array(layer->weights, c*n*size*size);
    layer->weight_updates_gpu = cuda_make_array(layer->weight_updates, c*n*size*size);

    layer->biases_gpu = cuda_make_array(layer->biases, n);
    layer->bias_updates_gpu = cuda_make_array(layer->bias_updates, n);

    layer->delta_gpu = cuda_make_array(layer->delta, batch * layer->out_h * layer->out_w * n);
    layer->output_gpu = cuda_make_array(layer->output, batch * layer->out_h * layer->out_w * n);

    if(batch_normalize){
        layer->scales_gpu = cuda_make_array(layer->scales, n);
        layer->scale_updates_gpu = cuda_make_array(layer->scale_updates, n);
        layer->mean_gpu = cuda_make_array(layer->mean, n);
        layer->mean_delta_gpu = cuda_make_array(layer->mean, n);
        layer->variance_delta_gpu = cuda_make_array(layer->variance, n);
        layer->variance_gpu = cuda_make_array(layer->variance, n);
        layer->rolling_mean_gpu = cuda_make_array(layer->mean, n);
        layer->rolling_variance_gpu = cuda_make_array(layer->variance, n);
        layer->x_gpu = cuda_make_array(layer->output, layer->batch * layer->out_h * layer->out_w * n);
        layer->x_norm_gpu = cuda_make_array(layer->output, layer->batch * layer->out_h * layer->out_w * n);
    }
    if(layer->activation == PRELU){
        layer->bottom_data_gpu = cuda_make_array(layer->bottom_data, batch * layer->out_h * layer->out_w * n);
        layer->slope_gpu = cuda_make_array(layer->slope, n);
        layer->slope_updates_gpu = cuda_make_array(layer->slope_updates, n);
    }
#endif
    return layer;
}

float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe! https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im, int channels,  int height,  int width, int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

void col2im_add_pixel(float *im, int height, int width, int channels,
                      int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}

void col2im_cpu(float* data_col, int channels,  int height,  int width, int ksize,  int stride, int pad, float* data_im)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void forward_batchnorm_layer(const convolutional_layer *layer, int test)
{
    if(0 == test){    // 0: train, 1: valid, 2: test
        memcpy(layer->x, layer->output, layer->batch * layer->out_h * layer->out_w * layer->n * sizeof(float));
        mean_cpu(layer->output, layer->batch, layer->n, layer->out_h*layer->out_w, layer->mean);
        variance_cpu(layer->output, layer->mean, layer->batch, layer->n, layer->out_h*layer->out_w, layer->variance);

        scal_cpu(layer->n, .99, layer->rolling_mean, 1);
        axpy_cpu(layer->n, .01, layer->mean, 1, layer->rolling_mean, 1);
        scal_cpu(layer->n, .99, layer->rolling_variance, 1);
        axpy_cpu(layer->n, .01, layer->variance, 1, layer->rolling_variance, 1);

        normalize_cpu(layer->output, layer->mean, layer->variance, layer->batch, layer->n, layer->out_h*layer->out_w);
        memcpy(layer->x_norm, layer->output, layer->batch * layer->out_h * layer->out_w * layer->n * sizeof(float));
    } else {
        normalize_cpu(layer->output, layer->rolling_mean, layer->rolling_variance,
                      layer->batch, layer->n, layer->out_h*layer->out_w);
    }
    scale_bias(layer->output, layer->scales, layer->batch, layer->n, layer->out_h*layer->out_w);
}

void activation_prelu(const convolutional_layer *layer){
    memcpy(layer->bottom_data, layer->output, layer->batch * layer->out_h * layer->out_w * layer->n * sizeof(float));
    int count = layer->batch * layer->out_h * layer->out_w * layer->n;
    int dim = layer->out_h * layer->out_w;
    for (int i = 0; i < count; ++i) {
        int c = (i / dim) % layer->n;
        layer->output[i] = fmaxf(layer->output[i], 0.0F) + layer->slope[c] * fminf(layer->output[i], 0.0F);
      }
}

void forward_convolutional_layer(const convolutional_layer *layer, float *in, float *workspace, int test)
{
    int m = layer->n;
    int n = layer->out_h * layer->out_w;
    int k = layer->size*layer->size*layer->c;
    memset(layer->output, 0, layer->batch * m*n*sizeof(float));
    for(int i = 0; i < layer->batch; ++i){
        float *a = layer->weights;
        float *b = workspace;
        float *c = layer->output + i * m * n;
        if (layer->size == 1){
            b = in + i * layer->w * layer->h * layer->c;
        } else {
            memset(workspace, 0, n*k*sizeof(float));
            im2col_cpu(in + i * layer->w * layer->h * layer->c,
                       layer->c,  layer->h,  layer->w,  layer->size,  layer->stride, layer->pad, b);
        }
        gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
    }

    if(layer->batch_normalize){
        forward_batchnorm_layer(layer, test);
    }
    for(int b = 0; b < layer->batch; ++b){
        for(int i = 0; i < layer->n; ++i){
            for(int j = 0; j < n; ++j){
                layer->output[(b*layer->n + i)*n + j] += layer->biases[i];
            }
        }
    }

    if(layer->activation == PRELU){
        /*int num = 5;
        for(int b = 0; b < num; ++b){
            printf("%d %f\n", b, layer->output[b]);
        }
        */
        activation_prelu(layer);
    } else if (layer->activation == LINEAR) {
    } else {
        for(int i = 0; i < layer->batch * m*n; ++i) layer->output[i] = activate(layer->output[i], layer->activation);
    }

    /*
    float max = -FLT_MAX, min = FLT_MAX;
    for(int i = 0; i < layer->batch * m*n; ++i){
    	if(layer->output[i] > max) max = layer->output[i];
    	if(layer->output[i] < min) min = layer->output[i];
    }
    printf("forward_convolutional_layer max: %f, min: %f\n", max, min);*/
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrtf(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters,
                         int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta,
                         int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrtf(variance[f] + .00001f)) +
                    variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void backward_batchnorm_layer(const convolutional_layer *layer, int test)
{
    if(0 != test){    // 0: train, 1: valid, 2: test
        fprintf(stderr, "backward_batchnorm_layer: use no used!\n");
        exit(-1);
        //layer->mean = layer->rolling_mean;
        //layer->variance = layer->rolling_variance;
    }
    backward_scale_cpu(layer->x_norm, layer->delta, layer->batch, layer->n, layer->out_w*layer->out_h, layer->scale_updates);
    scale_bias(layer->delta, layer->scales, layer->batch, layer->n, layer->out_h*layer->out_w);

    mean_delta_cpu(layer->delta, layer->variance, layer->batch, layer->n, layer->out_w*layer->out_h, layer->mean_delta);
    variance_delta_cpu(layer->x, layer->delta, layer->mean, layer->variance, layer->batch, layer->n,
            layer->out_w*layer->out_h, layer->variance_delta);
    normalize_delta_cpu(layer->x, layer->mean, layer->variance, layer->mean_delta, layer->variance_delta,
            layer->batch, layer->n, layer->out_w*layer->out_h, layer->delta);
}

void backward_convolutional_layer(const convolutional_layer *layer, float *input, float *delta,
                                  float *workspace, int test)
{
    int outputs = layer->batch * layer->out_h * layer->out_w * layer->n;
    if(layer->activation == PRELU){
        int count = layer->batch * layer->out_h * layer->out_w * layer->n;
        int dim = layer->out_h * layer->out_w;
        for (int i = 0; i < count; ++i) {
            int cc = (i / dim) % layer->n;
            layer->slope_updates[cc] += layer->delta[i] * layer->bottom_data[i] * (layer->bottom_data[i] <= 0);
            layer->delta[i] = layer->delta[i] * ((layer->bottom_data[i] > 0) + layer->slope[cc] * (layer->bottom_data[i] <= 0));
        }
    } else if (layer->activation == LINEAR) {
    } else {
        for(int i = 0; i < outputs; ++i){
            layer->delta[i] *= gradient(layer->output[i], layer->activation);
        }
    }
    for(int j = 0; j < layer->batch; ++j){
        for(int i = 0; i < layer->n; ++i){
            layer->bias_updates[i] += sum_array(layer->delta + layer->out_h * layer->out_w * (i + j*layer->n),
                                                layer->out_h * layer->out_w);
        }
    }
    if(layer->batch_normalize){
        backward_batchnorm_layer(layer, test);
    }
    for(int j = 0; j < layer->batch; ++j){
        int m = layer->n;
        int n = layer->size*layer->size*layer->c;
        int k = layer->out_w * layer->out_h;
        float *a = layer->delta + j * m * k;
        float *b = workspace;
        float *c = layer->weight_updates;
        float *im  = input + j*layer->c*layer->h*layer->w;
        if(layer->size == 1){
            b = im;
        } else {
            memset(workspace, 0, n*k*sizeof(float));
            im2col_cpu(im, layer->c, layer->h, layer->w, layer->size, layer->stride, layer->pad, b);
        }
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if (delta) {  // not first layer
            memset(delta + j * layer->h * layer->w * layer->c, 0, layer->h * layer->w * layer->c * sizeof(float));
            m = layer->size*layer->size*layer->c;
            n = layer->out_w * layer->out_h;
            k = layer->n;
            a = layer->weights;
            b = layer->delta + j * n * k;
            c = workspace;
            if (layer->size == 1) {
                c = delta + j * layer->h * layer->w * layer->c;
            }
            gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);
            if (layer->size != 1) {
                col2im_cpu(workspace, layer->c, layer->h, layer->w, layer->size, layer->stride, layer->pad,
                           delta + j * layer->h * layer->w * layer->c);
            }
        }
    }
}

void update_convolutional_layer(const convolutional_layer *layer, float learning_rate, float momentum, float decay)
{
    if(layer->batch_normalize){
        for(int i = 0; i < layer->n; i ++){
            layer->scales[i] += learning_rate / layer->batch * layer->scale_updates[i];
            layer->scale_updates[i] *= momentum;
        }
    }

    if(layer->activation == PRELU){
        for(int i = 0; i < layer->n; i ++){
            layer->slope[i] += learning_rate / layer->batch * layer->slope_updates[i];
            layer->slope_updates[i] *= momentum;
        }
    }

    for(int i = 0; i < layer->n; i ++){
        layer->bias_updates[i] += -decay * layer->bias_decay_mult * layer->batch * layer->biases[i];
        layer->biases[i] += learning_rate * layer->bias_mult / layer->batch * layer->bias_updates[i];
        layer->bias_updates[i] *= momentum;
    }

    int size = layer->size*layer->size*layer->c*layer->n;
    for(int i = 0; i < size; i ++){
        layer->weight_updates[i] += -decay * layer->lr_decay_mult * layer->batch * layer->weights[i];
        layer->weights[i] += learning_rate * layer->lr_mult / layer->batch * layer->weight_updates[i];
        layer->weight_updates[i] *= momentum;
    }
}
