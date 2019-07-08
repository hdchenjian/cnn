#include "convolutional_layer.h"
#include <float.h>

#ifdef USE_LINUX
#include <pthread.h>
#endif

image get_convolutional_image(const convolutional_layer *layer)
{
    int h = layer->out_h;
    int w = layer->out_w;
    int c = layer->n;
    return float_to_image(h,w,c,NULL);
}

#ifdef CUDNN
void cudnn_convolutional_setup(convolutional_layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->n, l->out_h, l->out_w);

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->n, l->out_h, l->out_w);
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->n, 1, 1);

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size);
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c, l->size, l->size);
#if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
#else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
#endif

#if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, 1);
#else
    printf("CUDNN < 7 doesn't support groups, please upgrade!");
#endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            4000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            4000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            4000000000,
            &l->bf_algo);
}
#endif

size_t get_workspace_size(convolutional_layer *layer){
#ifdef CUDNN
    size_t most = 0;
    size_t s = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(), layer->srcTensorDesc, layer->weightDesc,
                                            layer->convDesc, layer->dstTensorDesc, layer->fw_algo, &s);
    if (s > most) most = s;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(), layer->srcTensorDesc, layer->ddstTensorDesc,
                                                   layer->convDesc, layer->dweightDesc, layer->bf_algo, &s);
    if (s > most) most = s;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(), layer->weightDesc, layer->ddstTensorDesc,
                                                 layer->convDesc, layer->dsrcTensorDesc, layer->bd_algo, &s);
    if (s > most) most = s;
    return most;
#elif defined(OPENCL)
    int tile_width = 8;
    int k = ((layer->size*layer->size*layer->c + tile_width - 1) / tile_width) * tile_width;
    int n = ((layer->out_h*layer->out_w + tile_width - 1) / tile_width) * tile_width;
    return (size_t)(n * k *sizeof(float));
#else
    return (size_t)(layer->out_h*layer->out_w*layer->size*layer->size*layer->c*sizeof(float));
#endif
}

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, int batch,
                                              ACTIVATION activation, size_t *workspace_size, int batch_normalize, int pad,
                                              float lr_mult, float lr_decay_mult, float bias_mult, float bias_decay_mult,
                                              int weight_filler, float sigma, int subdivisions, int test)
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
    layer->subdivisions = subdivisions;
    layer->test = test;
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

    layer->biases = calloc(n, sizeof(float));
    int padding = 0;
    if(pad) padding = size / 2;
    layer->pad = padding;
    layer->out_h = (layer->h + 2*layer->pad - layer->size) / layer->stride + 1;
    layer->out_w = (layer->w + 2*layer->pad - layer->size) / layer->stride + 1;
    // 2.0F: multiplication add
    layer->bflop = (2.0F * layer->size*layer->size*layer->c * layer->n * layer->out_h*layer->out_w) / 1000000000.0F;
    layer->outputs = layer->out_h * layer->out_w * layer->n;
#ifndef FORWARD_GPU
    layer->output = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
#endif
    if(0 == layer->test){    // 0: train, 1: valid
        layer->delta  = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
        layer->weight_updates = calloc(c*n*size*size, sizeof(float));
        layer->bias_updates = calloc(n, sizeof(float));
    }
    layer->activation = activation;
    if(layer->activation == PRELU){
        if(0 == layer->test){    // 0: train, 1: valid
            layer->bottom_data = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
            layer->slope_updates = calloc(n, sizeof(float));
        }
        layer->slope = calloc(n, sizeof(float));
        for(int i = 0; i < n; i++) layer->slope[i] = 0.25F;
    }

    layer->batch_normalize = batch_normalize;
    if(batch_normalize){
        layer->scales = calloc(n, sizeof(float));
        layer->rolling_mean = calloc(n, sizeof(float));
        layer->rolling_variance = calloc(n, sizeof(float));
        if(0 == layer->test){    // 0: train, 1: valid
            layer->scale_updates = calloc(n, sizeof(float));
            for(int i = 0; i < n; ++i) layer->scales[i] = 1;
            layer->mean = calloc(n, sizeof(float));
            layer->variance = calloc(n, sizeof(float));
            layer->mean_delta = calloc(n, sizeof(float));
            layer->variance_delta = calloc(n, sizeof(float));
            layer->x = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
            layer->x_norm = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
        }
    }

#ifdef GPU
    layer->weights_gpu = cuda_make_array(layer->weights, c*n*size*size);
    layer->biases_gpu = cuda_make_array(layer->biases, n);
    layer->output_gpu = cuda_make_array(layer->output, batch * layer->out_h * layer->out_w * n);
    if(0 == layer->test){    // 0: train, 1: valid
        layer->weight_updates_gpu = cuda_make_array(layer->weight_updates, c*n*size*size);
        layer->bias_updates_gpu = cuda_make_array(layer->bias_updates, n);
        layer->delta_gpu = cuda_make_array(layer->delta, batch * layer->out_h * layer->out_w * n);
    }
    if(batch_normalize){
        layer->scales_gpu = cuda_make_array(layer->scales, n);
        layer->rolling_mean_gpu = cuda_make_array(layer->mean, n);
        layer->rolling_variance_gpu = cuda_make_array(layer->variance, n);
        if(0 == layer->test){    // 0: train, 1: valid
            layer->scale_updates_gpu = cuda_make_array(layer->scale_updates, n);
            layer->mean_gpu = cuda_make_array(layer->mean, n);
            layer->mean_delta_gpu = cuda_make_array(layer->mean, n);
            layer->variance_delta_gpu = cuda_make_array(layer->variance, n);
            layer->variance_gpu = cuda_make_array(layer->variance, n);
            layer->x_gpu = cuda_make_array(layer->output, layer->batch * layer->out_h * layer->out_w * n);
            layer->x_norm_gpu = cuda_make_array(layer->output, layer->batch * layer->out_h * layer->out_w * n);
        }
    }
    if(layer->activation == PRELU){
        if(0 == layer->test){    // 0: train, 1: valid
            layer->bottom_data_gpu = cuda_make_array(layer->bottom_data, batch * layer->out_h * layer->out_w * n);
            layer->slope_updates_gpu = cuda_make_array(layer->slope_updates, n);
        }
        layer->slope_gpu = cuda_make_array(layer->slope, n);
    }
    #ifdef CUDNN
    cudnnCreateTensorDescriptor(&layer->normTensorDesc);
    cudnnCreateTensorDescriptor(&layer->srcTensorDesc);
    cudnnCreateTensorDescriptor(&layer->dstTensorDesc);
    cudnnCreateFilterDescriptor(&layer->weightDesc);
    cudnnCreateTensorDescriptor(&layer->dsrcTensorDesc);
    cudnnCreateTensorDescriptor(&layer->ddstTensorDesc);
    cudnnCreateFilterDescriptor(&layer->dweightDesc);
    cudnnCreateConvolutionDescriptor(&layer->convDesc);
    cudnn_convolutional_setup(layer);
    #endif
#elif defined(OPENCL)
    layer->weights_cl = cl_make_weights(layer->n, layer->size*layer->size*layer->c, layer->weights);
    //layer->weights_cl = cl_make_array(layer->weights, c*n*size*size);
    layer->biases_cl = cl_make_array(layer->biases, n);
    //layer->output_cl = cl_make_share_array(layer->output, batch * layer->out_h * layer->out_w * n);
    layer->output_cl = cl_make_array(layer->output, batch * layer->out_h * layer->out_w * n);
    if(0 == layer->test){    // 0: train, 1: valid
        layer->weight_updates_cl = cl_make_array(layer->weight_updates, c*n*size*size);
        layer->bias_updates_cl = cl_make_array(layer->bias_updates, n);
        layer->delta_cl = cl_make_array(layer->delta, batch * layer->out_h * layer->out_w * n);
    }
    if(batch_normalize){
        layer->scales_cl = cl_make_array(layer->scales, n);
        layer->rolling_mean_cl = cl_make_array(layer->rolling_mean, n);
        layer->rolling_variance_cl = cl_make_array(layer->rolling_variance, n);
        if(0 == layer->test){    // 0: train, 1: valid
            layer->scale_updates_cl = cl_make_array(layer->scale_updates, n);
            layer->mean_cl = cl_make_array(layer->mean, n);
            layer->mean_delta_cl = cl_make_array(layer->mean, n);
            layer->variance_delta_cl = cl_make_array(layer->variance, n);
            layer->variance_cl = cl_make_array(layer->variance, n);
            layer->x_cl = cl_make_array(layer->output, layer->batch * layer->out_h * layer->out_w * n);
            layer->x_norm_cl = cl_make_array(layer->output, layer->batch * layer->out_h * layer->out_w * n);
        }
    }
    if(layer->activation == PRELU){
        if(0 == layer->test){    // 0: train, 1: valid
            layer->bottom_data_cl = cl_make_array(layer->bottom_data, batch * layer->out_h * layer->out_w * n);
            layer->slope_updates_cl = cl_make_array(layer->slope_updates, n);
        }
        layer->slope_cl = cl_make_array(layer->slope, n);
    }
#endif

    layer->workspace_size = get_workspace_size(layer);
    if (layer->workspace_size > *workspace_size) *workspace_size = layer->workspace_size;
    float Mb_size = 1024 * 1024;
    fprintf(
        stderr,
        "Convolutional:      %d x %d x %d inputs, %d weights size %d stride %d -> %d x %d x %d outputs, %.2fMb %5.3f BFLOPs\n",
        w,h,c, n, size, stride, layer->out_w, layer->out_h, n, layer->workspace_size / Mb_size, layer->bflop);
    return layer;
}

float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu_thread(float* data_im, int channels,  int height,  int width, int ksize,  int stride, int pad, float* data_col, int n_tile);
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

void forward_conv_batchnorm_layer(const convolutional_layer *layer, int test, int index)
{
    if(0 == test){    // 0: train, 1: valid
        memcpy(layer->x, layer->output, layer->batch * layer->out_h * layer->out_w * layer->n * sizeof(float));
        mean_cpu(layer->output, layer->batch, layer->n, layer->out_h*layer->out_w, layer->mean);
        variance_cpu(layer->output, layer->mean, layer->batch, layer->n, layer->out_h*layer->out_w, layer->variance);

        scal_cpu(layer->n, .99, layer->rolling_mean, 1);
        axpy_cpu(layer->n, .01, layer->mean, 1, layer->rolling_mean, 1);
        scal_cpu(layer->n, .99, layer->rolling_variance, 1);
        axpy_cpu(layer->n, .01, layer->variance, 1, layer->rolling_variance, 1);

        normalize_cpu(layer->output, layer->mean, layer->variance, layer->batch, layer->n, layer->out_h*layer->out_w);
        memcpy(layer->x_norm, layer->output, layer->batch * layer->out_h * layer->out_w * layer->n * sizeof(float));
        scale_bias(layer->output, layer->scales, layer->batch, layer->n, layer->out_h*layer->out_w);
    } else {
        normalize_cpu(layer->output, layer->rolling_mean, layer->rolling_variance,
                      layer->batch, layer->n, layer->out_h*layer->out_w);
        //if(index == 0) return;
        scale_bias(layer->output, layer->scales, layer->batch, layer->n, layer->out_h*layer->out_w);
    }
}

void activation_prelu(const convolutional_layer *layer, int test){
    if(0 == test){    // 0: train, 1: valid
        memcpy(layer->bottom_data, layer->output, layer->batch * layer->out_h * layer->out_w * layer->n * sizeof(float));
    }
    int count = layer->batch * layer->out_h * layer->out_w * layer->n;
    int dim = layer->out_h * layer->out_w;
    for (int i = 0; i < count; ++i) {
        int c = (i / dim) % layer->n;
        layer->output[i] = fmaxf(layer->output[i], 0.0F) + layer->slope[c] * fminf(layer->output[i], 0.0F);
      }
}

void forward_convolutional_layer(const convolutional_layer *layer, float *in, float *workspace, int test, int index)
{
    int m = layer->n;
    int n = layer->out_h * layer->out_w;
    int k = layer->size*layer->size*layer->c;
    //memset(layer->output, 0, layer->batch * m*n*sizeof(float));
    //double start = what_time_is_it_now();
    for(int i = 0; i < layer->batch; ++i){
        float *a = layer->weights;
        float *b = workspace;
        float *c = layer->output + i * m * n;
        if (layer->size == 1 && layer->stride == 1){
            b = in + i * layer->w * layer->h * layer->c;
        } else {
            memset(workspace, 0, n*k*sizeof(float));
            #ifndef USE_LINUX
            im2col_cpu(in + i * layer->w * layer->h * layer->c,
              layer->c,  layer->h,  layer->w,  layer->size,  layer->stride, layer->pad, b);
            #else
            im2col_cpu_thread(in + i * layer->w * layer->h * layer->c,
                              layer->c,  layer->h,  layer->w,  layer->size,  layer->stride, layer->pad, b, layer->out_h * layer->out_w);
            #endif
            //printf("im2col_cpu_thread: %d %f\n", index, what_time_is_it_now() - start);
        }
#if defined QML || defined INTEL_MKL || defined OPENBLAS_ARM || defined ARM_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 0, c, n);
        //printf("cblas_sgemm: %d %f, %d %d %d\n", index, what_time_is_it_now() - start, m, n, k);
#else
        gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
#endif
    }
    //if(index == 23) return;

    if(layer->batch_normalize){
        forward_conv_batchnorm_layer(layer, test, index);
    }
    //if(index == 0) return;
    for(int b = 0; b < layer->batch; ++b){
        for(int i = 0; i < layer->n; ++i){
            for(int j = 0; j < n; ++j){
                layer->output[(b*layer->n + i)*n + j] += layer->biases[i];
            }
        }
    }
    //if(index == 0) return;

    if(layer->activation == PRELU){
        activation_prelu(layer, test);
    } else if (layer->activation == LINEAR) {
    } else {
        for(int i = 0; i < layer->batch * m*n; ++i) layer->output[i] = activate(layer->output[i], layer->activation);
    }

    /*
    int count = 0;
    for(int i = 0; i < m*n && count < 10; i++){
        if(layer->output[i] > 0.001){
            printf("%d %f %f %f %f\n", i, layer->output[i], in[i], layer->weights[i], layer->slope[i]);
            count += 1;
        }
    }
    for(int i = 0; i < layer->n; i++) printf("%d %f %f %f %f\n", i, layer->rolling_mean[i], layer->rolling_variance[i], layer->scales[i], layer->biases[i]);
    exit(-1);
    float max = -FLT_MAX, min = FLT_MAX;
    for(int i = 0; i < layer->batch * m*n; ++i){
    	if(layer->output[i] > max) max = layer->output[i];
    	if(layer->output[i] < min) min = layer->output[i];
    }
    printf("forward_convolutional_layer max: %f, min: %f\n", max, min);*/
}

void backward_conv_batchnorm_layer(const convolutional_layer *layer, int test)
{
    if(0 != test){    // 0: train, 1: valid
        fprintf(stderr, "backward_conv_batchnorm_layer: use no used!\n");
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
        backward_conv_batchnorm_layer(layer, test);
    }
    for(int j = 0; j < layer->batch; ++j){
        int m = layer->n;
        int n = layer->size*layer->size*layer->c;
        int k = layer->out_w * layer->out_h;
        float *a = layer->delta + j * m * k;
        float *b = workspace;
        float *c = layer->weight_updates;
        float *im  = input + j*layer->c*layer->h*layer->w;
        if(layer->size == 1 && layer->stride == 1){
            b = im;
        } else {
            memset(workspace, 0, n*k*sizeof(float));
            im2col_cpu(im, layer->c, layer->h, layer->w, layer->size, layer->stride, layer->pad, b);
        }
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if (delta) {  // not first layer
            //memset(delta + j * layer->h * layer->w * layer->c, 0, layer->h * layer->w * layer->c * sizeof(float));
            m = layer->size*layer->size*layer->c;
            n = layer->out_w * layer->out_h;
            k = layer->n;
            a = layer->weights;
            b = layer->delta + j * n * k;
            c = workspace;
            if (layer->size == 1 && layer->stride == 1) {
                c = delta + j * layer->h * layer->w * layer->c;
            } else {
                memset(workspace, 0, m*n*sizeof(float));
            }
            gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);
            if (layer->size != 1) {
                col2im_cpu(workspace, layer->c, layer->h, layer->w, layer->size, layer->stride, layer->pad,
                           delta + j * layer->h * layer->w * layer->c);
            }
        }
    }
}

void update_convolutional_layer(const convolutional_layer *layer, float learning_rate, float momentum, float decay)
{
    int batch = layer->subdivisions * layer->batch;
    if(layer->batch_normalize){
        for(int i = 0; i < layer->n; i++){
            layer->scales[i] += learning_rate / batch * layer->scale_updates[i];
            layer->scale_updates[i] *= momentum;
        }
    }

    if(layer->activation == PRELU){
        for(int i = 0; i < layer->n; i++){
            layer->slope[i] += learning_rate / batch * layer->slope_updates[i];
            layer->slope_updates[i] *= momentum;
        }
    }

    for(int i = 0; i < layer->n; i++){
        layer->bias_updates[i] += -decay * layer->bias_decay_mult * batch * layer->biases[i];
        layer->biases[i] += learning_rate * layer->bias_mult / batch * layer->bias_updates[i];
        layer->bias_updates[i] *= momentum;
    }

    int size = layer->size*layer->size*layer->c*layer->n;
    for(int i = 0; i < size; i++){
        layer->weight_updates[i] += -decay * layer->lr_decay_mult * batch * layer->weights[i];
        layer->weights[i] += learning_rate * layer->lr_mult / batch * layer->weight_updates[i];
        layer->weight_updates[i] *= momentum;
    }
}

#ifdef USE_LINUX
#define HANDLE_THREAD_NUM 3
pthread_t handle_thread_id[HANDLE_THREAD_NUM];
typedef struct {
    int c_start, c_end, height_col, width_col, height, width, ksize, stride, pad, n_tile;
    float *data_im, *data_col;
} im2col_arg_struct;

void *im2col_thread(void *input)
{
    im2col_arg_struct *args = (im2col_arg_struct *)input;
    int c_start = args->c_start;
    int c_end = args->c_end;
    //printf("im2col_thread %d %d\n", c_start, c_end);
    int height_col = args->height_col;
    int width_col = args->width_col;
    int height = args->height;
    int width = args->width;
    int ksize = args->ksize;
    int stride = args->stride;
    int pad = args->pad;
    int n_tile = args->n_tile;
    float *data_im = args->data_im;
    float *data_col = args->data_col;
    for(int c = c_start; c < c_end; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for(int h = 0; h < height_col; ++h) {
            int row = h_offset + h * stride - pad;
            for(int w = 0; w < width_col; ++w) {
                int col = w_offset + w * stride - pad;
                int col_index = c * n_tile + h * width_col + w;
                //if(row >= 0 && col >= 0 && row < height && col < width) data_col[col_index] = data_im[col + width*(row + height*c_im)];
                //else data_col[col_index] = 0;
                if(row < 0 || col < 0 || row >= height || col >= width) data_col[col_index] = 0;
                else data_col[col_index] = data_im[col + width*(row + height*c_im)];
            }
        }
    }
    return 0;
}

void im2col_cpu_thread(float* data_im, int channels,  int height,  int width, int ksize,  int stride, int pad, float* data_col, int n_tile)
{
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    //printf("im2col_cpu_thread %d\n", channels_col);
    im2col_arg_struct args[HANDLE_THREAD_NUM];
    int row_index = 0;
    int element = (channels_col + HANDLE_THREAD_NUM - 1) / HANDLE_THREAD_NUM;
    for(int i = 0; i < HANDLE_THREAD_NUM; i++){
        args[i].c_start = row_index;
        args[i].c_end = (row_index + element > channels_col) ? channels_col : row_index + element;
        row_index += element;
        args[i].height_col = height_col;
        args[i].width_col = width_col;
        args[i].height = height;
        args[i].width = width;
        args[i].ksize = ksize;
        args[i].stride = stride;
        args[i].pad = pad;
        args[i].data_im = data_im;
        args[i].data_col = data_col;
        args[i].n_tile = n_tile;
        pthread_create(handle_thread_id + i, NULL, im2col_thread, &args[i]);
    }
    for(int i = 0; i < HANDLE_THREAD_NUM; i++){
        pthread_join(handle_thread_id[i], NULL);
    }
}
#endif

#ifdef OPENCL
void im2col_cl(cl_mem data_im, int offset, int channels,  int height,  int width,
               int ksize,  int stride,  int pad, cl_mem data_col, int width_tile)
{
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    //printf("%d %d %d %d %d %d %d\n", channels,  height_col, width_col, ksize,  stride,  pad, width_tile);
    cl_kernel kernel = get_kernel_by_name("im2col_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(data_im), (void*) &data_im);
    cl.error = clSetKernelArg(kernel, i++, sizeof(offset), (void*) &offset);
    cl.error = clSetKernelArg(kernel, i++, sizeof(height), (void*) &height);
    cl.error = clSetKernelArg(kernel, i++, sizeof(width), (void*) &width);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ksize), (void*) &ksize);
    cl.error = clSetKernelArg(kernel, i++, sizeof(pad), (void*) &pad);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(height_col), (void*) &height_col);
    cl.error = clSetKernelArg(kernel, i++, sizeof(width_col), (void*) &width_col);
    cl.error = clSetKernelArg(kernel, i++, sizeof(data_col), (void*) &data_col);
    cl.error = clSetKernelArg(kernel, i++, sizeof(width_tile), (void*) &width_tile);
    check_error(cl);
    size_t global_size = channels*height_col*width_col;
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, &global_size, 0, 0, 0, 0);
    check_error(cl);
}

void forward_conv_batchnorm_layer_cl(const convolutional_layer *layer, int test, int index)
{
    if(0 == test){    // 0: train, 1: valid
        copy_cl(layer->batch * layer->out_h * layer->out_w * layer->n, layer->output_cl, 1, layer->x_cl, 1);
        fast_mean_cl(layer->output_cl, layer->batch, layer->n, layer->out_h*layer->out_w, layer->mean_cl);
        fast_variance_cl(layer->output_cl, layer->mean_cl, layer->batch, layer->n, layer->out_h*layer->out_w,
                          layer->variance_cl);

        scal_cl(layer->n, .99, layer->rolling_mean_cl, 1);
        axpy_cl(layer->n, .01, layer->mean_cl, 1, layer->rolling_mean_cl, 1);
        scal_cl(layer->n, .99, layer->rolling_variance_cl, 1);
        axpy_cl(layer->n, .01, layer->variance_cl, 1, layer->rolling_variance_cl, 1);

        normalize_cl(layer->output_cl, layer->mean_cl, layer->variance_cl, layer->batch, layer->n,
                      layer->out_h*layer->out_w);
        copy_cl(layer->batch * layer->out_h * layer->out_w * layer->n, layer->output_cl, 1, layer->x_norm_cl, 1);
        scale_bias_cl(layer->output_cl, layer->scales_cl, layer->batch, layer->n, layer->out_h*layer->out_w);
        add_bias_cl(layer->batch, layer->out_h * layer->out_w, layer->n, layer->biases_cl, layer->output_cl);
    } else {
        normalize_cl(layer->output_cl, layer->rolling_mean_cl, layer->rolling_variance_cl,
                     layer->batch, layer->n, layer->out_h*layer->out_w);
        //if(index == 0) return;
        scale_bias_cl(layer->output_cl, layer->scales_cl, layer->batch, layer->n, layer->out_h*layer->out_w);
        add_bias_cl(layer->batch, layer->out_h * layer->out_w, layer->n, layer->biases_cl, layer->output_cl);
    }
}

void forward_convolutional_layer_cl(const convolutional_layer *layer, cl_mem in, cl_mem workspace, int test, int index, int workspace_size)
{
    int m = layer->n;
    int n = layer->out_h * layer->out_w;
    int k = layer->size*layer->size*layer->c;
    int tile_width = 8;
    double start = what_time_is_it_now();
    int n_tile = ((layer->out_h * layer->out_w + tile_width - 1) / tile_width) * tile_width;
    //int k_tile = ((layer->size*layer->size*layer->c + tile_width - 1) / tile_width) * tile_width;
    cl_mem a = layer->weights_cl;
    cl_mem b = workspace;
    cl_mem c = layer->output_cl;
    /*
    cl_float *in_cpu = (cl_float *)(clEnqueueMapBuffer(cl.queue, in, CL_TRUE, CL_MAP_READ, 0,
                                                       layer->batch * layer->c * layer->h * layer->w * sizeof(float),
                                                       0, NULL, NULL, &cl.error));
    check_error(cl);
    cl_float *workspace_cpu = (cl_float *)(clEnqueueMapBuffer(cl.queue, workspace, CL_TRUE, CL_MAP_WRITE, 0,
                                                              workspace_size, 0, NULL, NULL, &cl.error));
    check_error(cl);*/

    for(int i = 0; i < layer->batch; ++i){
        if (layer->size == 1 && layer->stride == 1){
            b = in;
            if(m % 8 == 0 && n % 8 == 0){
                gemm_fast_cl(0,0,m,n,k,1,a,0,k,b,0,n,0,c,i*n*m,n,n);
            } else {
                int m_tile = ((layer->n + tile_width - 1) / tile_width) * tile_width;
                //gemm_cl(0,0,m,n,k,1,a,0,k,b, i * layer->w * layer->h * layer->c, n,0,c,i*n*m,n);
                //gemm_native(0,0,m,n,k,1,a,0,k,b, i * layer->w * layer->h * layer->c, n,0,c,i*n*m,n);
                gemm_fast_direct_cl(0,0,m,n,k,1,a,0,k,b,0,n,0,c,i*n*m,n, m_tile);
                //printf("layer->size == 1 && layer->stride == 1 gemm_cl not implement %d %d %d\n", m,n,k);
                //exit(-1);
            }
        } else {
            //cl_memset_array(workspace, n_tile * k_tile);
            //printf("%d %d %d,   %d %d %d,  %d %d %d\n", layer->c,  layer->h,  layer->w,  layer->size,  layer->stride, layer->pad, m, n, k);
            im2col_cl(in, i*layer->c*layer->h*layer->w, layer->c,  layer->h,  layer->w,  layer->size,  layer->stride, layer->pad, b, n_tile);
            //printf("gemm im2col_cl: %d %f\n", index, what_time_is_it_now() - start);
            //im2col_cpu_thread(in_cpu + i*layer->c*layer->h*layer->w, layer->c,  layer->h,  layer->w,
            //                  layer->size,  layer->stride, layer->pad, workspace_cpu, n_tile);
            //printf("gemm im2col_cpu: %d %f\n", index, what_time_is_it_now() - start);
            gemm_fast_cl(0,0,m,n,k,1,a,0,k,b,0,n,0,c,i*n*m,n, n_tile);
        }
    }
    //cl.error = clEnqueueUnmapMemObject(cl.queue, in, in_cpu, 0, NULL, NULL);
    //check_error(cl);
    //cl.error = clEnqueueUnmapMemObject(cl.queue, workspace, workspace_cpu, 0, NULL, NULL);
    //check_error(cl);


    //printf("gemm: %d %f\n", index, what_time_is_it_now() - start);
    //if(index == 23) return;
    //cl_print_array(layer->output_cl, 1, "conv output: ", index);
    if (layer->batch_normalize) {
        forward_conv_batchnorm_layer_cl(layer, test, index);
    } else {
        add_bias_cl(layer->batch, layer->out_h * layer->out_w, layer->n, layer->biases_cl, layer->output_cl);
    }
    //cl_print_array(layer->output_cl, 1, "conv output: ", index);
    //if(index == 0) return;

    if(layer->activation == PRELU){
        if(0 == layer->test){    // 0: train, 1: valid
            copy_cl(layer->batch * layer->out_h * layer->out_w * layer->n, layer->output_cl, 1, layer->bottom_data_cl, 1);
        }
        int dim = layer->out_h * layer->out_w;
        activate_prelu_array_cl(layer->output_cl, layer->slope_cl, layer->batch, layer->n, dim);
    } else if (layer->activation == LINEAR) {
    } else {
        activate_array_cl(layer->output_cl, layer->batch * layer->out_h * layer->out_w * layer->n, layer->activation);
    }
}

void pull_convolutional_layer_cl(const convolutional_layer *layer)
{
    int size = layer->size*layer->size*layer->c*layer->n;
    cl_read_array(layer->weights_cl, layer->weights, size);
    cl_read_array(layer->biases_cl, layer->biases, layer->n);
    //cl_read_array(layer->weight_updates_cl, layer->weight_updates, size);
    //cl_read_array(layer->bias_updates_cl, layer->bias_updates, layer->n);
    if (layer->batch_normalize){
        cl_read_array(layer->scales_cl, layer->scales, layer->n);
        cl_read_array(layer->rolling_mean_cl, layer->rolling_mean, layer->n);
        cl_read_array(layer->rolling_variance_cl, layer->rolling_variance, layer->n);
    }
    if(layer->activation == PRELU){
        cl_read_array(layer->slope_cl, layer->slope, layer->n);
    }
}

void push_convolutional_layer_cl(convolutional_layer *layer)
{
    int size = layer->size*layer->size*layer->c*layer->n;
    //cl_write_array(layer->weights_cl, layer->weights, size);
    clReleaseMemObject(layer->weights_cl);
    layer->weights_cl = cl_make_weights(layer->n, layer->size*layer->size*layer->c, layer->weights);
    cl_write_array(layer->biases_cl, layer->biases, layer->n);
    //cl_write_array(layer->weight_updates_cl, layer->weight_updates, size);
    //cl_write_array(layer->bias_updates_cl, layer->bias_updates, layer->n);
    if (layer->batch_normalize){
        cl_write_array(layer->scales_cl, layer->scales, layer->n);
        //cl_write_array(layer->mean_cl, layer->mean, layer->n);
        //cl_write_array(layer->variance_cl, layer->variance, layer->n);
        cl_write_array(layer->rolling_mean_cl, layer->rolling_mean, layer->n);
        cl_write_array(layer->rolling_variance_cl, layer->rolling_variance, layer->n);
    }
    if(layer->activation == PRELU){
        cl_write_array(layer->slope_cl, layer->slope, layer->n);
    }
}
#endif

void free_convolutional_layer(void *input)
{
    convolutional_layer *layer = (convolutional_layer *)input;
    if(layer->weights) free_ptr((void *)&(layer->weights));
    if(layer->weight_updates) free_ptr((void *)&(layer->weight_updates));
    if(layer->biases) free_ptr((void *)&(layer->biases));
    if(layer->bias_updates) free_ptr((void *)&(layer->bias_updates));
    if(layer->output) free_ptr((void *)&(layer->output));
    if(layer->delta) free_ptr((void *)&(layer->delta));
    if(layer->scales) free_ptr((void *)&(layer->scales));
    if(layer->scale_updates) free_ptr((void *)&(layer->scale_updates));
    if(layer->mean) free_ptr((void *)&(layer->mean));
    if(layer->mean_delta) free_ptr((void *)&(layer->mean_delta));
    if(layer->variance) free_ptr((void *)&(layer->variance));
    if(layer->variance_delta) free_ptr((void *)&(layer->variance_delta));
    if(layer->rolling_mean) free_ptr((void *)&(layer->rolling_mean));
    if(layer->rolling_variance) free_ptr((void *)&(layer->rolling_variance));
    if(layer->x) free_ptr((void *)&(layer->x));
    if(layer->x_norm) free_ptr((void *)&(layer->x_norm));
    if(layer->bottom_data) free_ptr((void *)&(layer->bottom_data));
    if(layer->slope) free_ptr((void *)&(layer->slope));
    if(layer->slope_updates) free_ptr((void *)&(layer->slope_updates));
#ifdef GPU
    if(layer->weights_gpu) cuda_free(layer->weights_gpu);
    if(layer->weight_updates_gpu) cuda_free(layer->weight_updates_gpu);
    if(layer->biases_gpu) cuda_free(layer->biases_gpu);
    if(layer->bias_updates_gpu) cuda_free(layer->bias_updates_gpu);
    if(layer->output_gpu) cuda_free(layer->output_gpu);
    if(layer->delta_gpu) cuda_free(layer->delta_gpu);
    if(layer->scales_gpu) cuda_free(layer->scales_gpu);
    if(layer->scale_updates_gpu) cuda_free(layer->scale_updates_gpu);
    if(layer->mean_gpu) cuda_free(layer->mean_gpu);
    if(layer->mean_delta_gpu) cuda_free(layer->mean_delta_gpu);
    if(layer->variance_gpu) cuda_free(layer->variance_gpu);
    if(layer->variance_delta_gpu) cuda_free(layer->variance_delta_gpu);
    if(layer->rolling_mean_gpu) cuda_free(layer->rolling_mean_gpu);
    if(layer->rolling_variance_gpu) cuda_free(layer->rolling_variance_gpu);
    if(layer->x_gpu) cuda_free(layer->x_gpu);
    if(layer->x_norm_gpu) cuda_free(layer->x_norm_gpu);
    if(layer->bottom_data_gpu) cuda_free(layer->bottom_data_gpu);
    if(layer->slope_gpu) cuda_free(layer->slope_gpu);
    if(layer->slope_updates_gpu) cuda_free(layer->slope_updates_gpu);
#ifdef CUDNN
    cudnnDestroyTensorDescriptor(layer->normTensorDesc);
    cudnnDestroyTensorDescriptor(layer->srcTensorDesc);
    cudnnDestroyTensorDescriptor(layer->dstTensorDesc);
    cudnnDestroyTensorDescriptor(layer->dsrcTensorDesc);
    cudnnDestroyTensorDescriptor(layer->ddstTensorDesc);
    cudnnDestroyFilterDescriptor(layer->weightDesc);
    cudnnDestroyFilterDescriptor(layer->dweightDesc);
    cudnnDestroyConvolutionDescriptor(layer->convDesc);
#endif
#elif defined(OPENCL)
    if(layer->weights_cl) clReleaseMemObject(layer->weights_cl);
    if(layer->weight_updates_cl) clReleaseMemObject(layer->weight_updates_cl);
    if(layer->biases_cl) clReleaseMemObject(layer->biases_cl);
    if(layer->bias_updates_cl) clReleaseMemObject(layer->bias_updates_cl);
    if(layer->output_cl) clReleaseMemObject(layer->output_cl);
    if(layer->delta_cl) clReleaseMemObject(layer->delta_cl);
    if(layer->scales_cl) clReleaseMemObject(layer->scales_cl);
    if(layer->scale_updates_cl) clReleaseMemObject(layer->scale_updates_cl);
    if(layer->mean_cl) clReleaseMemObject(layer->mean_cl);
    if(layer->mean_delta_cl) clReleaseMemObject(layer->mean_delta_cl);
    if(layer->variance_cl) clReleaseMemObject(layer->variance_cl);
    if(layer->variance_delta_cl) clReleaseMemObject(layer->variance_delta_cl);
    if(layer->rolling_mean_cl) clReleaseMemObject(layer->rolling_mean_cl);
    if(layer->rolling_variance_cl) clReleaseMemObject(layer->rolling_variance_cl);
    if(layer->x_cl) clReleaseMemObject(layer->x_cl);
    if(layer->x_norm_cl) clReleaseMemObject(layer->x_norm_cl);
    if(layer->bottom_data_cl) clReleaseMemObject(layer->bottom_data_cl);
    if(layer->slope_cl) clReleaseMemObject(layer->slope_cl);
    if(layer->slope_updates_cl) clReleaseMemObject(layer->slope_updates_cl);
#endif
    free_ptr((void *)&layer);
}
