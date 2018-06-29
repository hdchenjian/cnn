#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <float.h>

extern "C" {
#include "convolutional_layer.h"
#include "gemm.h"
#include "blas.h"
#include "utils.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //*data_col_ptr = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void col2im_gpu_kernel(const int n, const float* data_col,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_im) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        float val = 0;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);
        // compute the start and end of the output
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int offset =
            (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}

void col2im_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, data_col, height, width, ksize, pad,
                stride, height_col,
                width_col, data_im);
}

void forward_batchnorm_layer_gpu(const convolutional_layer *layer, int test)
{
    if(0 == test){    // 0: train, 1: valid, 2: test
        copy_gpu(layer->batch * layer->out_h * layer->out_w * layer->n, layer->output_gpu, 1, layer->x_gpu, 1);
        fast_mean_gpu(layer->output_gpu, layer->batch, layer->n, layer->out_h*layer->out_w, layer->mean_gpu);
        fast_variance_gpu(layer->output_gpu, layer->mean_gpu, layer->batch, layer->n, layer->out_h*layer->out_w,
                          layer->variance_gpu);

        scal_gpu(layer->n, .99, layer->rolling_mean_gpu, 1);
        axpy_gpu(layer->n, .01, layer->mean_gpu, 1, layer->rolling_mean_gpu, 1);
        scal_gpu(layer->n, .99, layer->rolling_variance_gpu, 1);
        axpy_gpu(layer->n, .01, layer->variance_gpu, 1, layer->rolling_variance_gpu, 1);

        normalize_gpu(layer->output_gpu, layer->mean_gpu, layer->variance_gpu, layer->batch, layer->n,
                      layer->out_h*layer->out_w);
    } else {
        normalize_gpu(layer->output_gpu, layer->rolling_mean_gpu, layer->rolling_variance_gpu,
                      layer->batch, layer->n, layer->out_h*layer->out_w);
    }

}

void forward_convolutional_layer_gpu(const convolutional_layer *layer, float *in, float *workspace, int test)
{
    int m = layer->n;
    int n = layer->out_w*layer->out_h;
    int k = layer->size*layer->size*layer->c;
    for(int i = 0; i < layer->batch; ++i){
        float *a = layer->weights_gpu;
        float *b = workspace;
        float *c = layer->output_gpu + i*n*m;
        if (layer->size == 1){
            b = in + i * layer->w * layer->h * layer->c;
        } else {
            im2col_gpu(in + i * layer->w * layer->h * layer->c, layer->c, layer->h, layer->w, layer->size,
                       layer->stride, layer->pad, b);
        }
        gemm_gpu(0,0,m,n,k,1,a,k,b,n,0,c,n);
    }
    if (layer->batch_normalize) {
        forward_batchnorm_layer_gpu(layer, test);
    }
    add_bias_gpu(layer->output_gpu, layer->biases_gpu, layer->batch, layer->n, layer->out_w*layer->out_h);
    activate_array_gpu(layer->output_gpu, layer->batch * layer->out_h * layer->out_w * layer->n, layer->activation);

    /*char cuda_compare_error_string[128] = {0};
    sprintf(cuda_compare_error_string, "\n%s", "forward_convolutional_layer output");
    cuda_compare(layer->output_gpu, layer->output, n*m*layer->batch, cuda_compare_error_string);*/
}

void backward_batchnorm_layer_gpu(const convolutional_layer *layer, int test)
{
    if(0 != test){    // 0: train, 1: valid, 2: test
        fprintf(stderr, "backward_batchnorm_layer: use no used!\n");
        exit(-1);
    }
    fast_mean_delta_gpu(layer->delta_gpu, layer->variance_gpu, layer->batch, layer->n, layer->out_w*layer->out_h,
                        layer->mean_delta_gpu);
    fast_variance_delta_gpu(layer->x_gpu, layer->delta_gpu, layer->mean_gpu, layer->variance_gpu,
                            layer->batch, layer->n, layer->out_w*layer->out_h, layer->variance_delta_gpu);
    normalize_delta_gpu(layer->x_gpu, layer->mean_gpu, layer->variance_gpu, layer->mean_delta_gpu,
                        layer->variance_delta_gpu, layer->batch, layer->n, layer->out_w*layer->out_h, layer->delta_gpu);
}

void backward_convolutional_layer_gpu(const convolutional_layer *layer, float *input, float *delta,
        float *workspace, int test)
{
    gradient_array_gpu(layer->output_gpu, layer->batch * layer->out_h * layer->out_w * layer->n, layer->activation, layer->delta_gpu);
    backward_bias_gpu(layer->bias_updates_gpu, layer->delta_gpu, layer->batch, layer->n, layer->out_w*layer->out_h);
    if(layer->batch_normalize){
        backward_batchnorm_layer_gpu(layer, test);
    }

    for(int i = 0; i < layer->batch; ++i){
        int m = layer->n;
        int n = layer->size*layer->size*layer->c;
        int k = layer->out_w*layer->out_h;
        float *a = layer->delta_gpu + i*m*k;
        float *b = workspace;
        float *c = layer->weight_updates_gpu;

        float *im  = input + i*layer->c*layer->h*layer->w;
        if(layer->size == 1){
            b = im;
        } else {
            im2col_gpu(im, layer->c, layer->h, layer->w, layer->size, layer->stride, layer->pad, b);
        }
        gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if (delta) {
            fill_gpu(layer->h * layer->w * layer->c, 0, delta + i * layer->h * layer->w * layer->c, 1);
            m = layer->size*layer->size*layer->c;
            n = layer->out_w * layer->out_h;
            k = layer->n;
            a = layer->weights_gpu;
            b = layer->delta_gpu + i * n * k;
            c = workspace;
            if (layer->size == 1) {
                c = delta + i * layer->h * layer->w * layer->c;
            }
            gemm_gpu(1,0,m,n,k,1,a,m,b,n,0,c,n);
            if (layer->size != 1) {
                col2im_gpu(workspace, layer->c, layer->h, layer->w, layer->size, layer->stride,
                           layer->pad, delta + i * layer->h * layer->w * layer->c);
            }
        }
    }
}

void update_convolutional_layer_gpu(const convolutional_layer *layer, float learning_rate, float momentum, float decay)
{
    int size = layer->size*layer->size*layer->c*layer->n;
    axpy_gpu(size, -decay*layer->batch, layer->weights_gpu, 1, layer->weight_updates_gpu, 1);
    axpy_gpu(size, learning_rate/layer->batch, layer->weight_updates_gpu, 1, layer->weights_gpu, 1);
    scal_gpu(size, momentum, layer->weight_updates_gpu, 1);

    axpy_gpu(layer->n, learning_rate/layer->batch, layer->bias_updates_gpu, 1, layer->biases_gpu, 1);
    scal_gpu(layer->n, momentum, layer->bias_updates_gpu, 1);
}

void pull_convolutional_layer(const convolutional_layer *layer)
{
    cuda_pull_array(layer->weights_gpu, layer->weights, layer->size*layer->size*layer->c*layer->n);
    cuda_pull_array(layer->biases_gpu, layer->biases, layer->n);
    cuda_pull_array(layer->weight_updates_gpu, layer->weight_updates, layer->size*layer->size*layer->c*layer->n);
    cuda_pull_array(layer->bias_updates_gpu, layer->bias_updates, layer->n);
    if (layer->batch_normalize){
        cuda_pull_array(layer->rolling_mean_gpu, layer->rolling_mean, layer->n);
        cuda_pull_array(layer->rolling_variance_gpu, layer->rolling_variance, layer->n);
    }
}

void push_convolutional_layer(const convolutional_layer *layer)
{
    int size = layer->size*layer->size*layer->c*layer->n;
    cuda_push_array(layer->weights_gpu, layer->weights, size);
    cuda_push_array(layer->biases_gpu, layer->biases, layer->n);
    cuda_push_array(layer->weight_updates_gpu, layer->weight_updates, size);
    cuda_push_array(layer->bias_updates_gpu, layer->bias_updates, layer->n);
    if (layer->batch_normalize){
        cuda_push_array(layer->rolling_mean_gpu, layer->rolling_mean, layer->n);
        cuda_push_array(layer->rolling_variance_gpu, layer->rolling_variance, layer->n);
    }
}
