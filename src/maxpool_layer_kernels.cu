#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

extern "C" {
#include "maxpool_layer.h"
}

__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size,
                                             int pad, float *input, float *output, int *indexes, int test)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;  // width
    id /= w;
    int i = id % h;  // height
    id /= h;
    int k = id % c;  // channel
    id /= c;
    int b = id;      // batch

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    if(0 == test){    // 0: train, 1: valid
        indexes[out_index] = max_i;
    }
}

extern "C" void forward_maxpool_layer_gpu(const maxpool_layer *l, float *in_gpu)
{
    
    size_t n = l->out_h *l->out_w * l->c * l->batch;
    forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l->h, l->w, l->c, l->stride, l->size,
                                                              l->pad, in_gpu, l->output_gpu, l->indexes_gpu, l->test);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad,
                                              float *delta, float *prev_delta, int *indexes)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;
    int area = (size-1)/stride;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    float d = 0;
    int l, m;
    for(l = -area; l < area+1; ++l){
        for(m = -area; m < area+1; ++m){
            int out_w = (j-w_offset)/stride + m;
            int out_h = (i-h_offset)/stride + l;
            int out_index = out_w + w*(out_h + h*(k + c*b));
            int valid = (out_w >= 0 && out_w < w &&
                     out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }
    prev_delta[index] += d;
}

extern "C" void backward_maxpool_layer_gpu(const maxpool_layer *l, float *delta_gpu)
{
    //cudaError_t status = cudaMemset(delta_gpu, 0, sizeof(float) * l->h*l->w*l->c*l->batch);
    //check_error(status);
    size_t n = l->h*l->w*l->c*l->batch;
    backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l->h, l->w, l->c, l->stride, l->size,
                                                               l->pad, l->delta_gpu, delta_gpu, l->indexes_gpu);
    check_error(cudaPeekAtLastError());
}

