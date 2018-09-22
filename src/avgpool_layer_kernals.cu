#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "avgpool_layer.h"
#include "cuda.h"
}

__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    float sum = 0.0F;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        sum += input[in_index];
    }
    output[out_index] = sum / (w*h);
}

extern "C" void forward_avgpool_layer_gpu(const avgpool_layer *l, float *input)
{
    size_t n = l->c*l->batch;
    forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l->w, l->h, l->c, input, l->output_gpu);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    float temp = out_delta[out_index];
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += temp / (w*h);
    }
}

extern "C" void backward_avgpool_layer_gpu(const avgpool_layer *l, float *delta)
{
    size_t n = l->c*l->batch;
    backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l->w, l->h, l->c, delta, l->delta_gpu);
    check_error(cudaPeekAtLastError());
}

