#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
}

__global__ void drop_layer_kernal(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(const dropout_layer *l, float *input, int test)
{
    if (0 != test) return;  // 0: train, 1: valid
    int size = l->inputs*l->batch;
    cuda_random(l->rand_gpu, size);
    drop_layer_kernal<<<cuda_gridsize(size), BLOCK>>>(input, size, l->rand_gpu, l->probability, l->scale);
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(const dropout_layer *l, float *delta)
{
    if(!delta) return;
    int size = l->inputs*l->batch;
    drop_layer_kernal<<<cuda_gridsize(size), BLOCK>>>(delta, size, l->rand_gpu, l->probability, l->scale);
    check_error(cudaPeekAtLastError());
}

