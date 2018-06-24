#ifndef CUDA_H
#define CUDA_H

#define BLOCK 512

#ifdef GPU

#ifdef CUDNN
#include "cudnn.h"
#endif

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"
#include "blas.h"

int cuda_get_device();
void cuda_set_device(int n);
void check_error(cudaError_t status);
cublasHandle_t blas_handle();
int *cuda_make_int_array(int *x, size_t n);
float *cuda_make_array(float *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
void cuda_free(float *x_gpu);
void cuda_free_int(int *x_gpu);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
