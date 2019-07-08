#ifndef GEMM_H
#define GEMM_H

#ifdef QML
#include <qml_cblas3.h>
#endif

#ifdef ARM_BLAS
#include <armpl.h>
#endif

#ifdef INTEL_MKL
#include "mkl.h"
#endif

#ifdef OPENBLAS_ARM
#include "cblas.h"
#endif

/* C = ALPHA * A * B + BETA * C,     C: M * N,      lda ldb ldc is the column of A B C */
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU

#include "cuda.h"

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);
#elif defined(OPENCL)
#include "opencl.h"
#ifdef CLBLAS
#include <clBLAS.h>
#endif

#define TS 16
#define T_WIDTH 8
#define TILE_ROW 8
#define TILE_COL 4
void gemm_cl(int TA, int TB, int M, int N, int K, float ALPHA,
             cl_mem A_gpu, int a_off, int lda,
             cl_mem B_gpu, int b_off, int ldb,
             float BETA,
             cl_mem C_gpu, int c_off, int ldc);
void gemm_fast_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                  cl_mem A_gpu, int a_off, int lda,
                  cl_mem B_gpu, int b_off, int ldb,
                  float BETA,
                  cl_mem C_gpu, int c_off, int ldc, int N_tile);//, int M_tile, int N_tile, int K_tile);
void gemm_fast_direct_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                         cl_mem A_gpu, int a_off, int lda,
                         cl_mem B_gpu, int b_off, int ldb,
                         float BETA,
                         cl_mem C_gpu, int c_off, int ldc, int M_tile);
#endif
#endif
