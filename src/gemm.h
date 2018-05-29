#ifndef GEMM_H
#define GEMM_H

/* C = ALPHA * A * B + BETA * C,     C: M * N,      lda ldb ldc is the column of A B C */
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);
#endif
#endif
