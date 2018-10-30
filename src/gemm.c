#include "gemm.h"
#include <math.h>

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i){
        for(int k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(int j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            register float sum = 0;
            for(int k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i){
        for(int k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(int j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    #pragma omp parallel for
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            register float sum = 0;
            for(int k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


/* C = ALPHA * A * B + BETA * C,     C: M * N,      lda ldb ldc is the column of A B C */
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc)
{
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}


#ifdef GPU

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda,
        float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#elif defined(OPENCL)

void gemm_cl(int TA, int TB, int M, int N, int K, float ALPHA,
             cl_mem A_gpu, int a_off, int lda,
             cl_mem B_gpu, int b_off, int ldb,
             float BETA,
             cl_mem C_gpu, int c_off, int ldc)
{
#ifdef CLBLAS
    cl_command_queue queue = cl.queue;
    cl.error = clblasSgemm(clblasRowMajor, TA ? clblasTrans : clblasNoTrans, TB ? clblasTrans : clblasNoTrans,
                           M, N, K, ALPHA, A_gpu, a_off, lda, B_gpu, b_off, ldb, BETA, C_gpu, c_off, ldc, 1, &queue, 0, NULL, 0);
    check_error(cl);
#else
    cl_kernel      gemm_kernel = get_kernel_by_name("gemm", "-D BLOCK=" STR(TS));
    if(!TA && !TB) gemm_kernel = get_kernel_by_name("gemm_nn", "-D BLOCK=" STR(TS));
    if(!TA && TB)  gemm_kernel = get_kernel_by_name("gemm_nt", "-D BLOCK=" STR(TS));
    if(TA && !TB)  gemm_kernel = get_kernel_by_name("gemm_tn", "-D BLOCK=" STR(TS));
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(TA), (void*) &TA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(TB), (void*) &TB);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(M), (void*) &M);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(K), (void*) &K);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(A_gpu), (void*) &A_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(a_off), (void*) &a_off);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(lda), (void*) &lda);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(B_gpu), (void*) &B_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(b_off), (void*) &b_off);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ldb), (void*) &ldb);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(BETA), (void*) &BETA);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(C_gpu), (void*) &C_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(c_off), (void*) &c_off);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(ldc), (void*) &ldc);
    check_error(cl);

    const size_t global_size[] = {ceilf((float)N / TS)*TS, ceilf((float)M / TS)*TS};
    const size_t local_size[] = {TS, TS};

    cl.error = clEnqueueNDRangeKernel(queue, gemm_kernel, 2, 0, global_size, local_size, 0, 0, 0);
    check_error(cl);
#endif
}
#endif

