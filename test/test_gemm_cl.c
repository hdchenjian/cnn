#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "opencl.h"
#include "utils.h"
#include "gemm.h"

void gemm_native_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                    cl_mem A_gpu, int a_off, int lda,
                    cl_mem B_gpu, int b_off, int ldb,
                    float BETA,
                    cl_mem C_gpu, int c_off, int ldc);
void gemm_tile_8x4_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                  cl_mem A_gpu, int a_off, int lda,
                  cl_mem B_gpu, int b_off, int ldb,
                  float BETA,
                      cl_mem C_gpu, int c_off, int ldc);
void gemm_image_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                   cl_mem A_gpu, int a_off, int lda,
                   cl_mem B_gpu, int b_off, int ldb,
                   float BETA,
                   cl_mem C_gpu, int c_off, int ldc);
void gemm_image_buf_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                       cl_mem A_gpu, int a_off, int lda,
                       cl_mem B_gpu, int b_off, int ldb,
                       float BETA,
                       cl_mem C_gpu, int c_off, int ldc);

void gemm_fast_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                     cl_mem A_gpu, int a_off, int lda,
                     cl_mem B_gpu, int b_off, int ldb,
                     float BETA,
                     cl_mem C_gpu, int c_off, int ldc);
void gemm_fast_image_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                        cl_mem A_gpu, int a_off, int lda,
                        cl_mem B_gpu, int b_off, int ldb,
                        float BETA,
                        cl_mem C_gpu, int c_off, int ldc);
void gemm_fast_direct_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                         cl_mem A_gpu, int a_off, int lda,
                         cl_mem B_gpu, int b_off, int ldb,
                         float BETA,
                         cl_mem C_gpu, int c_off, int ldc);

void gemm_with_local_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                         cl_mem A_gpu, int a_off, int lda,
                         cl_mem B_gpu, int b_off, int ldb,
                         float BETA,
                         cl_mem C_gpu, int c_off, int ldc);
void gemm_with_local_image_cl(int TA, int TB, int M, int N, int K, float ALPHA,
                              cl_mem A_gpu, int a_off, int lda,
                              cl_mem B_gpu, int b_off, int ldb,
                              float BETA,
                              cl_mem C_gpu, int c_off, int ldc);


void gemm_matrix_transpose_cl(cl_mem A_gpu, cl_mem B_gpu, int width, int height);
void gemm_matrix_transpose_direct_cl(cl_mem A_gpu, cl_mem B_gpu, int width, int height);

float *make_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = rand_uniform(0, 30 / (float)rows);
        m[i] = 1;
    }
    return m;
}

void test_gemm_cl(int w, int h)
{
    float gflop = (float)w * h * h * 2.0F / (1000 * 1000 * 1000.0F);
    double start, end;
    cl_setup();
    float *a = make_matrix(h, w);
    float *b = make_matrix(h, w);
    float *c = make_matrix(h, w);
    start = what_time_is_it_now();
    gemm(0,0,h,w,w,1,a,w,b,w,0,c,w);
    end = what_time_is_it_now();
    float sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("Matrix Multiplication cpu %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n", h, w, h, w, sum, end-start, gflop / (end - start));
    cl_mem a_cl = cl_make_array(a, w * h);
    cl_mem a_transpose_cl = cl_make_array(a, w * h);
    gemm_matrix_transpose_cl(a_cl, a_transpose_cl, w, h);
    cl_mem b_cl = cl_make_array(b, w * h);
    cl_mem c_cl = cl_make_array(0, w * h);

    cl_image_format matrix_b_format;
    matrix_b_format.image_channel_order     = CL_RGBA;
    matrix_b_format.image_channel_data_type = CL_FLOAT;
    cl_image_desc matrix_b_desc;
    memset(&matrix_b_desc, 0, sizeof(matrix_b_desc));
    matrix_b_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    matrix_b_desc.image_width     = (w + 3) / 4;
    matrix_b_desc.image_height    = h;
    cl_mem b_image = clCreateImage(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &matrix_b_format, &matrix_b_desc, b, &cl.error);
    check_error(cl);

    for(int i = 0; i < 1; i++){
        gemm_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
        //gemm_native_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
        gemm_tile_8x4_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
        gemm_image_cl(0,0,h,w,w,1,a_cl,0,w,b_image,0,w,0,c_cl,0,w);
        gemm_image_buf_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
        gemm_native_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
        gemm_fast_cl(0,0,h,w,w,1,a_transpose_cl,0,w,b_cl,0,w,0,c_cl,0,w);
        gemm_fast_image_cl(0,0,h,w,w,1,a_transpose_cl,0,w,b_image,0,w,0,c_cl,0,w);
    }
    int try_times = 1;
    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
    }
    end = what_time_is_it_now();
    cl_read_array(c_cl, c, w * h);
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("gemm_cl: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           h, w, h, w, sum, (end-start) / try_times, gflop / ((end-start) / try_times));

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        //gemm_native_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
    }
    end = what_time_is_it_now();
    cl_read_array(c_cl, c, w * h);
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("gemm_native: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           h, w, h, w, sum, (end-start) / try_times, gflop / ((end-start) / try_times));

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_tile_8x4_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
    }
    end = what_time_is_it_now();
    cl_read_array(c_cl, c, w * h);
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("gemm_tile_8x4: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           h, w, h, w, sum, (end-start) / try_times, gflop / ((end-start) / try_times));

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_image_cl(0,0,h,w,w,1,a_cl,0,w,b_image,0,w,0,c_cl,0,w);
    }
    end = what_time_is_it_now();
    cl_read_array(c_cl, c, w * h);
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("gemm_image: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           h, w, h, w, sum, (end-start) / try_times, gflop / ((end-start) / try_times));

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_image_buf_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
    }
    end = what_time_is_it_now();
    cl_read_array(c_cl, c, w * h);
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("gemm_image_buf_cl: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           h, w, h, w, sum, (end-start) / try_times, gflop / ((end-start) / try_times));

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_fast_cl(0,0,h,w,w,1,a_transpose_cl,0,w,b_cl,0,w,0,c_cl,0,w);
    }
    end = what_time_is_it_now();
    cl_read_array(c_cl, c, w * h);
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("gemm_fast_cl: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           h, w, h, w, sum, (end-start) / try_times, gflop / ((end-start) / try_times));

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_fast_image_cl(0,0,h,w,w,1,a_transpose_cl,0,w,b_image,0,w,0,c_cl,0,w);
    }
    end = what_time_is_it_now();
    cl_read_array(c_cl, c, w * h);
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("gemm_fast_image_cl: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           h, w, h, w, sum, (end-start) / try_times, gflop / ((end-start) / try_times));

    clReleaseMemObject(a_cl);
    clReleaseMemObject(a_transpose_cl);
    clReleaseMemObject(b_cl);
    clReleaseMemObject(b_image);
    clReleaseMemObject(c_cl);
    free(a);
    free(b);
    free(c);
}

void test_gemm_fast_direct_cl(int m, int n, int k)
{
    float gflop = (float)m * n * k * 2.0F / (1000 * 1000 * 1000.0F);
    double start, end;
    cl_setup();
    float *a = make_matrix(m, k);
    float *b = make_matrix(k, n);
    float *c = make_matrix(m, n);
    start = what_time_is_it_now();
    gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
    end = what_time_is_it_now();
    float sum = 0;
    for(int i = 0; i < m * n; i++) sum += c[i];
    printf("Matrix Multiplication cpu %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n", m, k, k, n, sum, end-start, gflop / (end - start));
    cl_mem a_cl = cl_make_array(a, m * k);
    cl_mem a_transpose_cl = cl_make_array(a, m * k);
    gemm_matrix_transpose_direct_cl(a_cl, a_transpose_cl, m, k);
    cl_mem b_cl = cl_make_array(b, k * n);
    cl_mem c_cl = cl_make_array(0, m * n);

    cl_image_format matrix_b_format;
    matrix_b_format.image_channel_order     = CL_RGBA;
    matrix_b_format.image_channel_data_type = CL_FLOAT;
    cl_image_desc matrix_b_desc;
    memset(&matrix_b_desc, 0, sizeof(matrix_b_desc));
    matrix_b_desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    matrix_b_desc.image_width     = (n + 3) / 4;
    matrix_b_desc.image_height    = k;
    cl_mem b_image = clCreateImage(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &matrix_b_format, &matrix_b_desc, b, &cl.error);
    check_error(cl);

    for(int i = 0; i < 1; i++){
        //gemm_cl(0,0,m,n,k,1,a_cl,0,n,b_cl,0,n,0,c_cl,0,n);
        cl_memset_array(c_cl, m*n);
        //gemm_fast_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_cl,0,n,0,c_cl,0,n);
        //gemm_with_local_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_cl,0,n,0,c_cl,0,n);
        //gemm_with_local_image_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_image,0,n,0,c_cl,0,n);
        gemm_fast_direct_cl(0,0,m,n,k,1,a_transpose_cl,0,k,b_cl,0,n,0,c_cl,0,n);
        float diff_error = cl_compare_array(c_cl, c, m*n, "gemm diff: ", -1);
        if(diff_error > 0.0001) exit(-1);
        return;
        cl_memset_array(c_cl, m*n);
    }
    int try_times = 1;

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_fast_direct_cl(0,0,m,n,k,1,a_transpose_cl,0,k,b_cl,0,n,0,c_cl,0,n);
        //gemm_with_local_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_cl,0,n,0,c_cl,0,n);
        //gemm_with_local_image_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_image,0,n,0,c_cl,0,n);
        //gemm_fast_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_cl,0,n,0,c_cl,0,n);
    }
    end = what_time_is_it_now();
    cl_compare_array(c_cl, c, m*n, "gemm diff: ", -1);
    cl_read_array(c_cl, c, m * n);
    sum = 0;
    for(int i = 0; i < m * n; i++) sum += c[i];
    printf("gemm_fast_direct_cl: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           m, k, k, n, sum, (end-start) / try_times, gflop / ((end-start) / try_times));
    clReleaseMemObject(a_cl);
    clReleaseMemObject(a_transpose_cl);
    clReleaseMemObject(b_cl);
    clReleaseMemObject(b_image);
    clReleaseMemObject(c_cl);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv)
{
    // https://pjreddie.com/projects/mnist-in-csv/
    //load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_train.csv", "/home/luyao/git/cnn/.data/mnist/train");
    //load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_test.csv", "/home/luyao/git/cnn/.data/mnist/test");
    //test_convolutional_layer();
    //time_gemm(2000, 2000);
    srand(time(0));
    int m = 1024*4;
    int n = 1024*4;
    int k = 1024*4;
    //test_gemm_fast_direct_cl(m, n, k);
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            test_gemm_fast_direct_cl(9 + i, 8 + j, 10);
            usleep(3000000);
        }
    }
    //test_gemm_cl(m, n);
    return 0;
}
