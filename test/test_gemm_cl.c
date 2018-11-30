#include <stdio.h>
#include <stdlib.h>

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


float *make_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = rand_uniform(0, 10 / (float)rows);
        //m[i] = i * 0.001;
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
        gemm_native_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
        gemm_tile_8x4_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
        gemm_image_cl(0,0,h,w,w,1,a_cl,0,w,b_image,0,w,0,c_cl,0,w);
        gemm_image_buf_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
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
        gemm_native_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
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

    clReleaseMemObject(a_cl);
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
    int w = 1024;
    int h = 1024;
    test_gemm_cl(w, h);
    return 0;
}
