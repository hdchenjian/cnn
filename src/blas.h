#ifndef BLAS_H
#define BLAS_H
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);

void test_blas();

void const_cpu(int N, float ALPHA, float *X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2,
                  float s1, float s2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta,
                         int batch, int filters, int spatial, float *delta);
void scale_bias(float *output, float *scales, int batch, int n, int size);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters,
                         int spatial, float *variance_delta);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int batch, int n, float *pred, int *truth_label_index, float *delta, float *error);
void softmax_x_ent_cpu(int batch, int n, float *pred, int *truth, float *delta, float *error);
void l2normalize_cpu(float *x, int batch, int filters, int spatial, float *norm_data);
void backward_l2normalize_cpu(int batch, int filters, int spatial, float *norm_data, float *output, float *delta, float *previous_delta);
void weighted_delta_cpu(int num, float *state, float *h, float *z, float *delta_state, float *delta_h, float *delta_z, float *delta);
void mult_add_into_cpu(int num, float *a, float *b, float *c);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void const_gpu(int N, float ALPHA, float *X, int INCX);
void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);

void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta,
                         int batch, int filters, int spatial, float *delta);

void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters,
                             int spatial, float *variance_delta);

void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);

void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_gpu(int batch, int n, float *pred, int *truth_label_index_gpu, float *delta, float *error);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void softmax_gpu_me(float *input, int n, int batch, float *output);
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
void softmax_x_ent_gpu(int batch, int n, float *pred, int *truth, float *delta, float *error);
void l2normalize_gpu(float *x, int batch, int filters, int spatial, float *norm_data);
void backward_l2normalize_gpu(int batch, int filters, int spatial, float *norm_data, float *output, float *delta, float *previous_delta);
void specific_margin_add_gpu(int batch, int inputs, float *input, float label_specific_margin_bias, int margin_scale,
                             int *truth_label_index_gpu);
void is_max_gpu(int batch, int inputs, float *output_gpu, int *truth_label_index_gpu, int *is_not_max);
void weight_normalize_gpu(int inputs, int outputs, float *x);
void weighted_delta_gpu(int num, float *state, float *h, float *z, float *delta_state, float *delta_h, float *delta_z, float *delta);
void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
void mult_add_into_gpu(int num, float *a, float *b, float *c);
void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);
#elif defined(OPENCL)
#include "opencl.h"
void array_add_cl(cl_mem A, cl_mem B, cl_mem C, int n);
void axpy_cl(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY);
void copy_cl(int N, cl_mem X, int INCX, cl_mem Y, int INCY);
void scal_cl(int N, float ALPHA, cl_mem X, int INCX);
void scale_bias_cl(cl_mem output, cl_mem biases, int batch, int n, int size);
void normalize_cl(cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial);
void fast_mean_cl(cl_mem x, int batch, int filters, int spatial, cl_mem mean);
void fast_variance_cl(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance);
void activate_prelu_array_cl(cl_mem x, cl_mem slope_cl, int batch, int filters, int spatial);
void shortcut_cl(int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, float s1, float s2, cl_mem out);
void add_bias_cl(int batch, int spatial, int channel, cl_mem biases_cl, cl_mem output_cl);
void l2normalize_cl(cl_mem x, int batch, int filters, int spatial, cl_mem norm_data);
#endif
#endif
