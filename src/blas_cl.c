#include "blas.h"
#include <assert.h>

#ifdef OPENCL

void array_add_cl(cl_mem A, cl_mem B, cl_mem C, int n)
{
    cl_kernel kernel = get_kernel_by_name("array_add_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(A), (void*)&A);
    cl.error = clSetKernelArg(kernel, i++, sizeof(B), (void*)&B);
    cl.error = clSetKernelArg(kernel, i++, sizeof(C), (void*)&C);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*)&n);
    check_error(cl);
    const size_t global_size[] = {(n + 8 - 1) / 8};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void axpy_cl(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY) {
    cl_kernel kernel = get_kernel_by_name("axpy_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*)&N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*)&ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*)&X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*)&INCX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y), (void*)&Y);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*)&INCY);
    check_error(cl);
    const size_t global_size[] = {N};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void copy_cl(int N, cl_mem X, int INCX, cl_mem Y, int INCY){
    if(INCX == 1 && INCY == 1){
        cl_copy_array(X, Y, N);
        return;
    }
    cl_kernel kernel = get_kernel_by_name("copy_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*)&N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*)&X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*)&INCX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(Y), (void*)&Y);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*)&INCY);
    check_error(cl);
    const size_t global_size[] = {N};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void scal_cl(int N, float ALPHA, cl_mem X, int INCX){
    cl_kernel kernel = get_kernel_by_name("scal_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*)&N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*)&ALPHA);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*)&X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*)&INCX);
    check_error(cl);
    const size_t global_size[] = {N};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void scale_bias_cl(cl_mem output, cl_mem biases, int batch, int n, int size) {
    cl_kernel kernel = get_kernel_by_name("scale_bias_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(output), (void*)&output);
    cl.error = clSetKernelArg(kernel, i++, sizeof(biases), (void*)&biases);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*)&batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*)&n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*)&size);
    check_error(cl);
    const size_t global_size[] = {batch * n * size};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void normalize_cl(cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial){
    cl_kernel kernel = get_kernel_by_name("normalize_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*)&x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean), (void*)&mean);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance), (void*)&variance);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*)&batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*)&filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*)&spatial);
    check_error(cl);
    const size_t global_size[] = {batch*filters*spatial};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void fast_mean_cl(cl_mem x, int batch, int filters, int spatial, cl_mem mean){
}
void fast_variance_cl(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance) {
}

void activate_prelu_array_cl(cl_mem x, cl_mem slope_cl, int batch, int filters, int spatial){
    cl_kernel kernel = get_kernel_by_name("activate_prelu_array_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*)&x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(slope_cl), (void*)&slope_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*)&filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*)&spatial);
    check_error(cl);
    const size_t global_size[] = {batch*filters*spatial};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void shortcut_cl(int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, float s1, float s2, cl_mem out){
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    cl_kernel kernel = get_kernel_by_name("shortcut_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(minw), (void*)&minw);
    cl.error = clSetKernelArg(kernel, i++, sizeof(minh), (void*)&minh);
    cl.error = clSetKernelArg(kernel, i++, sizeof(minc), (void*)&minc);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*)&stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(sample), (void*)&sample);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*)&batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(w1), (void*)&w1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(h1), (void*)&h1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c1), (void*)&c1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(add), (void*)&add);
    cl.error = clSetKernelArg(kernel, i++, sizeof(w2), (void*)&w2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(h2), (void*)&h2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c2), (void*)&c2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(s1), (void*)&s1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(s2), (void*)&s2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(out), (void*)&out);
    check_error(cl);
    const size_t global_size[] = {batch * minw * minh * minc};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);

}

void add_bias_cl(int batch, int spatial, int channel, cl_mem biases_cl, cl_mem output_cl)
{
    cl_kernel kernel = get_kernel_by_name("convolutional_bias_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(channel), (void*)&channel);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*)&spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(biases_cl), (void*)&biases_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(output_cl), (void*)&output_cl);
    check_error(cl);

    const size_t global_size[] = {channel*spatial, batch};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 2, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void l2normalize_cl(cl_mem x, int batch, int filters, int spatial, cl_mem norm_data)
{
    cl_kernel kernel = get_kernel_by_name("l2normalize_cl", 0);
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*)&x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*)&batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*)&filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*)&spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(norm_data), (void*)&norm_data);
    check_error(cl);

    const size_t global_size[] = {batch*spatial};
    cl.error = clEnqueueNDRangeKernel(cl.queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

#endif
