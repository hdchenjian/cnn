#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

//#include "opencl.h"
#include "utils.h"
#include "gemm.h"
#include "blas.h"

float *make_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = rand_uniform(0, 30 / (float)rows);
        //m[i] = i;
    }
    return m;
}

#ifdef OPENCL
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
void gemm_fast_image_cl(int TA, int TB, int M, int N, int K, float ALPHA,
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
        gemm_fast_cl(0,0,h,w,w,1,a_transpose_cl,0,w,b_cl,0,w,0,c_cl,0,w, h);//, h,w,w);
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
        gemm_fast_cl(0,0,h,w,w,1,a_transpose_cl,0,w,b_cl,0,w,0,c_cl,0,w, h);//, h,w,w);
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
    gemm_matrix_transpose_direct_cl(a_cl, a_transpose_cl, k, m);
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
    //cl_mem b_image = clCreateImage(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &matrix_b_format, &matrix_b_desc, b, &cl.error);
    check_error(cl);

    for(int i = 0; i < 1; i++){
        //gemm_cl(0,0,m,n,k,1,a_cl,0,n,b_cl,0,n,0,c_cl,0,n);
        //cl_memset_array(c_cl, m*n);
        //gemm_native_cl(0,0,m,n,k,1,a_cl,0,k,b_cl,0,n,0,c_cl,0,n);
        //cl_compare_array(c_cl, c, m*n, "gemm_native diff: ", 22);
        cl_memset_array(c_cl, m*n);
        gemm_fast_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_cl,0,n,0,c_cl,0,n, n);//, m,n,k);
        //gemm_with_local_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_cl,0,n,0,c_cl,0,n);
        //gemm_with_local_image_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_image,0,n,0,c_cl,0,n);
        //cl_print_array(a_transpose_cl, 16*8, "conv input: ", 1);
        //gemm_fast_direct_cl(0,0,m,n,k,1,a_transpose_cl,0,k,b_cl,0,n,0,c_cl,0,n, m);
        //gemm_image_cl(0,0,m,n,k,1,a_cl,0,k,b_image,0,n,0,c_cl,0,n);
        float diff_error = cl_compare_array(c_cl, c, m*n, "gemm diff: ", 56);
        if(diff_error > 0.0001) exit(-1);
        //return;
        cl_memset_array(c_cl, m*n);
    }
    int try_times = 10;

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        //gemm_fast_direct_cl(0,0,m,n,k,1,a_transpose_cl,0,k,b_cl,0,n,0,c_cl,0,n, m);
        //gemm_image_cl(0,0,m,n,k,1,a_cl,0,k,b_image,0,n,0,c_cl,0,n);
        //gemm_with_local_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_cl,0,n,0,c_cl,0,n);
        //gemm_with_local_image_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_image,0,n,0,c_cl,0,n);
        gemm_fast_cl(0,0,m,n,k,1,a_transpose_cl,0,n,b_cl,0,n,0,c_cl,0,n, n);//, m,n,k);
        //printf("%d\n", i);
    }
    end = what_time_is_it_now();
    cl_compare_array(c_cl, c, m*n, "gemm diff: ", 0);
    cl_read_array(c_cl, c, m * n);
    sum = 0;
    for(int i = 0; i < m * n; i++) sum += c[i];
    printf("gemm_fast_direct_cl: Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s GFLOPS: %f\n",
           m, k, k, n, sum, (end-start) / try_times, gflop / ((end-start) / try_times));
    clReleaseMemObject(a_cl);
    clReleaseMemObject(a_transpose_cl);
    clReleaseMemObject(b_cl);
    //clReleaseMemObject(b_image);
    clReleaseMemObject(c_cl);
    free(a);
    free(b);
    free(c);
}

void test_array_add_cl(int n)
{
    float *a = calloc(n, sizeof(float));
    for(int i = 0; i < n; ++i) a[i] = rand_uniform(0, 10);
    float *b = calloc(n, sizeof(float));
    for(int i = 0; i < n; ++i) b[i] = rand_uniform(0, 10);
    float *c = calloc(n, sizeof(float));
    axpy_cpu(n, 1, a, 1, c, 1);
    axpy_cpu(n, 1, b, 1, c, 1);

    cl_setup();
    cl_mem a_cl = cl_make_array(a, n);
    cl_mem b_cl = cl_make_array(b, n);
    cl_mem c_cl = cl_make_array(0, n);
    array_add_cl(a_cl, b_cl, c_cl, n);
    cl_compare_array(c_cl, c, n, "gemm_native diff : ", 56);
    return;
}

/*
void test_share_memery()
{
    int share_mem_struct_index = 0;
    cl_share_mem_bakeup share_mem_struct[256];
    int n = 10;
    int size = n * sizeof(float);
    float *in = calloc(n, sizeof(float));
    for(int i = 0; i < n; i++) in[i] = i;
    cl_setup();
    cl_uint device_page_size;
    cl_device_id m_device;
    cl_platform_id platform;
    cl_int err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_device, NULL);
    err = clGetDeviceInfo(m_device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(device_page_size), &device_page_size, NULL);
    if (err != CL_SUCCESS) {
        printf("Error with clGetDeviceInfo for page size.\n");
        exit(-1);
    }
    
    struct ion_allocation_data allocation_data;
    allocation_data.len = size;
    allocation_data.align = device_page_size;
    allocation_data.heap_id_mask = ION_HEAP(ION_IOMMU_HEAP_ID);
    allocation_data.flags = 0;
    int m_ion_device_fd;
    m_ion_device_fd = open("/dev/ion", O_RDONLY);
    if(m_ion_device_fd < 0) {
        printf("Error opening /dev/ion\n");
        exit(-1);
    }
    if(ioctl(m_ion_device_fd, ION_IOC_ALLOC, &allocation_data)) {
        printf("Error allocating ion memory: %s\n", strerror(errno));
        exit(-1);
    }

    struct ion_handle_data handle_data;
    struct ion_fd_data fd_data;
    handle_data.handle = allocation_data.handle;
    fd_data.handle = allocation_data.handle;
    if(ioctl(m_ion_device_fd, ION_IOC_MAP, &fd_data)) {
        ioctl(m_ion_device_fd, ION_IOC_FREE, &handle_data);
        printf("Error mapping ion memory to cpu-addressable fd: %s\n", strerror(errno));
        exit(-1);
    }

    void *host_addr = mmap(NULL, allocation_data.len, PROT_READ | PROT_WRITE, MAP_SHARED, fd_data.fd, 0);
    if (MAP_FAILED == host_addr) {
        close(fd_data.fd);
        ioctl(m_ion_device_fd, ION_IOC_FREE, &handle_data);
        printf("Error: mmapping fd to pointer: %s\n", strerror(errno));
        exit(-1);
    }

    cl_mem_ion_host_ptr ion_mem;
    ion_mem.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
    ion_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    ion_mem.ion_filedesc = fd_data.fd;
    ion_mem.ion_hostptr = host_addr;

    share_mem_struct[share_mem_struct_index].host_addr = ion_mem.ion_hostptr;
    share_mem_struct[share_mem_struct_index].size = allocation_data.len;
    share_mem_struct[share_mem_struct_index].fd = fd_data.fd;
    share_mem_struct[share_mem_struct_index].handle_data = handle_data;
    share_mem_struct_index += 1;

    memcpy(ion_mem.ion_hostptr, in, size);    
    cl_mem mem = clCreateBuffer(cl.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, size, &ion_mem, &cl.error);
    check_error(cl);
    float *ptr = (float *)(clEnqueueMapBuffer(cl.queue, mem, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &cl.error));
    check_error(cl);
    for(int i = 0; i < n; i++) printf("unmap cl mem: %d %f\n", i, ptr[i]);
    err = clEnqueueUnmapMemObject(cl.queue, mem, ptr, 0, NULL, NULL);
    check_error(cl);

    for(int i = 0; i < share_mem_struct_index; i++) {
        if (munmap(share_mem_struct[i].host_addr, share_mem_struct[i].size) < 0) {
            printf("Error munmap-ing ion alloc: %s\n", strerror(errno));
            exit(-1);
        }
        if(close(share_mem_struct[i].fd) < 0) {
            printf("Error closing ion_fd_data fd: %s\n", strerror(errno));
            exit(-1);
        }
        if (ioctl(m_ion_device_fd, ION_IOC_FREE, &(share_mem_struct[i].handle_data)) < 0) {
            printf("Error freeing ion alloc with ioctl: %s\n", strerror(errno));
            exit(-1);
        }            
    }
    if(close(m_ion_device_fd) < 0) {
        printf("Error closing ion device fd: %s\n", strerror(errno));
        exit(-1);
    }
    //clReleaseKernel(kernel);
    //clReleaseProgram(program);
    //clReleaseContext(m_context);
}
*/

void im2col_cpu_thread(float* data_im, int channels,  int height,  int width, int ksize,  int stride, int pad, float* data_col, int n_tile);
void im2col_cl(cl_mem data_im, int offset, int channels,  int height,  int width,
               int ksize,  int stride,  int pad, cl_mem data_col, int width_tile);

void im2col_cpu_local(float* data_im, int channels,  int height,  int width, int ksize,  int stride, int pad, float* data_col)
{
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    //printf("im2col_cpu_local %d\n", channels_col);
//#pragma omp parallel for
    for(int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for(int h = 0; h < height_col; ++h) {
            int row = h_offset + h * stride - pad;
            for(int w = 0; w < width_col; ++w) {
                int col = w_offset + w * stride - pad;
                int col_index = (c * height_col + h) * width_col + w;
                //if(row >= 0 && col >= 0 && row < height && col < width) data_col[col_index] = data_im[col + width*(row + height*c_im)];
                //else data_col[col_index] = 0;
                if(row < 0 || col < 0 || row >= height || col >= width) data_col[col_index] = 0;
                else data_col[col_index] = data_im[col + width*(row + height*c_im)];
            }
        }
    }
}

void test_im2col()
{
    int n = 208 * 208;
    int w = 416;
    int h = 416;
    int c = 32;
    int in_size = w * h * c;
    int out_size = 288 * n;
    float *in = calloc(in_size, sizeof(float));
    float *out = calloc(out_size, sizeof(float));
    for(int i = 0; i < in_size; i++) in[i] = rand_uniform(0, 10);
    for(int i = 0; i < out_size; i++) out[i] = rand_uniform(0, 10);
    cl_setup();
    cl_mem in_cl = cl_make_array(in, in_size);
    cl_mem out_cl = cl_make_array(out, out_size);
    im2col_cl(in_cl, 0, c, h, w, 3, 2, 1, out_cl, n);

    int try_times = 50;
    double start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        im2col_cl(in_cl, 0, c, h, w, 3, 2, 1, out_cl, n);
    }
    printf("im2col_cl: %f\n", (what_time_is_it_now() - start) / try_times);

    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        im2col_cpu_local(in, c, h, w, 3, 2, 1, out);
    }
    printf("im2col_cpu: %f\n", (what_time_is_it_now() - start) / try_times);
    cl_compare_array(out_cl, out, out_size, "im2col diff : ", 56);

    try_times = 100;
    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        int tile_width = 8;
        int size = 3;
        int stride = 2;
        int pad = 1;
        int out_h = (h + 2*pad - size) / stride + 1;
        int out_w = (w + 2*pad - size) / stride + 1;
        int n_tile = ((out_h * out_w + tile_width - 1) / tile_width) * tile_width;
        im2col_cpu_thread(in, c, h, w, size, stride, pad, out, n_tile);
    }
    printf("im2col_cpu_thread_local: %f\n", (what_time_is_it_now() - start) / try_times);
    cl_compare_array(out_cl, out, out_size, "im2col diff : ", 56);
}

#endif


#ifdef QML
#include <qml_cblas3.h>
void test_qml_gemm(int m, int n, int k)
{
    float *a = make_matrix(m, k);
    float *b = make_matrix(k, n);
    float *c = make_matrix(m, n);
    int try_times = 10;
    double start = what_time_is_it_now();
    printf("cblas_dgemm: %lu, %d %d %d\n", sizeof(int), m, n, k);
    for(int i = 0; i < try_times; i++){
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0F, a, k, b, n, 0.0F, c, n);
    }
    printf("cblas_dgemm: %f\n", (what_time_is_it_now() - start) / try_times);
}

#else
void test_qml_gemm(int m, int n, int k){printf("not define QML\n");}
#endif

int main(int argc, char **argv)
{
    // https://pjreddie.com/projects/mnist-in-csv/
    //load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_train.csv", "/home/luyao/git/cnn/.data/mnist/train");
    //load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_test.csv", "/home/luyao/git/cnn/.data/mnist/test");
    //test_convolutional_layer();
    //time_gemm(2000, 2000);
    //test_array_add_cl(2768896);
    //test_im2col();
    //test_share_memery();
    srand(time(0));
    int m = 1024;
    int n = 1024;
    int k = 1024;
    //test_qml_gemm(64, 43264, k);
    #ifdef OPENCL
    test_gemm_fast_direct_cl(m, n, k);
    //test_gemm_fast_direct_cl(64, 43264, 288);
    //test_gemm_fast_direct_cl(64, 43264, 288*2);
    //test_gemm_fast_direct_cl(64, 43264, 288*3);
    #endif
    //test_gemm_fast_direct_cl(64, 43264, 1024+1);
    /*
    for(int i = 7; i < 100; i++){
        for(int j = 0; j < 100; j++){
            test_gemm_fast_direct_cl(9 + i, 8 + j, 10);
            usleep(3000000);
        }
    }
    */
    //test_gemm_cl(1024, 1024);
    return 0;
}
