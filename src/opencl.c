#ifdef OPENCL

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#ifdef CLBLAS
#include <clBLAS.h>
#endif

#include "opencl.h"
#include "utils.h"
#include "blas.h"
//#include "activations.h"

cl_info cl = {0};
int gpu_index;

void check_error(cl_info info)
{
    clFinish(cl.queue);
    if (info.error != CL_SUCCESS) {
        printf("\n Error number %d\n", info.error);
        abort();
        exit(-1);
    }
}

#define MAX_DEVICES 10

cl_info cl_init(int index)
{
    cl_info info;
    info.share_mem_index = 0;
    //info.share_mem_struct;
    info.share_mem_index_max = 512;
    info.initialized = 0;
    if(index < 0) error("Won't initialize negative gpu id\n");
    cl_uint num_platforms, num_devices;
    // Fetch the Platform and Device IDs; we only want one.
    cl_device_id devices[MAX_DEVICES];
    info.error=clGetPlatformIDs(1, &info.platform, &num_platforms);
    check_error(info);

    info.error=clGetDeviceIDs(info.platform, CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
    check_error(info);

    index = index%num_devices;
    info.device = devices[index];
    /*
    cl_uint device_page_size;
    info.error = clGetDeviceInfo(info.device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(device_page_size), &device_page_size, NULL);
    check_error(info);
    info.device_page_size = device_page_size;
    int m_ion_device_fd = open("/dev/ion", O_RDONLY);
    if(m_ion_device_fd < 0) {
        printf("Error: failed opening /dev/ion\n");
        exit(-1);
    }
    info.m_ion_device_fd = m_ion_device_fd; */

    cl_context_properties properties[]={
        CL_CONTEXT_PLATFORM, (cl_context_properties)info.platform, 0};

    // Note that nVidia's OpenCL requires the platform property
    info.context=clCreateContext(properties, 1, &info.device, 0, 0, &info.error);
    check_error(info);

    info.queue = clCreateCommandQueue(info.context, info.device, 0, &info.error);
    check_error(info);
#ifdef CLBLAS
    info.error = clblasSetup();
    check_error(info);
#endif
    info.initialized = 1;

    int printf_log = 0;
    if(printf_log) printf("=== %d OpenCL platform(s) found: ===\n", num_platforms);
    char buffer[10240];
    clGetPlatformInfo(info.platform, CL_PLATFORM_PROFILE, 10240, buffer, NULL);
    if(printf_log) printf("  PROFILE = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_VERSION, 10240, buffer, NULL);
    if(printf_log) printf("  VERSION = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_NAME, 10240, buffer, NULL);
    if(printf_log) printf("  NAME = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_VENDOR, 10240, buffer, NULL);
    if(printf_log) printf("  VENDOR = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
    if(printf_log) printf("  EXTENSIONS = %s\n", buffer);
    check_error(info);

    if(num_devices > MAX_DEVICES) num_devices = MAX_DEVICES;
    if(printf_log) printf("=== %d OpenCL device(s) found on platform:\n", num_devices);
    int i;
    for (i=0; i<num_devices; i++){
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        if(printf_log) printf("  -- %d --\n", i);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        if(printf_log) printf("  DEVICE_NAME = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
        if(printf_log) printf("  DEVICE_VENDOR = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
        if(printf_log) printf("  DEVICE_VERSION = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
        if(printf_log) printf("  DRIVER_VERSION = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
        if(printf_log) printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
        if(printf_log) printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        if(printf_log) printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        if(printf_log) printf("  DEVICE_MAX_MEM_ALLOC_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        if(printf_log) printf("  DEVICE_MAX_WORK_GROUP_SIZE = %llu\n", (unsigned long long)buf_ulong);
        cl_uint items;
        clGetDeviceInfo( devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), 
                &items, NULL);
        if(printf_log) printf("  DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n", (unsigned int)items);
        size_t workitem_size[10];
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, 10*sizeof(workitem_size), workitem_size, NULL);
        if(printf_log) {
            printf("  DEVICE_MAX_WORK_ITEM_SIZES = %u / %u / %u \n\n\n",
                   (unsigned int)workitem_size[0], (unsigned int)workitem_size[1], (unsigned int)workitem_size[2]);
        }
    }
    return info;
}

cl_program cl_fprog(char *filename, char *options, cl_info info)
{
    size_t srcsize;
    char src[128*1024] = {0};
    FILE *fil=fopen(filename,"r");
    if(fil == 0) file_error(filename);
    srcsize=fread(src, sizeof(src), 1, fil);
    fclose(fil);
    const char *srcptr[]={src};
    // Submit the source code of the example kernel to OpenCL
    cl_program prog=clCreateProgramWithSource(info.context,1, srcptr, &srcsize, &info.error);
    check_error(info);
    char build_c[1024*64];
    // and compile it (after this we could extract the compiled version)
    info.error = clBuildProgram(prog, 0, 0, options, 0, 0);
    if ( info.error != CL_SUCCESS ) {
        fprintf(stderr, "Error Building Program: %d\n", info.error);
        clGetProgramBuildInfo( prog, info.device, CL_PROGRAM_BUILD_LOG, 1024*64, build_c, 0);
        fprintf(stderr, "Build Log for %s program:\n%s\n", filename, build_c);
    }
    check_error(info);
    return prog;
}

void cl_setup()
{
    if(!cl.initialized){
        fprintf(stderr, "Initializing OpenCL\n");
        cl = cl_init(gpu_index);
    }
}

cl_kernel get_kernel(char *kernelname, char *options)
{
    cl_program prog = cl_fprog("src/opencl.cl", options, cl);
    cl_kernel kernel=clCreateKernel(prog, kernelname, &cl.error);
    check_error(cl);
    return kernel;
}

cl_kernel get_kernel_by_name(char *kernelname, char *options)
{
    //printf("get_kernel_by_name kernelname: %s, options: %s\n", kernelname, options);
    static cl_kernel kernel_im2col_cl = 0;
    static cl_kernel kernel_convolutional_bias = 0;
    static cl_kernel kernel_gemm = 0;
    static cl_kernel kernel_gemm_nn = 0;
    static cl_kernel kernel_gemm_nt = 0;
    static cl_kernel kernel_gemm_tn = 0;

    static cl_kernel kernel_gemm_native = 0;
    static cl_kernel kernel_gemm_tile_8x4 = 0;
    static cl_kernel kernel_gemm_image = 0;
    static cl_kernel kernel_gemm_image_buf = 0;
    static cl_kernel kernel_gemm_fast = 0;
    static cl_kernel kernel_gemm_fast_image = 0;
    static cl_kernel kernel_gemm_fast_direct = 0;
    static cl_kernel kernel_gemm_with_local = 0;
    static cl_kernel kernel_gemm_with_local_image = 0;
    static cl_kernel kernel_matrix_transpose_cl = 0;
    static cl_kernel kernel_matrix_transpose_direct_cl = 0;

    static cl_kernel kernel_array_add_cl = 0;
    static cl_kernel kernel_copy_cl = 0;
    static cl_kernel kernel_axpy_cl = 0;
    static cl_kernel kernel_scal_cl = 0;
    static cl_kernel kernel_scale_bias_cl = 0;
    static cl_kernel kernel_normalize_cl = 0;
    static cl_kernel kernel_activate_prelu_array_cl = 0;
    static cl_kernel kernel_activate_array_cl = 0;
    static cl_kernel kernel_activate_array_with_offset_cl = 0;
    static cl_kernel kernel_gradient_array_cl = 0;
    static cl_kernel kernel_shortcut_cl = 0;
    static cl_kernel kernel_forward_maxpool_layer_cl = 0;
    static cl_kernel kernel_upsample_cl = 0;
    static cl_kernel kernel_l2normalize_cl = 0;
    if(strcmp(kernelname, "convolutional_bias_cl") == 0){
        if(!kernel_convolutional_bias) kernel_convolutional_bias = get_kernel(kernelname, options);
        return kernel_convolutional_bias;
    } else if(strcmp(kernelname, "im2col_cl") == 0){
        if(!kernel_im2col_cl) kernel_im2col_cl = get_kernel(kernelname, options);
        return kernel_im2col_cl;
    } else if(strcmp(kernelname, "gemm") == 0){
        if(!kernel_gemm) kernel_gemm = get_kernel(kernelname, options);
        return kernel_gemm;
    } else if(strcmp(kernelname, "gemm_nn") == 0){
        if(!kernel_gemm_nn) kernel_gemm_nn = get_kernel(kernelname, options);
        return kernel_gemm_nn;
    } else if(strcmp(kernelname, "gemm_nt") == 0){
        if(!kernel_gemm_nt) kernel_gemm_nt = get_kernel(kernelname, options);
        return kernel_gemm_nt;
    } else if(strcmp(kernelname, "gemm_tn") == 0){
        if(!kernel_gemm_tn) kernel_gemm_tn = get_kernel(kernelname, options);
        return kernel_gemm_tn;
    } else if(strcmp(kernelname, "gemm_native") == 0){
        if(!kernel_gemm_native) kernel_gemm_native = get_kernel(kernelname, options);
        return kernel_gemm_native;
    } else if(strcmp(kernelname, "gemm_image") == 0){
        if(!kernel_gemm_image) kernel_gemm_image = get_kernel(kernelname, options);
        return kernel_gemm_image;
    } else if(strcmp(kernelname, "gemm_image_buf") == 0){
        if(!kernel_gemm_image_buf) kernel_gemm_image_buf = get_kernel(kernelname, options);
        return kernel_gemm_image_buf;
    } else if(strcmp(kernelname, "gemm_fast") == 0){
        if(!kernel_gemm_fast) kernel_gemm_fast = get_kernel(kernelname, options);
        return kernel_gemm_fast;
    } else if(strcmp(kernelname, "gemm_fast_image") == 0){
        if(!kernel_gemm_fast_image) kernel_gemm_fast_image = get_kernel(kernelname, options);
        return kernel_gemm_fast_image;
    } else if(strcmp(kernelname, "gemm_fast_direct") == 0){
        if(!kernel_gemm_fast_direct) kernel_gemm_fast_direct = get_kernel(kernelname, options);
        return kernel_gemm_fast_direct;
    } else if(strcmp(kernelname, "gemm_with_local") == 0){
        if(!kernel_gemm_with_local) kernel_gemm_with_local = get_kernel(kernelname, options);
        return kernel_gemm_with_local;
    } else if(strcmp(kernelname, "gemm_with_local_image") == 0){
        if(!kernel_gemm_with_local_image) kernel_gemm_with_local_image = get_kernel(kernelname, options);
        return kernel_gemm_with_local_image;
    } else if(strcmp(kernelname, "gemm_tile_8x4") == 0){
        if(!kernel_gemm_tile_8x4) kernel_gemm_tile_8x4 = get_kernel(kernelname, options);
        return kernel_gemm_tile_8x4;
    } else if(strcmp(kernelname, "matrix_transpose_cl") == 0){
        if(!kernel_matrix_transpose_cl) kernel_matrix_transpose_cl = get_kernel(kernelname, options);
        return kernel_matrix_transpose_cl;
    } else if(strcmp(kernelname, "matrix_transpose_direct_cl") == 0){
        if(!kernel_matrix_transpose_direct_cl) kernel_matrix_transpose_direct_cl = get_kernel(kernelname, options);
        return kernel_matrix_transpose_direct_cl;
    } else if(strcmp(kernelname, "axpy_cl") == 0){
        if(!kernel_axpy_cl) kernel_axpy_cl = get_kernel(kernelname, options);
        return kernel_axpy_cl;
    } else if(strcmp(kernelname, "array_add_cl") == 0){
        if(!kernel_array_add_cl) kernel_array_add_cl = get_kernel(kernelname, options);
        return kernel_array_add_cl;
    } else if(strcmp(kernelname, "copy_cl") == 0){
        if(!kernel_copy_cl) kernel_copy_cl = get_kernel(kernelname, options);
        return kernel_copy_cl;
    } else if(strcmp(kernelname, "scal_cl") == 0){
        if(!kernel_scal_cl) kernel_scal_cl = get_kernel(kernelname, options);
        return kernel_scal_cl;
    } else if(strcmp(kernelname, "scale_bias_cl") == 0){
        if(!kernel_scale_bias_cl) kernel_scale_bias_cl = get_kernel(kernelname, options);
        return kernel_scale_bias_cl;
    } else if(strcmp(kernelname, "normalize_cl") == 0){
        if(!kernel_normalize_cl) kernel_normalize_cl = get_kernel(kernelname, options);
        return kernel_normalize_cl;
    } else if(strcmp(kernelname, "activate_prelu_array_cl") == 0){
        if(!kernel_activate_prelu_array_cl) kernel_activate_prelu_array_cl = get_kernel(kernelname, options);
        return kernel_activate_prelu_array_cl;
    } else if(strcmp(kernelname, "activate_array_cl") == 0){
        if(!kernel_activate_array_cl) kernel_activate_array_cl = get_kernel(kernelname, options);
        return kernel_activate_array_cl;
    } else if(strcmp(kernelname, "activate_array_with_offset_cl") == 0){
        if(!kernel_activate_array_with_offset_cl) kernel_activate_array_with_offset_cl = get_kernel(kernelname, options);
        return kernel_activate_array_with_offset_cl;
    } else if(strcmp(kernelname, "gradient_array_cl") == 0){
        if(!kernel_gradient_array_cl) kernel_gradient_array_cl = get_kernel(kernelname, options);
        return kernel_gradient_array_cl;
    } else if(strcmp(kernelname, "shortcut_cl") == 0){
        if(!kernel_shortcut_cl) kernel_shortcut_cl = get_kernel(kernelname, options);
        return kernel_shortcut_cl;
    } else if(strcmp(kernelname, "forward_maxpool_layer_cl") == 0){
        if(!kernel_forward_maxpool_layer_cl) kernel_forward_maxpool_layer_cl = get_kernel(kernelname, options);
        return kernel_forward_maxpool_layer_cl;
    } else if(strcmp(kernelname, "upsample_cl") == 0){
        if(!kernel_upsample_cl) kernel_upsample_cl = get_kernel(kernelname, options);
        return kernel_upsample_cl;
    } else if(strcmp(kernelname, "l2normalize_cl") == 0){
        if(!kernel_l2normalize_cl) kernel_l2normalize_cl = get_kernel(kernelname, options);
        return kernel_l2normalize_cl;
    } else {
        printf("get_kernel_by_name kernelname: %s, not found\n", kernelname);
        exit(-1);
    }
}

void cl_read_array(cl_mem mem, float *x, int n)
{
    if(gpu_index < 0) return;
    cl.error = clEnqueueReadBuffer(cl.queue, mem, CL_TRUE, 0, sizeof(float)*n,x,0,0,0);
    check_error(cl);
}

float cl_compare_array(cl_mem mem, float *x, int n, char *s, int i)
{
    float *x_cl = calloc(n, sizeof(float));
    cl_read_array(mem, x_cl, n);
    if(i == 56){
        int count = 0;
        for(int j = 0; j < n && count < 10; j++){
            if(fabsf(x[j] - x_cl[j]) > 0.00001){
                printf("diff %d %f %f\n", j, x[j], x_cl[j]);
                count += 1;
            }
        }
    }
    axpy_cpu(n, -1, x, 1, x_cl, 1);
    float err = dot_cpu(n, x_cl, 1, x_cl, 1);
    printf("%d: %s, error: %f, ", i, s, err);
    if(err < 0.00001) printf("\n");
    else printf(" sqrtf(error / n): %f, compare array length: %d\n", sqrtf(err/n), n);
    free(x_cl);
    //if(err > 0.001) exit(-1);
    return err;
}

void cl_print_array(cl_mem mem, int n, char *s, int i)
{
    float *x_cl = calloc(n, sizeof(float));
    cl_read_array(mem, x_cl, n);
    for(int j = 0; j < n; j++) printf("layer: %d %s, %d %f\n", i, s, j, x_cl[j]);
    free(x_cl);
}

void cl_write_array(cl_mem mem, float *x, int n)
{
    if(gpu_index < 0) return;
    cl.error = clEnqueueWriteBuffer(cl.queue, mem, CL_TRUE, 0,sizeof(float)*n,x,0,0,0);
    check_error(cl);
}

void cl_memset_array(cl_mem mem, int n)
{
    int value = 0;
    cl.error = clEnqueueFillBuffer(cl.queue, mem, &value, sizeof(int), 0, sizeof(float)*n, 0, 0, 0);
    check_error(cl);
}

void cl_copy_array_with_offset(cl_mem src, cl_mem dst, int n,  size_t src_offset, size_t dst_offset)
{
    cl.error = clEnqueueCopyBuffer(cl.queue, src, dst, sizeof(float)*src_offset, sizeof(float)*dst_offset, sizeof(float)*n,0,0,0);
    check_error(cl);
}

void cl_copy_array(cl_mem src, cl_mem dst, int n)
{
    cl.error = clEnqueueCopyBuffer(cl.queue, src, dst, 0, 0, sizeof(float)*n,0,0,0);
    check_error(cl);
}

cl_mem cl_sub_array(cl_mem src, int offset, int size)
{
    cl_buffer_region r;
    r.origin = offset*sizeof(float);
    r.size = size*sizeof(float);
    cl_mem sub = clCreateSubBuffer(src, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &cl.error);
    check_error(cl);
    return sub;
}


void gemm_matrix_transpose_tile_cl(cl_mem A_gpu, cl_mem B_gpu, int width, int height, int width_t)
{
    cl_kernel gemm_kernel = get_kernel_by_name("matrix_transpose_direct_cl", 0);
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(A_gpu), (void*) &A_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(B_gpu), (void*) &B_gpu);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(width), (void*) &width);
    cl.error = clSetKernelArg(gemm_kernel, i++, sizeof(width_t), (void*) &width_t);
    check_error(cl);

    const size_t global_size[] = {width, height};
    cl.error = clEnqueueNDRangeKernel(queue, gemm_kernel, 2, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

cl_mem cl_make_weights(int h, int w, float *weights)
{
    int tile_width = 8;
    int h_tile = ((h + tile_width - 1) / tile_width) * tile_width;
    //int w_tile = ((w + tile_width - 1) / tile_width) * tile_width;
    cl_mem mem = cl_make_array(0, h_tile * w);
    cl_mem weights_cl = cl_make_array(weights, h * w);
    gemm_matrix_transpose_tile_cl(weights_cl, mem, w, h, h_tile);
    clReleaseMemObject(weights_cl);
    return mem;
}

/*
cl_mem cl_make_share_array(float *x, int element_num)
{
    int mem_byte = element_num * sizeof(float);
    if(gpu_index < 0) return 0;
    struct ion_allocation_data allocation_data;
    allocation_data.len = mem_byte;
    allocation_data.align = cl.device_page_size;
    allocation_data.heap_id_mask = ION_HEAP(ION_IOMMU_HEAP_ID);
    allocation_data.flags = 0;
    if(ioctl(cl.m_ion_device_fd, ION_IOC_ALLOC, &allocation_data)) {
        printf("Error allocating ion memory: %s\n", strerror(errno));
        exit(-1);
    }

    struct ion_handle_data handle_data;
    struct ion_fd_data fd_data;
    handle_data.handle = allocation_data.handle;
    fd_data.handle = allocation_data.handle;
    if(ioctl(cl.m_ion_device_fd, ION_IOC_MAP, &fd_data)) {
        ioctl(cl.m_ion_device_fd, ION_IOC_FREE, &handle_data);
        printf("Error mapping ion memory to cpu-addressable fd: %s\n", strerror(errno));
        exit(-1);
    }

    void *host_addr = mmap(NULL, allocation_data.len, PROT_READ | PROT_WRITE, MAP_SHARED, fd_data.fd, 0);
    if (MAP_FAILED == host_addr) {
        close(fd_data.fd);
        ioctl(cl.m_ion_device_fd, ION_IOC_FREE, &handle_data);
        printf("Error: mmapping fd to pointer: %s\n", strerror(errno));
        exit(-1);
    }

    cl_mem_ion_host_ptr ion_mem;
    ion_mem.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
    ion_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    ion_mem.ion_filedesc = fd_data.fd;
    ion_mem.ion_hostptr = host_addr;

    if(cl.share_mem_index >= cl.share_mem_index_max){
        printf("Error: cl.share_mem_index exceeds\n");
        exit(-1);
    }
    cl.share_mem_struct[cl.share_mem_index].host_addr = ion_mem.ion_hostptr;
    cl.share_mem_struct[cl.share_mem_index].size = allocation_data.len;
    cl.share_mem_struct[cl.share_mem_index].fd = fd_data.fd;
    cl.share_mem_struct[cl.share_mem_index].handle_data = handle_data;
    cl.share_mem_index += 1;

    if(x) memcpy(ion_mem.ion_hostptr, x, mem_byte);
    else memset(ion_mem.ion_hostptr, 0, mem_byte);
    cl_mem mem = clCreateBuffer(cl.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, mem_byte, &ion_mem, &cl.error);
    check_error(cl);
    return mem;
}
*/

cl_mem cl_make_array(float *x, int n)
{
    if(gpu_index < 0) return 0;
    cl_mem mem;
    if(x){
        mem = clCreateBuffer(cl.context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(float)*n, x, &cl.error);
    } else {
        mem = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &cl.error);
        cl_memset_array(mem, n);
    }
    check_error(cl);
    //activate_array_ongpu(mem, n, LINEAR);
    return mem;
}

cl_mem cl_make_int_array(int *x, int n)
{
    if(gpu_index < 0) return 0;
    cl_mem mem;
    if(x){
        mem = clCreateBuffer(cl.context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(int)*n, x, &cl.error);
    } else {
        mem = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(int)*n, NULL, &cl.error);
        cl_memset_array(mem, n);
    }
    check_error(cl);
    return mem;
}
#endif
