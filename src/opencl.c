#ifdef OPENCL
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

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
        printf("\n Error number %d", info.error);
        abort();
        exit(1);
    }
}

#define MAX_DEVICES 10

cl_info cl_init(int index)
{
    cl_info info;
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
    check_error(info);

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

    printf("=== %d OpenCL platform(s) found: ===\n", num_platforms);
    char buffer[10240];
    clGetPlatformInfo(info.platform, CL_PLATFORM_PROFILE, 10240, buffer, NULL);
    printf("  PROFILE = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_VERSION, 10240, buffer, NULL);
    printf("  VERSION = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_NAME, 10240, buffer, NULL);
    printf("  NAME = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_VENDOR, 10240, buffer, NULL);
    printf("  VENDOR = %s\n", buffer);
    clGetPlatformInfo(info.platform, CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
    printf("  EXTENSIONS = %s\n", buffer);
    check_error(info);

    if(num_devices > MAX_DEVICES) num_devices = MAX_DEVICES;
    printf("=== %d OpenCL device(s) found on platform:\n", num_devices);
    int i;
    for (i=0; i<num_devices; i++){
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        printf("  -- %d --\n", i);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_NAME = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VENDOR = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VERSION = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DRIVER_VERSION = %s\n", buffer);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_MAX_MEM_ALLOC_SIZE = %llu\n", (unsigned long long)buf_ulong);
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_MAX_WORK_GROUP_SIZE = %llu\n", (unsigned long long)buf_ulong);
        cl_uint items;
        clGetDeviceInfo( devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), 
                &items, NULL);
        printf("  DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n", (unsigned int)items);
        size_t workitem_size[10];
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, 10*sizeof(workitem_size), workitem_size, NULL);
        printf("  DEVICE_MAX_WORK_ITEM_SIZES = %u / %u / %u \n\n\n", (unsigned int)workitem_size[0], (unsigned int)workitem_size[1], (unsigned int)workitem_size[2]);
    }
    return info;
}

cl_program cl_fprog(char *filename, char *options, cl_info info)
{
    size_t srcsize;
    char src[64*1024] = {0};
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
    info.error=clBuildProgram(prog, 0, 0, options, 0, 0);
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
    } else if(strcmp(kernelname, "axpy_cl") == 0){
        if(!kernel_axpy_cl) kernel_axpy_cl = get_kernel(kernelname, options);
        return kernel_axpy_cl;
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
    if(i == 0){
        for(int j = 0; j < 10; j++) printf("%d %f %f\n", i, x[j], x_cl[j]);
    }
    axpy_cpu(n, -1, x, 1, x_cl, 1);
    float err = dot_cpu(n, x_cl, 1, x_cl, 1);
    printf("%d: %s, error: %f, ", i, s, err);
    if(err < 0.00001) printf("\n");
    else printf(" sqrtf(error / n): %f, compare array length: %d\n", sqrtf(err/n), n);
    free(x_cl);
    return err;
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


cl_mem cl_make_array(float *x, int n)
{
    if(gpu_index < 0) return 0;
    cl_mem mem;
    if(x){
        mem = clCreateBuffer(cl.context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(float)*n, x, &cl.error);
    } else {
        mem = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(float)*n, NULL, &cl.error);
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
    }
    check_error(cl);
    return mem;
}
#endif
