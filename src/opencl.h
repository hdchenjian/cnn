#ifndef OPENCL_H
#define OPENCL_H

#ifdef OPENCL
#include <CL/cl.h>
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

typedef struct {
    int initialized;
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
}cl_info;

extern cl_info cl;

void cl_setup();
void check_error(cl_info info);
void cl_read_array(cl_mem mem, float *x, int n);
void cl_write_array(cl_mem mem, float *x, int n);
cl_mem cl_make_array(float *x, int n);
cl_mem cl_make_int_array(int *x, int n);
void cl_copy_array(cl_mem src, cl_mem dst, int n);
void cl_copy_array_with_offset(cl_mem src, cl_mem dst, int n,  size_t src_offset, size_t dst_offset);
cl_mem cl_sub_array(cl_mem src, int offset, int size);
float cl_compare_array(cl_mem mem, float *x, int n, char *s, int i);
cl_kernel get_kernel_by_name(char *kernelname, char *options);
void cl_memset_array(cl_mem mem, int n);
#endif
#endif
