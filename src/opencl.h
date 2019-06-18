#ifndef OPENCL_H
#define OPENCL_H

#ifdef OPENCL
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
//#include <CL/cl_ext_qcom.h>
//#include <linux/ion.h>
//#include "msm_ion.h"
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

typedef struct {
    void *host_addr;
    size_t size;
    int fd;
    //struct ion_handle_data handle_data;
} cl_share_mem_bakeup;

typedef struct {
    int initialized;
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    int share_mem_index;
    int share_mem_index_max;
    //cl_share_mem_bakeup share_mem_struct[512];
    cl_uint device_page_size;
    int m_ion_device_fd;
}cl_info;

extern cl_info cl;

void cl_setup();
void check_error(cl_info info);
void cl_read_array(cl_mem mem, float *x, int n);
void cl_write_array(cl_mem mem, float *x, int n);
cl_mem cl_make_share_array(float *x, int element_num);
cl_mem cl_make_array(float *x, int n);
cl_mem cl_make_weights(int h, int w, float *weights);
cl_mem cl_make_int_array(int *x, int n);
void cl_copy_array(cl_mem src, cl_mem dst, int n);
void cl_copy_array_with_offset(cl_mem src, cl_mem dst, int n,  size_t src_offset, size_t dst_offset);
cl_mem cl_sub_array(cl_mem src, int offset, int size);
float cl_compare_array(cl_mem mem, float *x, int n, char *s, int i);
void cl_print_array(cl_mem mem, int n, char *s, int i);
cl_kernel get_kernel_by_name(char *kernelname, char *options);
void cl_memset_array(cl_mem mem, int n);
#endif
#endif
