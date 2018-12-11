#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case PRELU:
            return "prelu";
        case ELU:
            return "elu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "prelu")==0) return PRELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case PRELU:
            printf("activate error: PRELU not implement, exit\n");
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case PRELU:
            printf("gradient error: PRELU not implement, exit\n");
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}

#ifdef OPENCL
void activate_array_cl(cl_mem x, int n, ACTIVATION a)
{
    static cl_kernel kernel = 0;
    if(0 == kernel) kernel = get_kernel_by_name("activate_array_cl", 0);
    cl_command_queue queue = cl.queue;
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(a), (void*) &a);
    check_error(cl);
    size_t gsize = n;
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &gsize, 0, 0, 0, 0);
    check_error(cl);
}

void activate_array_with_offset_cl(cl_mem x, int offset, int n, ACTIVATION a)
{
    static cl_kernel kernel = 0;
    if(0 == kernel) kernel = get_kernel_by_name("activate_array_with_offset_cl", 0);
    cl_command_queue queue = cl.queue;
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(offset), (void*) &offset);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(a), (void*) &a);
    check_error(cl);
    size_t gsize = n;
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &gsize, 0, 0, 0, 0);
    check_error(cl);
}

void gradient_array_cl(cl_mem x, int n, ACTIVATION a, cl_mem delta)
{
    static cl_kernel kernel = 0;
    if(0 == kernel) kernel = get_kernel_by_name("gradient_array_cl", 0);
    cl_command_queue queue = cl.queue;
    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(a), (void*) &a);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    check_error(cl);
    size_t gsize = n;
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &gsize, 0, 0, 0, 0);
    check_error(cl);
}
#endif
