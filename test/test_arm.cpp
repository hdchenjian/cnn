#include "arm_compute/core/Types.h"
//#include "arm_compute/runtime/NEON/NEFunctions.h"
//#include "arm_compute/runtime/NEON/NEScheduler.h"
//#define CL_TARGET_OPENCL_VERSION 120
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
//#include "arm_compute/runtime/CL/CLTuner.h"


#include "utils/Utils.h"

#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

using namespace arm_compute;
using namespace utils;

// /media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++  aa.cpp  -I/home/luyao/git/install_package/arm_compute-v19.05-bin-android -I/home/luyao/git/install_package/arm_compute-v19.05-bin-android/include -std=c++11 -larm_compute-static -larm_compute_core-static -L/home/luyao/git/install_package/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-neon -o aa  -static-libstdc++ -pie

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

template <typename T>
void fill_random_tensor(T &tensor, float lower_bound, float upper_bound)
{
    if(tensor.info()->data_type() != arm_compute::DataType::F32){
        printf("Unsupported format\n");
        return;
    }
    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());
    map(tensor, true);
    Iterator it(&tensor, window);

    execute_window_loop(window, [&](const Coordinates & id) {
            *reinterpret_cast<float *>(it.ptr()) = 0.1;
        },
        it);
    unmap(tensor);
}

/*
void test() {
    NEGEMM armGEMM;
    Tensor matrixA, matrixB, matrixOut;
    
    int rowNum = 2000;
    int colNum = 2000;
    int k = 2000;
    const TensorShape shapeA((unsigned int)k, (unsigned int)rowNum);
    const TensorShape shapeB((unsigned int)colNum, (unsigned int)k);
    const TensorShape shapeOut((unsigned int)colNum, (unsigned int)rowNum);
    matrixA.allocator()->init(TensorInfo(shapeA, 1, DataType::F32));
    matrixB.allocator()->init(TensorInfo(shapeB, 1, DataType::F32));
    matrixOut.allocator()->init(TensorInfo(shapeOut, 1, DataType::F32));

    armGEMM.configure(&matrixA, &matrixB, nullptr, &matrixOut, 1.0f, 0.0f); // Configure the functions to call
    matrixA.allocator()->allocate(); // Now that the padding requirements are known we can allocate the images:
    matrixB.allocator()->allocate();
    matrixOut.allocator()->allocate();

    float tmpV = 0;
    float scale = 0.000001;
    double start = what_time_is_it_now();
    for(int h = 0; h < rowNum; h++) {
        for(int w = 0; w < k; w++) {
            tmpV = *reinterpret_cast<float*>( matrixA.buffer() + matrixA.info()->offset_element_in_bytes(Coordinates(w,h,0))) = (float)(w + h) * scale;
            //printf("%f,", tmpV);
        }
        //printf("\n");
    }
    printf("\n");

    tmpV = 0;
    for(int h = 0; h < k; h++) {
        for(int w = 0; w < colNum; w++) {
            tmpV = *reinterpret_cast<float*>( matrixB.buffer() + matrixB.info()->offset_element_in_bytes(Coordinates(w,h,0))) = (float)(w + h) * scale;
            //printf("%f,", tmpV);
        }
        //printf("\n");
    }
    printf("\n");

    double end;
    for(int i = 0; i < 120; i++){
        start = what_time_is_it_now();
        armGEMM.run();
        end = what_time_is_it_now();
        printf("spend: %f\n", end - start);
    }
    
    float sum = 0;
    for(int h = 0; h < rowNum; h++) {
        for(int w = 0; w < colNum; w++) {
            tmpV = *reinterpret_cast<float*>( matrixOut.buffer() + matrixOut.info()->offset_element_in_bytes(Coordinates(w,h,0)));
            sum += tmpV;
            //printf("%f,", tmpV);
        }
        //printf("\n");
    }
    printf("%f spend: %f\n", sum, end-start);
}
*/

void test_cl() {
    CLGEMM armGEMM;
    CLTensor matrixA, matrixB, matrixOut;
    
    int rowNum = 2000;
    int colNum = 2000;
    int k = 2000;
    const TensorShape shapeA((unsigned int)k, (unsigned int)rowNum);
    const TensorShape shapeB((unsigned int)colNum, (unsigned int)k);
    const TensorShape shapeOut((unsigned int)colNum, (unsigned int)rowNum);
    matrixA.allocator()->init(TensorInfo(shapeA, 1, DataType::F32));
    matrixB.allocator()->init(TensorInfo(shapeB, 1, DataType::F32));
    matrixOut.allocator()->init(TensorInfo(shapeOut, 1, DataType::F32));

    printf("CLGEMM start 01\n");
    armGEMM.configure(&matrixA, &matrixB, nullptr, &matrixOut, 1.0f, 0.0f); // Configure the functions to call
    printf("CLGEMM start 02\n");
    matrixA.allocator()->allocate(); // Now that the padding requirements are known we can allocate the images:
    matrixB.allocator()->allocate();
    matrixOut.allocator()->allocate();
    printf("CLGEMM start 1\n");
    
    float tmpV = 0;
    float scale = 0.000001;
    double start = what_time_is_it_now();
    for(int h = 0; h < rowNum; h++) {
        for(int w = 0; w < k; w++) {
            tmpV = *reinterpret_cast<float*>( matrixA.buffer() + matrixA.info()->offset_element_in_bytes(Coordinates(w,h,0))) = (float)(w + h) * scale;
            //printf("%f,", tmpV);
        }
        //printf("\n");
    }
    printf("\n");

    tmpV = 0;
    for(int h = 0; h < k; h++) {
        for(int w = 0; w < colNum; w++) {
            tmpV = *reinterpret_cast<float*>( matrixB.buffer() + matrixB.info()->offset_element_in_bytes(Coordinates(w,h,0))) = (float)(w + h) * scale;
            //printf("%f,", tmpV);
        }
        //printf("\n");
    }
    printf("\n");

    double end;
    for(int i = 0; i < 120; i++){
        start = what_time_is_it_now();
        printf("CLGEMM start 2\n");
        armGEMM.run();
        end = what_time_is_it_now();
        printf("spend: %f\n", end - start);
    }
    
    float sum = 0;
    for(int h = 0; h < rowNum; h++) {
        for(int w = 0; w < colNum; w++) {
            tmpV = *reinterpret_cast<float*>( matrixOut.buffer() + matrixOut.info()->offset_element_in_bytes(Coordinates(w,h,0)));
            sum += tmpV;
            //printf("%f,", tmpV);
        }
        //printf("\n");
    }
    printf("%f spend: %f\n", sum, end-start);
}




#define MAX_DEVICES 10

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
    cl_share_mem_bakeup share_mem_struct[512];
    cl_uint device_page_size;
    //int m_ion_device_fd;
} cl_info;


void check_error(cl_info info)
{
    //clFinish(cl.queue);
    if (info.error != CL_SUCCESS) {
        printf("\n Error number %d\n", info.error);
        abort();
        exit(-1);
    }
}

cl_info cl_init(int index)
{
    cl_info info;
    info.share_mem_index = 0;
    info.share_mem_struct;
    info.share_mem_index_max = 512;
    info.initialized = 0;
    //if(index < 0) error("Won't initialize negative gpu id\n");
    cl_uint num_platforms, num_devices;
    // Fetch the Platform and Device IDs; we only want one.
    cl_device_id devices[MAX_DEVICES];
    info.error=clGetPlatformIDs(1, &info.platform, &num_platforms);
    check_error(info);
    printf("=== %d OpenCL platform(s) found: ===\n", num_platforms);

    info.error=clGetDeviceIDs(info.platform, CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
    check_error(info);
    printf("=== %d OpenCL num_devices(s) found: ===\n", num_devices);

    index = index%num_devices;
    info.device = devices[index];

    int printf_log = 1;
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

    for (int i=0; i<num_devices; i++){
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

void test_opencl(){
    printf("start\n\n");
    cl_init(0);
    printf("end\n\n");
}

int main() {
    test_opencl();
    //test();
    test_cl();
}
