LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)
LOCAL_MODULE    := face_detect_cnn
LOCAL_SRC_FILES := examples/detector.c src/activations.c src/box.c src/data.c src/list.c src/opencl.c src/shortcut_layer.c src/yolo_layer.c src/avgpool_layer.c src/connected_layer.c src/dropout_layer.c src/lstm_layer.c src/option_list.c src/softmax_layer.c src/batchnorm_layer.c src/convolutional_layer.c src/gemm.c src/maxpool_layer.c src/parser.c src/tree.c src/blas.c src/cost_layer.c src/gru_layer.c src/network.c src/rnn_layer.c src/upsample_layer.c src/blas_cl.c src/cuda.c src/image.c src/normalize_layer.c src/route_layer.c src/utils.c
LOCAL_C_INCLUDES += ../src
LOCAL_C_INCLUDES += /home/luyao/git/opencl-sdk-1.2.2/inc
LOCAL_C_INCLUDES += /home/luyao/download/opencv_arm/android-ndk-r12b/sources/cxx-stl/gnu-libstdc++/4.9/include
LOCAL_C_INCLUDES += /home/luyao/download/opencv_arm/android-ndk-r12b/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/include
LOCAL_CFLAGS := -Ofast -std=c11 -DFORWARD_GPU -DOPENCL
# Using android logging library
# LOCAL_LDLIBS := -llog
#LOCAL_STATIC_LIBRARIES := $(common_static_libraries)
include $(BUILD_STATIC_LIBRARY)


include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib1
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libCB.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib2
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libgsl.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib3
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libc++.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib4
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libutils.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib5
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libui.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib6
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libsync.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib7
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libhardware.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib8
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libbacktrace.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib9
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libbase.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib10
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libunwind.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib11
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libc.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib12
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libcutils.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencl_lib
LOCAL_SRC_FILES := /home/luyao/git/face_demo/jni/libOpenCL.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty6
LOCAL_SRC_FILES := /home/luyao/git/cnn/obj/local/arm64-v8a/libface_detect_cnn.a
include $(PREBUILT_STATIC_LIBRARY)

common_static_libraries := thirdparty6
common_shared_libraries := opencl_lib opencl_lib3 opencl_lib4 opencl_lib5 opencl_lib6 opencl_lib7 opencl_lib8 \
                           opencl_lib9 opencl_lib10 opencl_lib11 opencl_lib12 \
                           opencl_lib1 opencl_lib2

include $(CLEAR_VARS)
LOCAL_MODULE    := test_gemm_cl
LOCAL_SRC_FILES := test/test_gemm_cl.c
LOCAL_C_INCLUDES += ../src
LOCAL_C_INCLUDES += /home/luyao/git/opencl-sdk-1.2.2/inc
LOCAL_C_INCLUDES += /home/luyao/download/opencv_arm/android-ndk-r12b/sources/cxx-stl/gnu-libstdc++/4.9/include
LOCAL_C_INCLUDES += /home/luyao/download/opencv_arm/android-ndk-r12b/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/include
LOCAL_SHARED_LIBRARIES := $(common_shared_libraries)
LOCAL_STATIC_LIBRARIES := $(common_static_libraries)

LOCAL_CFLAGS := -Ofast -std=c11 -DOPENCL

include $(BUILD_EXECUTABLE)

$(call import-module,android/native_app_glue)
$(call import-module,cxx-stl/gnu-libstdc++)
