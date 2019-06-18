LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)
LOCAL_MODULE    := face_detect_cnn
LOCAL_SRC_FILES := examples/detector.c examples/classifier.c src/activations.c src/box.c src/data.c src/list.c src/opencl.c src/shortcut_layer.c src/yolo_layer.c src/avgpool_layer.c src/connected_layer.c src/dropout_layer.c src/lstm_layer.c src/option_list.c src/softmax_layer.c src/batchnorm_layer.c src/convolutional_layer.c src/gemm.c src/maxpool_layer.c src/parser.c src/tree.c src/blas.c src/cost_layer.c src/gru_layer.c src/network.c src/rnn_layer.c src/upsample_layer.c src/blas_cl.c src/cuda.c src/image.c src/normalize_layer.c src/route_layer.c src/utils.c
LOCAL_C_INCLUDES += ../src
LOCAL_C_INCLUDES += /home/luyao/git/a/OpenBLAS/install/include
#LOCAL_C_INCLUDES += /home/luyao/git/cnn/jni
LOCAL_C_INCLUDES += /home/luyao/git/install_package/arm_compute-v19.05-bin-android/include
#LOCAL_C_INCLUDES += /home/luyao/download/Snapdragon/opencl-sdk-1.2.2/inc
#LOCAL_C_INCLUDES += /home/luyao/download/opencv_arm/android-ndk-r12b/sources/cxx-stl/gnu-libstdc++/4.9/include
#LOCAL_C_INCLUDES += /home/luyao/download/opencv_arm/android-ndk-r12b/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/include
#LOCAL_SHARED_LIBRARIES := QML
#LOCAL_STATIC_LIBRARIES := openblas_static
LOCAL_CFLAGS := -Ofast -std=c11 -DOPENCL #-DOPENBLAS_ARM #-DQML -DINTEL_MKL -DOPENCL #-DFORWARD_GPU
# Using android logging library
# LOCAL_LDLIBS := -llog
#LOCAL_STATIC_LIBRARIES := $(common_static_libraries)
include $(BUILD_STATIC_LIBRARY)


include $(CLEAR_VARS)
LOCAL_MODULE := arm_opencl_so
LOCAL_SRC_FILES := /home/luyao/git/install_package/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-cl/libOpenCL.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_binder
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libbinder.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_crypto
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libcrypto.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_c++
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libc++.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_ui
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libui.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_cutils
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libcutils.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_utils
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libutils.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_sync
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libsync.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_hardware
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libhardware.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_backtrace
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libbacktrace.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_base
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libbase.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_unwind
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libunwind.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_blzma
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/liblzma.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_c
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libc.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_dl
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni/libdl.so
include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS)
LOCAL_MODULE := QML
LOCAL_SRC_FILES := /home/luyao/download/Snapdragon/qml-1.0.0/opt/Qualcomm/QML/1.0.0/arm64/lp64/lib/libQML-1.0.0.so
LOCAL_EXPORT_C_INCLUDES := /home/luyao/download/Snapdragon/qml-1.0.0/opt/Qualcomm/QML/1.0.0/arm64/lp64/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := openblas_static
LOCAL_SRC_FILES := /home/luyao/git/a/OpenBLAS/install/lib/libopenblas_cortexa57p-r0.3.7.dev.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty6
LOCAL_SRC_FILES := /home/luyao/git/cnn/obj/local/arm64-v8a/libface_detect_cnn.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE    := test_gemm_cl
LOCAL_SRC_FILES := test/test_gemm_cl.c
LOCAL_C_INCLUDES += ../src
LOCAL_C_INCLUDES += /home/luyao/git/cnn/jni
#LOCAL_C_INCLUDES += /home/luyao/git/a/OpenBLAS/install/include
#LOCAL_C_INCLUDES += /home/luyao/download/Snapdragon/opencl-sdk-1.2.2/inc
LOCAL_C_INCLUDES += /home/luyao/git/install_package/arm_compute-v19.05-bin-android/include
#LOCAL_C_INCLUDES += /home/luyao/download/opencv_arm/android-ndk-r12b/platforms/android-21/arch-arm64/usr/include
#LOCAL_C_INCLUDES += /home/luyao/download/Snapdragon/qml-1.0.0/opt/Qualcomm/QML/1.0.0/arm64/lp64/include
LOCAL_SHARED_LIBRARIES := arm_opencl_so arm_binder arm_crypto arm_c++ arm_ui arm_cutils arm_utils arm_sync arm_hardware arm_backtrace arm_base arm_unwind arm_blzma arm_c arm_dl #QML #$(common_shared_libraries)
LOCAL_STATIC_LIBRARIES := thirdparty6 #openblas_static

LOCAL_LDLIBS := -fopenmp
LOCAL_CFLAGS := -Ofast -std=c11 -DOPENCL #-DOPENBLAS_ARM #-DQML -DOPENCL #-fopenmp

include $(BUILD_EXECUTABLE)
