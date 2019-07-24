LOCAL_PATH := $(call my-dir)/..



include $(CLEAR_VARS)
LOCAL_MODULE := openblas_static
LOCAL_SRC_FILES := /home/luyao/git/a/OpenBLAS/install/lib/libopenblas_cortexa57p-r0.3.7.dev.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_opencl_so
LOCAL_SRC_FILES := /home/luyao/git/cnn/jni_arm/libOpenCL.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := QML
LOCAL_SRC_FILES := /home/luyao/download/Snapdragon/qml-1.0.0/opt/Qualcomm/QML/1.0.0/arm64/lp64/lib/libQML-1.0.0.so
LOCAL_EXPORT_C_INCLUDES := /home/luyao/download/Snapdragon/qml-1.0.0/opt/Qualcomm/QML/1.0.0/arm64/lp64/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty6
LOCAL_SRC_FILES := /home/luyao/git/cnn/obj/local/arm64-v8a/libface_detect_cnn.a
include $(PREBUILT_STATIC_LIBRARY)


include $(CLEAR_VARS)
LOCAL_MODULE    := test_cnn
LOCAL_SRC_FILES := test/test.c
LOCAL_C_INCLUDES += ../src
LOCAL_C_INCLUDES += /home/luyao/git/cnn/jni
#LOCAL_C_INCLUDES += /home/luyao/git/a/OpenBLAS/install/include
#LOCAL_C_INCLUDES += /home/luyao/download/Snapdragon/opencl-sdk-1.2.2/inc
LOCAL_C_INCLUDES += /home/luyao/git/install_package/arm_compute-v19.05-bin-android/include
#LOCAL_C_INCLUDES += /home/luyao/download/opencv_arm/android-ndk-r12b/platforms/android-21/arch-arm64/usr/include
#LOCAL_C_INCLUDES += /home/luyao/download/Snapdragon/qml-1.0.0/opt/Qualcomm/QML/1.0.0/arm64/lp64/include
#LOCAL_SHARED_LIBRARIES := arm_opencl_so #QML #$(common_shared_libraries)
LOCAL_STATIC_LIBRARIES := thirdparty6 openblas_static

LOCAL_LDLIBS := -fopenmp
LOCAL_CFLAGS := -Ofast -std=c11 #-DOPENCL #-DOPENBLAS_ARM #-DQML -DOPENCL #-fopenmp

include $(BUILD_EXECUTABLE)


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
LOCAL_STATIC_LIBRARIES := openblas_static
LOCAL_CFLAGS := -Ofast -std=c11  -DOPENBLAS_ARM -DUSE_LINUX #-DQML -DINTEL_MKL -DOPENCL #-DFORWARD_GPU
# Using android logging library
# LOCAL_LDLIBS := -llog
#LOCAL_STATIC_LIBRARIES := $(common_static_libraries)
include $(BUILD_STATIC_LIBRARY)
