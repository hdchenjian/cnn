LOCAL_PATH := $(call my-dir)/..

common_static_libraries := opencv_lib14 opencv_lib8 opencv_lib11 opencv_lib9 opencv_lib2 opencv_lib5 \
                        opencv_lib6 opencv_lib12 \
                        opencv_lib16 opencv_lib17 opencv_lib18 opencv_lib20 opencv_lib21  opencv_lib4 \
                        thirdparty1 thirdparty2 thirdparty3 thirdparty4 thirdparty5 \
                        thirdparty6 thirdparty7 thirdparty8 thirdparty9 thirdparty10

include $(CLEAR_VARS)
LOCAL_C_INCLUDES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/jni/include
LOCAL_STATIC_LIBRARIES := $(common_static_libraries) arm_compute arm_compute_core
LOCAL_WHOLE_STATIC_LIBRARIES := arm_compute_graph
LOCAL_CPPFLAGS := -std=c++11 -O3
LOCAL_LDLIBS += -llog -lz #-Wl,--whole-archive -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static
LOCAL_CPP_FEATURES := rtti
LOCAL_CPP_FEATURES += exceptions

LOCAL_C_INCLUDES += /media/luyao/video_send_back/install_package/ComputeLibrary
LOCAL_C_INCLUDES += /media/luyao/video_send_back/install_package/ComputeLibrary/include
LOCAL_C_INCLUDES += /media/luyao/video_send_back/install_package/ComputeLibrary/utils
LOCAL_C_INCLUDES += /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/jni/include
LOCAL_MODULE := dnn_network
LOCAL_SRC_FILES := utils/Utils.cpp utils/GraphUtils.cpp yolo_layer.cpp arcface_34.cpp
#LOCAL_SHARED_LIBRARIES := libSNPE libSYMPHONYCPU libSYMPHONYPOWER
include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_compute_graph
LOCAL_SRC_FILES := /media/luyao/video_send_back/install_package/ComputeLibrary/build/libarm_compute_graph-static.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_compute
LOCAL_SRC_FILES := /media/luyao/video_send_back/install_package/ComputeLibrary/build/libarm_compute-static.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_compute_core
LOCAL_SRC_FILES := /media/luyao/video_send_back/install_package/ComputeLibrary/build/libarm_compute_core-static.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib2
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_calib3d.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib4
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_core.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib5
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_features2d.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib6
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_flann.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib8
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_highgui.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib9
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_imgproc.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib11
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_imgcodecs.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib12
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_ml.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib14
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_objdetect.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib16
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_photo.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib17
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_stitching.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib18
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_superres.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib20
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_video.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_lib21
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a/libopencv_videostab.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty1
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/liblibtiff.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty2
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/libIlmImf.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty3
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/liblibjasper.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty4
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/liblibjpeg-turbo.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty5
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/liblibpng.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty6
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/liblibwebp.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty7
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/libcpufeatures.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty8
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/liblibprotobuf.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty9
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/libquirc.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := thirdparty10
LOCAL_SRC_FILES := /home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a/libtegra_hal.a
include $(PREBUILT_STATIC_LIBRARY)


