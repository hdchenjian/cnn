LOCAL_PATH := $(call my-dir)/..


include $(CLEAR_VARS)
LOCAL_MODULE := arm_compute_core
LOCAL_SRC_FILES := /home/luyao/git/install_package/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-neon-cl/libarm_compute_core-static.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_compute
LOCAL_SRC_FILES := /home/luyao/git/install_package/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-neon-cl/libarm_compute-static.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_compute_core_so
LOCAL_SRC_FILES := /home/luyao/git/install_package/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-cl/libarm_compute_core.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := arm_compute_so
LOCAL_SRC_FILES := /home/luyao/git/install_package/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-cl/libarm_compute.so
include $(PREBUILT_SHARED_LIBRARY)

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
LOCAL_MODULE    := test_gemm_arm
LOCAL_SRC_FILES := test/test_arm.cpp
LOCAL_C_INCLUDES += /home/luyao/git/install_package/arm_compute-v19.05-bin-android
LOCAL_C_INCLUDES += /home/luyao/git/install_package/arm_compute-v19.05-bin-android/include
#LOCAL_SHARED_LIBRARIES := arm_compute_so arm_compute_core_so #arm_opencl_so
LOCAL_SHARED_LIBRARIES :=  arm_opencl_so arm_binder arm_crypto arm_c++ arm_ui arm_cutils arm_utils arm_sync arm_hardware arm_backtrace arm_base arm_unwind arm_blzma arm_c arm_dl
LOCAL_STATIC_LIBRARIES := arm_compute arm_compute_core

LOCAL_CPPFLAGS := -fexceptions -Ofast -std=c++11 -static-libstdc++ -DARM_COMPUTE_CL  # -pie #-DQML -DOPENCL #-fopenmp
LOCAL_LDLIBS += -llog -lz -lc
LOCAL_CPP_FEATURES := rtti
LOCAL_CPP_FEATURES += exceptions

include $(BUILD_EXECUTABLE)
