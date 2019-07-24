#/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ ../test/test_arm.cpp -I/home/luyao/git/install_package/arm_compute-v19.05-bin-android -I/home/luyao/git/install_package/arm_compute-v19.05-bin-android/include -std=c++11     -lOpenCL   -larm_compute -larm_compute_core -L/home/luyao/git/install_package/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-cl-debug  -o aa  -static-libstdc++ -pie

#/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -L/media/luyao/video_send_back/install_package/ComputeLibrary/build arcface_34.o  GraphUtils.o  Utils.o  yolo_layer.o -shared -fPIC -o libdnn_network.so
#exit

rm arcface_34 mtcc_detect

/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ \
    utils/Utils.cpp utils/GraphUtils.cpp yolo_layer.cpp arcface_34.cpp \
    -std=c++11 -Wl,--whole-archive \
    -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static \
    -llog -lz -lm -ldl \
    -lopencv_dnn -lopencv_superres -lopencv_ml -lopencv_videostab -lopencv_shape -lopencv_video -lopencv_stitching -lopencv_photo -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_flann -lopencv_core \
    -Wl,--no-whole-archive -llibwebp -lcpufeatures -lIlmImf -llibjasper -llibjpeg-turbo -llibpng -llibprotobuf -llibtiff -lquirc -ltegra_hal \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/include \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/utils \
    -I/home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/jni/include \
    -L/media/luyao/video_send_back/install_package/ComputeLibrary/build \
    -L/home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/staticlibs/arm64-v8a \
    -L/home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/3rdparty/libs/arm64-v8a \
    -o arcface_34 -static-libstdc++ -DARM_COMPUTE_CL -pie

#/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ utils/Utils.cpp utils/GraphUtils.cpp mtcc_detect.cpp -std=c++11 -Wl,--whole-archive -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -I/media/luyao/video_send_back/install_package/ComputeLibrary -I/media/luyao/video_send_back/install_package/ComputeLibrary/include -I/media/luyao/video_send_back/install_package/ComputeLibrary/utils -L/media/luyao/video_send_back/install_package/ComputeLibrary/build/ -o mtcc_detect -static-libstdc++ -pie -DARM_COMPUTE_CL


adb push arcface_34 /data/local/tmp/A/

exit






/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ \
    utils/Utils.cpp \
    -std=c++11 \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/include \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/utils \
    -I/home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/jni/include \
    -c  -DARM_COMPUTE_CL 

/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ \
    utils/GraphUtils.cpp \
    -std=c++11 \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/include \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/utils \
    -I/home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/jni/include \
    -c  -DARM_COMPUTE_CL 

/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ \
    yolo_layer.cpp \
    -std=c++11 \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/include \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/utils \
    -I/home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/jni/include \
    -c  -DARM_COMPUTE_CL 

/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ \
    arcface_34.cpp \
    -std=c++11 \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/include \
    -I/media/luyao/video_send_back/install_package/ComputeLibrary/utils \
    -I/home/luyao/git/install_package/opencv-3.4.6/build/install/sdk/native/jni/include \
    -c  -DARM_COMPUTE_CL 


/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ arcface_34.o  GraphUtils.o  Utils.o  yolo_layer.o -shared -fPIC -o libdnn_network.so


#adb push mtcc_detect /data/local/tmp/A/
# ./graph_lenet_aarch64 --data=lenet/ --target=cl --layout=NCHW --image=mnist_8.jpeg --labels=lenet_label.txt
