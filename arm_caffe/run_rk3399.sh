#/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ ../test/test_arm.cpp -I/home/luyao/git/install_package/arm_compute-v19.05-bin-android -I/home/luyao/git/install_package/arm_compute-v19.05-bin-android/include -std=c++11     -lOpenCL   -larm_compute -larm_compute_core -L/home/luyao/git/install_package/arm_compute-v19.05-bin-android/lib/android-arm64-v8a-cl-debug  -o aa  -static-libstdc++ -pie

#/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -L/media/luyao/video_send_back/install_package/ComputeLibrary/build arcface_34.o  GraphUtils.o  Utils.o  yolo_layer.o -shared -fPIC -o libdnn_network.so
#exit

rm face_ai
export PKG_CONFIG_PATH=/opt/ego/opencv/lib/pkgconfig/
export LD_LIBRARY_PATH=/opt/ego/opencv/lib/:/usr/lib/mali


g++ utils/Utils.cpp utils/GraphUtils.cpp yolo_layer.cpp face_detector.cpp \
    -std=c++11 -Wl,--whole-archive -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static \
    -Wl,--no-whole-archive \
    -I../../ComputeLibrary \
    -I../../ComputeLibrary/include \
    -I../../ComputeLibrary/utils \
    -L../../ComputeLibrary/build \
    `pkg-config --cflags --libs opencv` \
     -lpthread -ldl \
    -o face_ai -DARM_COMPUTE_CL
scp face_ai linaro@192.168.0.155:~/.bin/
#/media/luyao/video_send_back/install_package/aarch64-linux-android-ndk-r17c/bin/aarch64-linux-android-clang++ utils/Utils.cpp utils/GraphUtils.cpp mtcc_detect.cpp -std=c++11 -Wl,--whole-archive -larm_compute_graph-static -Wl,--no-whole-archive -larm_compute-static -larm_compute_core-static -I/media/luyao/video_send_back/install_package/ComputeLibrary -I/media/luyao/video_send_back/install_package/ComputeLibrary/include -I/media/luyao/video_send_back/install_package/ComputeLibrary/utils -L/media/luyao/video_send_back/install_package/ComputeLibrary/build/ -o mtcc_detect -static-libstdc++ -pie -DARM_COMPUTE_CL


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
