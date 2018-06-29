#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cudnn/cudnn-v5/

#./darknet detector train cfg/voc.data cfg/yolo-voc.cfg  backup/yolo-voc.backup  >> train_log_person_face
#./darknet detector train cfg/person_face.data cfg/yolo-voc-person-face.cfg    >> train_log_person_face
#./darknet classifier train cfg/cifar.data cfg/cifar.cfg
#./darknet classifier train cfg/mnist.data cfg/mnist.cfg
#./darknet classifier valid cfg/mnist.data cfg/mnist.cfg backup/mnist_final.weights
#./darknet classifier valid cfg/cifar.data cfg/cifar.cfg backup/cifar.weights
./darknet classifier train cfg/face_recognition.data cfg/densenet201.cfg
for w in backup/*.weights; do
    echo $w >> valid_log
    ./darknet classifier valid cfg/mnist.data cfg/mnist.cfg $w >> valid_log
done
