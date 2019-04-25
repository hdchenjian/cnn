#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64:/opt/intel/lib/intel64
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cudnn/cudnn-v5/
#export LD_LIBRARY_PATH=/opt/ego/cudnn-v7

#./cnn detector train cfg/voc.data cfg/yolo-voc.cfg  backup/yolo-voc.backup  >> train_log_person_face
#./cnn detector train cfg/person_face.data cfg/yolo-voc-person-face.cfg    >> train_log_person_face
#./cnn classifier train cfg/cifar.data cfg/cifar.cfg
#./cnn classifier train cfg/mnist.data cfg/mnist.cfg
#./cnn classifier valid cfg/mnist.data cfg/mnist.cfg backup/mnist_final.weights
#./cnn classifier valid cfg/cifar.data cfg/cifar.cfg backup/cifar.weights
#./cnn classifier train cfg/face_recognition.data cfg/densenet201.cfg
#./cnn classifier train cfg/mnist.data cfg/densenet201.cfg
# for w in backup/*.weights; do
#     echo $w >> valid_log
#     ./cnn classifier valid cfg/mnist.data cfg/mnist.cfg $w >> valid_log
# done

#./cnn rnn train cfg/rnn.cfg -data shakespear.txt
#./cnn rnn generate cfg/rnn.cfg backup/rnn_000699.weights -seed Chapter -len 1000

# for i in range(0,12): print "'" + str(a-i*0.005) + "'"

# for lr in '0.1' '0.05' '0.025' '0.01' '0.005' '0.001'; do
#     echo "learning_rate=" $lr `date` >> log_test
#     sed -i "s/learning_rate=.*/learning_rate=$lr/g" cfg/densenet.cfg
#     ./cnn classifier train  cfg/lfw_small.data cfg/densenet.cfg >> log_test
# done

for lr in '0.1' '0.05' '0.025' '0.01' '0.005' '0.001'; do
    echo "learning_rate=" $lr `date` >> log_test
    sed -i "s/learning_rate=.*/learning_rate=$lr/g" cfg/cosface.cfg
    sed -i "s/batch=.*/batch=150/g" cfg/cosface.cfg
    ./cnn classifier train cfg/face_recognition.data cfg/cosface.cfg  >> log_test
    sed -i "s/batch=.*/batch=1/g" cfg/cosface.cfg
    ./cnn classifier valid cfg/face_recognition.data cfg/cosface.cfg backup/cosface_final.weights
    python scripts/evaluation_recongnition.py >> log_test
done
