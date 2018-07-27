#### This is a neural network framwork for deep learning written in C and CUDA.

[Darknet](https://pjreddie.com/darknet/) is a powerful neural network framework, You can even train a Go game.
I reimplement a tiny neural network(CNN, RNN) framework from scratch.

#### Compile
```
modify Makefile
change GPU=1 if you have GPU support CUDA
change OPENMP=1 if you want use openmp

make
```

#### you can train cifar dataset use:
```
./cnn classifier train  cfg/cifar.data cfg/cifar.cfg
```
#### you can see this:

```
layer     filters    size              input                output
  0: Convolutional:      32 x 32 x 3 inputs, 128 weights size 3 stride 1 -> 32 x 32 x 128 outputs 0.007 BFLOPs
  1: Convolutional:      32 x 32 x 128 inputs, 128 weights size 3 stride 1 -> 32 x 32 x 128 outputs 0.302 BFLOPs
  2: Convolutional:      32 x 32 x 128 inputs, 128 weights size 3 stride 1 -> 32 x 32 x 128 outputs 0.302 BFLOPs
  3: Maxpool:            32 x 32 x 128 inputs, size: 2, 2 stride
  4: Convolutional:      16 x 16 x 128 inputs, 256 weights size 3 stride 1 -> 16 x 16 x 256 outputs 0.151 BFLOPs
  5: Convolutional:      16 x 16 x 256 inputs, 256 weights size 3 stride 1 -> 16 x 16 x 256 outputs 0.302 BFLOPs
  6: Convolutional:      16 x 16 x 256 inputs, 256 weights size 3 stride 1 -> 16 x 16 x 256 outputs 0.302 BFLOPs
  7: Maxpool:            16 x 16 x 256 inputs, size: 2, 2 stride
  8: Convolutional:      8 x 8 x 256 inputs, 512 weights size 3 stride 1 -> 8 x 8 x 512 outputs 0.151 BFLOPs
  9: Convolutional:      8 x 8 x 512 inputs, 512 weights size 3 stride 1 -> 8 x 8 x 512 outputs 0.302 BFLOPs
 10: Convolutional:      8 x 8 x 512 inputs, 512 weights size 3 stride 1 -> 8 x 8 x 512 outputs 0.302 BFLOPs
 11: Convolutional:      8 x 8 x 512 inputs, 10 weights size 1 stride 1 -> 8 x 8 x 10 outputs 0.001 BFLOPs
 12: Avgpool:            8 x 8 x 10 image -> 1 x 1 x 10 image
 13: Softmax:            10 inputs, label_specific_margin_bias: 0.000000, margin_scale: 0
 net->workspace_gpu is not null, calloc for net->workspace just for test!!!

network total_bflop: 2.122 BFLOPs
Learning Rate: 0.005, Momentum: 0.9, Decay: 0.0005
image net has seen: 0, train_set_size: 50000, max_batches of net: 5000, net->classes: 10, net->batch: 128

epoch: 1, batch: 1, accuracy: 0.0625, loss: 118.318260, avg_loss: 118.32, learning_rate: 0.00500000, 2.8910 s, seen 128 images, max_accuracy: 0.0625
epoch: 1, batch: 2, accuracy: 0.1016, loss: 115.288254, avg_loss: 118.02, learning_rate: 0.00499600, 2.4676 s, seen 256 images, max_accuracy: 0.1016
epoch: 1, batch: 3, accuracy: 0.1042, loss: 115.342644, avg_loss: 117.75, learning_rate: 0.00499200, 2.4749 s, seen 384 images, max_accuracy: 0.1042
epoch: 1, batch: 4, accuracy: 0.1152, loss: 115.435532, avg_loss: 117.52, learning_rate: 0.00498801, 2.4740 s, seen 512 images, max_accuracy: 0.1152

```
##### the network will save to backup/cifar_final.weights, and valid network accuracy:
```
./cnn classifier valid  cfg/cifar.data cfg/cifar.cfg backup/cifar_final.weights
```
