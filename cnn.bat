D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include  ..\src\cuda.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\utils.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\gemm.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\image.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\box.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\blas.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\data.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\tree.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\list.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\parser.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\network.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\option_list.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\activations.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\convolutional_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\maxpool_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\softmax_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\avgpool_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\cost_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\connected_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\dropout_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\route_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\shortcut_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\normalize_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\rnn_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\lstm_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\gru_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\upsample_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\yolo_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU  -O2 -ID:\software\cuda\cuda\include ..\src\batchnorm_layer.c -c 
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU -I..\src  -O2 -ID:\software\cuda\cuda\include ..\examples\detector.c -c
D:\software\vs2015\VC\bin\amd64\cl -DFORWARD_GPU -DGPU -I..\src -O2 -ID:\software\cuda\cuda\include ..\examples\classifier.c -c

nvcc -use_fast_math -ID:\software\cuda\cuda\include   -gencode arch=compute_75,code=[sm_75,compute_75]  -I..\src -DFORWARD_GPU -DGPU  -c ..\src\blas_kernels.cu
nvcc -use_fast_math -ID:\software\cuda\cuda\include   -gencode arch=compute_75,code=[sm_75,compute_75]  -I..\src -DFORWARD_GPU -DGPU   -c ..\src\convolutional_kernels.cu
nvcc -use_fast_math -ID:\software\cuda\cuda\include   -gencode arch=compute_75,code=[sm_75,compute_75]  -I..\src -DFORWARD_GPU -DGPU   -c ..\src\activation_kernels.cu
nvcc -use_fast_math -ID:\software\cuda\cuda\include   -gencode arch=compute_75,code=[sm_75,compute_75]  -I..\src -DFORWARD_GPU -DGPU   -c ..\src\maxpool_layer_kernels.cu
nvcc -use_fast_math -ID:\software\cuda\cuda\include   -gencode arch=compute_75,code=[sm_75,compute_75]  -I..\src -DFORWARD_GPU -DGPU   -c ..\src\dropout_layer_kernals.cu
nvcc -use_fast_math -ID:\software\cuda\cuda\include   -gencode arch=compute_75,code=[sm_75,compute_75]  -I..\src -DFORWARD_GPU -DGPU   -c ..\src\avgpool_layer_kernals.cu

D:\software\vs2015\VC\bin\amd64\lib cuda.obj utils.obj gemm.obj image.obj box.obj blas.obj data.obj tree.obj list.obj parser.obj network.obj option_list.obj activations.obj convolutional_layer.obj maxpool_layer.obj softmax_layer.obj avgpool_layer.obj cost_layer.obj connected_layer.obj dropout_layer.obj route_layer.obj shortcut_layer.obj normalize_layer.obj rnn_layer.obj lstm_layer.obj gru_layer.obj upsample_layer.obj yolo_layer.obj batchnorm_layer.obj detector.obj classifier.obj blas_kernels.obj convolutional_kernels.obj activation_kernels.obj maxpool_layer_kernels.obj dropout_layer_kernals.obj avgpool_layer_kernals.obj

move cuda.lib cnn.lib
