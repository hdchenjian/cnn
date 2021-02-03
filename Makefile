FORWARD_GPU=0
GPU=0
USE_CUBLAS=0
INTEL_MKL=0
QML=0
ARM_BLAS=0
DEBUG=0
CUDNN=0
OPENMP=0
OPENCL=0
CLBLAS=0
ARCH= -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_61,code=[sm_61,compute_61]

ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

VPATH=./src/:./examples:./test
EXEC=cnn
EXEC_TEST=cnn_test
OBJDIR=./obj/
SLIB = $(addprefix $(OBJDIR), libcnn.so)
ALIB = $(addprefix $(OBJDIR), libcnn.a)

CC=gcc
NVCC=/usr/local/cuda/bin/nvcc 
AR=ar
ARFLAGS=rcs
LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc/ -DUSE_LINUX
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC --std=gnu11 -Wunused-but-set-variable -Wno-unused-result -DUSE_LINUX

ifeq ($(FORWARD_GPU), 1)
COMMON+= -DFORWARD_GPU
CFLAGS+= -DFORWARD_GPU
endif

ifeq ($(INTEL_MKL), 1)
COMMON+= -DINTEL_MKL -I/opt/intel/mkl/include
CFLAGS+= -DINTEL_MKL
LDFLAGS+= -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
endif

ifeq ($(ARM_BLAS), 1)
COMMON+= -DARM_BLAS -I/opt/arm/armpl-19.2.0_Cortex-A72_Ubuntu-16.04_arm-hpc-compiler_19.2_aarch64-linux/include_mp
CFLAGS+= -DARM_BLAS
endif

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

OPTS=-Ofast
NVCC_OPTS="-Wall -fPIC -use_fast_math"
ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
NVCC_OPTS="-Wall -fPIC -lineinfo"
endif
CFLAGS+=$(OPTS)

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(USE_CUBLAS), 1)
COMMON+= -DUSE_CUBLAS
CFLAGS+= -DUSE_CUBLAS
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN -I/opt/ego/cudnn-v7/
CFLAGS+= -DCUDNN
LDFLAGS+= -L/opt/ego/cudnn-v7 -lcudnn
endif

ifeq ($(OPENCL), 1)
COMMON+= -DOPENCL -I/usr/local/cuda-8.0/targets/x86_64-linux/include
LDFLAGS+= -L/usr/local/cuda-8.0/targets/x86_64-linux/lib -lOpenCL
endif

OBJ=cuda.o utils.o gemm.o image.o box.o blas.o data.o tree.o list.o parser.o network.o option_list.o activations.o convolutional_layer.o maxpool_layer.o softmax_layer.o avgpool_layer.o cost_layer.o connected_layer.o dropout_layer.o route_layer.o shortcut_layer.o normalize_layer.o rnn_layer.o lstm_layer.o gru_layer.o upsample_layer.o yolo_layer.o batchnorm_layer.o detector.o classifier.o

ifeq ($(OPENCL), 1)
OBJ+= blas_cl.o opencl.o
endif

ifeq ($(CLBLAS), 1)
COMMON+= -DCLBLAS
LDFLAGS= -lclBLAS
endif

ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=blas_kernels.o convolutional_kernels.o activation_kernels.o maxpool_layer_kernels.o dropout_layer_kernals.o avgpool_layer_kernals.o
endif

EXECOBJA=cnn.o rnn.o
EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
EXECOBJA_TEST=test.o
EXECOBJ_TEST = $(addprefix $(OBJDIR), $(EXECOBJA_TEST))

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: backup obj $(SLIB) $(ALIB) $(EXEC) $(EXEC_TEST)
#all: obj $(SLIB) $(ALIB) $(EXEC)

$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(EXEC_TEST) : $(EXECOBJ_TEST) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options $(NVCC_OPTS) -c $< -o $@

obj:
	mkdir -p obj

backup:
	mkdir -p backup

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(EXECOBJ_TEST) $(EXEC_TEST)

