GPU=0
DEBUG=1
CUDNN=0
OPENMP=0
ARCH= -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_61,code=[sm_61,compute_61]

VPATH=./src/:./examples:./test
EXEC=darknet
EXEC_TEST=darknet_test
OBJDIR=./obj/
SLIB = $(addprefix $(OBJDIR), libdarknet.so)
ALIB = $(addprefix $(OBJDIR), libdarknet.a)

CC=gcc
NVCC=/usr/local/cuda/bin/nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC --std=gnu11 -Wunused-but-set-variable

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN -I/opt/ego/cudnn-v7/
CFLAGS+= -DCUDNN
LDFLAGS+= -L/opt/ego/cudnn-v7 -lcudnn
endif

OBJ=cuda.o utils.o gemm.o image.o box.o blas.o data.o tree.o list.o parser.o network.o option_list.o activations.o convolutional_layer.o maxpool_layer.o softmax_layer.o avgpool_layer.o cost_layer.o connected_layer.o dropout_layer.o

ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=blas_kernels.o convolutional_kernels.o activation_kernels.o maxpool_layer_kernels.o
endif

EXECOBJA=classifier.o darknet.o
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
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "-Wall -fPIC" -c $< -o $@

obj:
	mkdir -p obj

backup:
	mkdir -p backup

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(EXECOBJ_TEST) $(EXEC_TEST)

