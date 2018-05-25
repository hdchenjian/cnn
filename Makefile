GPU=0
DEBUG=1
CUDNN=0
OPENMP=0
OPENCV=0

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

VPATH=./src/:./examples:./test
EXEC=darknet
EXEC_TEST=darknet_test
OBJDIR=./obj/
SLIB = $(addprefix $(OBJDIR), libdarknet.so)
ALIB = $(addprefix $(OBJDIR), libdarknet.a)

CC=gcc
NVCC=nvcc 
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

ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=utils.o image.o box.o blas.o data.o list.o parser.o network.o option_list.o activations.o convolutional_layer.o maxpool_layer.o softmax_layer.o avgpool_layer.o cost_layer.o connected_layer.o

ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o
endif

EXECOBJA=classifier.o darknet.o
EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
EXECOBJA_TEST=test.o
EXECOBJ_TEST = $(addprefix $(OBJDIR), $(EXECOBJA_TEST))

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: backup obj $(SLIB) $(ALIB) $(EXEC)
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
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

backup:
	mkdir -p backup

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(EXECOBJ_TEST) $(EXEC_TEST)

