#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "cuda.h"

void run_classifier(int argc, char **argv);
void run_char_rnn(int argc, char **argv);
void run_detector(int argc, char **argv);
void cl_setup();

int main(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    int gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#if defined(OPENCL)
    cl_setup();
#elif defined(GPU)
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")){
        run_char_rnn(argc, argv);
    } else {
        fprintf(stderr, "Not an option: %s gpu_index: %d\n", argv[1], gpu_index);
    }
    return 0;
}

