#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "cuda.h"

void run_classifier(int argc, char **argv);

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

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else {
        fprintf(stderr, "Not an option: %s gpu_index: %d\n", argv[1], gpu_index);
    }
    return 0;
}

