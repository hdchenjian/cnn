#include <math.h>
#include <stdio.h>

#include "utils.h"
#include "network.h"

cost_layer *make_cost_layer(int batch, int inputs, enum COST_TYPE cost_type, float scale)
{
    fprintf(stderr, "Cost:               %d inputs\n", inputs);
    cost_layer *l = calloc(1, sizeof(cost_layer));;
    l->scale = scale;   // scale error to previous layer: backward_cost_layer
    l->batch = batch;
    l->inputs = inputs;
    l->outputs = inputs;
    l->cost_type = cost_type;
    l->delta = calloc(inputs*batch, sizeof(float));
    l->output = calloc(inputs*batch, sizeof(float));
    l->cost = calloc(1, sizeof(float));
    #ifdef GPU
    l->delta_gpu = cuda_make_array(l->output, inputs*batch);
    l->output_gpu = cuda_make_array(l->delta, inputs*batch);
    #endif
    return l;
}

void forward_cost_layer(const cost_layer *l, float *input, network *net)
{
    if (net->test == 2) return;  // 0: train, 1: valid
    l2_cpu(l->batch, l->inputs, input, net->truth_label_index, l->delta, l->output);

    for(int b = 0; b < l->batch; ++b){
        int index = b * l->inputs;
        int max_i = net->truth_label_index[b];
        double max = input[index + net->truth_label_index[b]];
        for(int j = 0; j < net->classes; ++j){
            //printf("%d %d %f %f\n", j, j == net->truth_label_index[b], input[j], l->delta[j]);
            if(input[j + index] >= max && j != max_i){
                max = input[j + index];
                max_i = j;
                //break;
            }
        }
        if(net->truth_label_index[b] == max_i) net->correct_num += 1;
    }
    l->cost[0] = sum_array(l->output, l->batch*l->inputs) / l->batch;
    net->loss = l->cost[0];
}

void backward_cost_layer(const cost_layer *l, float *delta)
{
    for(int i = 0; i < l->batch*l->inputs; ++i) delta[i] = l->scale * l->delta[i];
}

#ifdef GPU

void forward_cost_layer_gpu(const cost_layer *l, float *input_gpu, network *net)
{
    if (net->test == 2) return;  // 0: train, 1: valid
    l2_gpu(l->batch, l->inputs, input_gpu, net->truth_label_index_gpu, l->delta_gpu, l->output_gpu);
    cuda_pull_array(l->output_gpu, l->output, l->batch*l->inputs);
    l->cost[0] = sum_array(l->output, l->batch*l->inputs);
    net->loss = l->cost[0];
    cuda_pull_array(l->delta_gpu, l->delta, l->batch*l->inputs);

    /*
    cuda_pull_array(l->delta_gpu, l->delta, l->batch*l->inputs);
    float *input_temp = calloc(l->inputs*l->batch, sizeof(float));
    cuda_pull_array(input_gpu, input_temp, l->batch*l->inputs);*/
    float *input_temp = calloc(l->inputs*l->batch, sizeof(float));
    cuda_pull_array(input_gpu, input_temp, l->batch*l->inputs);
    for(int b = 0; b < l->batch; ++b){
        int max_i = 0;
        double max = input_temp[b * l->inputs];
        for(int j = 0; j < net->classes; ++j){
            //printf("%d %d %f %f %f\n", j, j == net->truth_label_index[b], input_temp[j], l->output[j], l->delta[j]);
            if(input_temp[j + b * l->inputs] > max){
                max = input_temp[j + b * l->inputs];
                max_i = j;
            }
        }
        if(net->truth_label_index[b] == max_i) net->correct_num += 1;
    }
}

void backward_cost_layer_gpu(const cost_layer *l, float *delta_gpu)
{
    fill_gpu(l->batch*l->inputs, 0, delta_gpu, 1);
    axpy_gpu(l->batch*l->inputs, l->scale, l->delta_gpu, 1, delta_gpu, 1);
}
#endif

