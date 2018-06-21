#include "softmax_layer.h"

softmax_layer *make_softmax_layer(int inputs, int batch, int is_last_layer)
{
    fprintf(stderr, "Softmax:            %d inputs\n", inputs);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->is_last_layer = is_last_layer;
    layer->inputs = inputs;
    layer->batch = batch;
    layer->output = calloc(batch * inputs, sizeof(float));
    layer->delta = calloc(batch * inputs, sizeof(float));
    layer->loss = calloc(inputs*batch, sizeof(float));
    layer->cost = calloc(1, sizeof(float));
#ifdef GPU
    layer->output_gpu = cuda_make_array(layer->output, inputs*batch);
    layer->delta_gpu = cuda_make_array(layer->delta, inputs*batch);
    layer->loss_gpu = cuda_make_array(layer->loss, 1);
#endif
    return layer;
}

void forward_softmax_layer(const softmax_layer *layer, float *input, network *net)
{
    int i;
    for(int b = 0; b < layer->batch; b++){
        int index = b * layer->inputs;
        float sum = 0;
        float largest = -FLT_MAX;
        for(i = 0; i < layer->inputs; ++i){
            if(input[i + index] > largest) largest = input[i + index];
        }
        for(i = 0; i < layer->inputs; ++i){
            float e = exp(input[i + index] - largest);
            sum += e;
            layer->output[i + index] = e;
        }
        for(i = 0; i < layer->inputs; ++i){
            layer->output[i + index] /= sum;
            printf("%f %f\n", input[i + index], layer->output[i + index]);
        }
    }

    if(layer->is_last_layer){
        for(int b = 0; b < layer->batch; ++b){
            int max_i = 0;
            double max = input[b * layer->inputs];
            for(int j = 0; j < net->classes; ++j){
                if(input[j + b * layer->inputs] > max){
                    max = input[j + b * layer->inputs];
                    max_i = j;
                }
            }
            if(net->truth[max_i + b * layer->inputs] > 0.99F) net->correct_num += 1;
        }
        softmax_x_ent_cpu(layer->batch*layer->inputs, layer->output, net->truth, layer->delta, layer->loss);
        layer->cost[0] = sum_array(layer->loss, layer->batch*layer->inputs);
        net->loss = layer->cost[0];
    }
}

void backward_softmax_layer(const softmax_layer *layer, float *delta)
{
    int element_num = layer->inputs * layer->batch;
    for(int i = 0; i < element_num; ++i){
        delta[i] = layer->delta[i];
    }
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer *layer)
{
    cuda_pull_array(layer->output_gpu, layer->output, layer->inputs*layer->batch);
}

void forward_softmax_layer_gpu(const softmax_layer *layer, float *input_gpu, network *net)
{
    softmax_gpu_me(input_gpu, layer->inputs, layer->batch, layer->output_gpu);

    float *output_temp = (float *)calloc(layer->inputs*layer->batch, sizeof(float));
    cuda_pull_array(layer->output_gpu, output_temp, layer->inputs*layer->batch);
    float *input_temp = calloc(layer->inputs*layer->batch, sizeof(float));
    cuda_pull_array(input_gpu, input_temp, layer->inputs*layer->batch);
    for(int i = 0; i < layer->inputs*layer->batch; i++){
        printf("%f %f\n", input_temp[i], output_temp[i]);
    }
    printf("\n");
    /*
    if(net->truth){
        softmax_x_ent_gpu(layer->batch*layer->inputs, layer->output_gpu, net.truth_gpu,
        		layer->delta_gpu, layer->loss_gpu);
        cuda_pull_array(layer->loss_gpu, layer->loss, layer->batch*layer->inputs);
        layer->cost[0] = sum_array(layer->loss, layer->batch*layer->inputs);
    }*/
}

void backward_softmax_layer_gpu(const softmax_layer *layer, float *delta_gpu)
{
    axpy_gpu(layer->batch*layer->inputs, 1, layer->delta_gpu, 1, delta_gpu, 1);
}

#endif
