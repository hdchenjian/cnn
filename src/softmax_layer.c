#include "softmax_layer.h"

softmax_layer *make_softmax_layer(int inputs, int batch, int is_last_layer,
                                  float label_specific_margin_bias, int margin_scale)
{
    fprintf(stderr, "Softmax:            %d inputs, label_specific_margin_bias: %f, margin_scale: %d\n",
            inputs, label_specific_margin_bias, margin_scale);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->is_last_layer = is_last_layer;
    layer->inputs = inputs;
    layer->batch = batch;
    layer->label_specific_margin_bias = label_specific_margin_bias;
    layer->margin_scale = margin_scale;

    layer->output = calloc(batch * inputs, sizeof(float));
    layer->delta = calloc(batch * inputs, sizeof(float));
    layer->loss = calloc(inputs*batch, sizeof(float));
    layer->cost = calloc(1, sizeof(float));
#ifdef GPU
    layer->output_gpu = cuda_make_array(layer->output, inputs*batch);
    layer->delta_gpu = cuda_make_array(layer->delta, inputs*batch);
    layer->loss_gpu = cuda_make_array(layer->loss, inputs*batch);
#endif
    return layer;
}

void forward_softmax_layer(const softmax_layer *layer, float *input, network *net)
{
    for(int b = 0; b < layer->batch; b++){
        int index = b * layer->inputs;
        if(layer->label_specific_margin_bias < -0.01 && net->test == 0){    // 0: train, 1: valid, 2: test
            for(int i = 0; i < layer->inputs; ++i){
                if(input[b * layer->inputs + net->truth_label_index[b]] > -layer->label_specific_margin_bias){
                    input[b * layer->inputs + net->truth_label_index[b]] += layer->label_specific_margin_bias;
                }
                if(layer->margin_scale > 0){
                    input[b * layer->inputs + i] *= layer->margin_scale;
                }
            }
        }

        float sum = 0;
        float largest = -FLT_MAX;
        for(int i = 0; i < layer->inputs; ++i){
            if(input[i + index] > largest) largest = input[i + index];
        }
        for(int i = 0; i < layer->inputs; ++i){
            float e = exp(input[i + index] - largest);
            sum += e;
            layer->output[i + index] = e;
        }
        for(int i = 0; i < layer->inputs; ++i){
            layer->output[i + index] /= sum;
            //printf("%f %f\n", input[i + index], layer->output[i + index]);
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
        //softmax_x_ent_cpu(layer->batch*layer->inputs, layer->output, net->truth, layer->delta, layer->loss);
        l2_cpu(layer->batch*layer->inputs, layer->output, net->truth, layer->delta, layer->loss);
        layer->cost[0] = sum_array(layer->loss, layer->batch*layer->inputs);
        net->loss = layer->cost[0];
    }
}

void backward_softmax_layer(const softmax_layer *layer, float *delta)
{
    int element_num = layer->inputs * layer->batch;
    float scale = 1.0F;
    if(layer->margin_scale > 0){
        scale = layer->margin_scale;
    }
    for(int i = 0; i < element_num; ++i){
        delta[i] = scale * layer->delta[i];
    }
}

#ifdef GPU

void forward_softmax_layer_gpu(const softmax_layer *layer, float *input_gpu, network *net)
{
    if(layer->label_specific_margin_bias < -0.01 && net->test == 0){    // 0: train, 1: valid, 2: test
        specific_margin_gpu(layer->batch, layer->inputs, input_gpu, layer->label_specific_margin_bias, layer->margin_scale, net->truth_label_index_gpu);
    }

    softmax_gpu_me(input_gpu, layer->inputs, layer->batch, layer->output_gpu);
    if(layer->is_last_layer){
        cudaError_t status = cudaMemset(net->is_not_max_gpu, 0, sizeof(int) * layer->batch);
        check_error(status);
        is_max_gpu(layer->batch, layer->inputs, layer->output_gpu, net->truth_label_index_gpu, net->is_not_max_gpu);
        int *is_not_max_cpu = calloc(layer->batch, sizeof(int));
        cuda_pull_array(net->is_not_max_gpu, is_not_max_cpu, layer->batch);
        for(int b = 0; b < layer->batch; ++b){
            if(is_not_max_cpu[b] == 0){
            }net->correct_num += 1;
        }
        /*float *input_temp = calloc(layer->inputs*layer->batch, sizeof(float));
        cuda_pull_array(input_gpu, input_temp, layer->batch*layer->inputs);
        for(int b = 0; b < layer->batch; ++b){
            int max_i = 0;
            double max = input_temp[b * layer->inputs];
            for(int j = 0; j < net->classes; ++j){
                //printf("%d %f %f %f\n", j, net->truth[j], layer->output[j], layer->delta[j]);
                if(input_temp[j + b * layer->inputs] > max){
                    max = input_temp[j + b * layer->inputs];
                    max_i = j;
                }
            }
            if(net->truth[max_i + b * layer->inputs] > 0.99F) net->correct_num += 1;
            }*/
        //softmax_x_ent_gpu(layer->batch*layer->inputs, layer->output_gpu, net->truth_gpu, layer->delta_gpu, layer->loss_gpu);
        l2_gpu(layer->batch*layer->inputs, layer->output_gpu, net->truth_gpu, layer->delta_gpu, layer->loss_gpu);
        cuda_pull_array(layer->loss_gpu, layer->loss, layer->batch*layer->inputs);
        layer->cost[0] = sum_array(layer->loss, layer->batch*layer->inputs);
        net->loss = layer->cost[0];
    }

    /* float *input_temp = calloc(layer->inputs*layer->batch, sizeof(float));
    cuda_pull_array(input_gpu, input_temp, layer->inputs*layer->batch);
    float *output_temp = (float *)calloc(layer->inputs*layer->batch, sizeof(float));
    cuda_pull_array(layer->output_gpu, output_temp, layer->inputs*layer->batch);
    for(int i = 0; i < layer->inputs*layer->batch; i++){
        printf("%f %f %f\n", net->truth[i], input_temp[i], output_temp[i]);
        if(i % net->classes == 0) printf("\n");
    }
    free(output_temp);
    free(input_temp);
    printf("\n");*/
}

void backward_softmax_layer_gpu(const softmax_layer *layer, float *delta_gpu)
{
    fill_gpu(layer->batch*layer->inputs, 0, delta_gpu, 1);
    float scale = 1.0F;
    if(layer->margin_scale > 0){
        scale = layer->margin_scale;
    }
    axpy_gpu(layer->batch*layer->inputs, scale, layer->delta_gpu, 1, delta_gpu, 1);
}

#endif
