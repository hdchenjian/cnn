#include "network.h"

softmax_layer *make_softmax_layer(int inputs, int batch, int is_last_layer,
                                  float label_specific_margin_bias, int margin_scale)
{
    fprintf(stderr, "Softmax:            %d inputs, label_specific_margin_bias: %f, margin_scale: %d\n",
            inputs, label_specific_margin_bias, margin_scale);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->is_last_layer = is_last_layer;
    layer->inputs = inputs;
    layer->outputs = inputs;
    layer->batch = batch;
    layer->label_specific_margin_bias = label_specific_margin_bias;
    layer->margin_scale = margin_scale;

    layer->output = calloc(batch * inputs, sizeof(float));
    layer->delta = calloc(batch * inputs, sizeof(float));
    layer->loss = calloc(inputs*batch, sizeof(float));
    layer->cost = calloc(1, sizeof(float));
    if(layer->label_specific_margin_bias < -0.01)
        layer->input_backup = calloc(batch * inputs, sizeof(float));
#ifdef GPU
    layer->output_gpu = cuda_make_array(layer->output, inputs*batch);
    layer->delta_gpu = cuda_make_array(layer->delta, inputs*batch);
    layer->loss_gpu = cuda_make_array(layer->loss, inputs*batch);
    if(layer->label_specific_margin_bias < -0.01)
        layer->input_backup_gpu = cuda_make_array(layer->input_backup, inputs*batch);
#endif
    return layer;
}

void forward_softmax_layer(softmax_layer *layer, float *input, network *net)
{
    if(layer->label_specific_margin_bias < -0.01 && net->test == 0){    // 0: train, 1: valid
        // float *input is the output of previous layer, we can not modify
        memcpy(layer->input_backup, input, layer->batch * layer->inputs * sizeof(float));
    } else {
        layer->input_backup = input;
    }
    for(int b = 0; b < layer->batch; b++){
        int index = b * layer->inputs;
        if(layer->label_specific_margin_bias < -0.01 && net->test == 0){    // 0: train, 1: valid
            if(layer->input_backup[index + net->truth_label_index[b]] > -layer->label_specific_margin_bias){
                layer->input_backup[index + net->truth_label_index[b]] += layer->label_specific_margin_bias;
            }
            for(int i = 0; i < layer->inputs; ++i){
                if(layer->margin_scale > 0){
                    layer->input_backup[index + i] *= layer->margin_scale;
                }
            }
        }
        float sum = 0;
        float largest = -FLT_MAX;
        for(int i = 0; i < layer->inputs; ++i){
            if(layer->input_backup[i + index] > largest) largest = layer->input_backup[i + index];
        }
        for(int i = 0; i < layer->inputs; ++i){
            float e = exp(layer->input_backup[i + index] - largest);
            sum += e;
            layer->output[i + index] = e;
        }
        for(int i = 0; i < layer->inputs; ++i){
            layer->output[i + index] /= sum;
            //printf("%d %d %f %f\n",
            //       i, i == net->truth_label_index[b], layer->input_backup[i + index], layer->output[i + index]);
        }
    }

    if(layer->is_last_layer && net->truth_label_index){
        for(int b = 0; b < layer->batch; ++b){
            int index = b * layer->inputs;
            int max_i = net->truth_label_index[b];
            double max = layer->input_backup[index + net->truth_label_index[b]];
            for(int j = 0; j < net->classes; ++j){
                //printf("forward_softmax_layer: %d truth_index %d %f %f %f\n",
                //       j, j == net->truth_label_index[b], input[j], layer->input_backup[j], layer->output[j]);
                if(layer->input_backup[j + index] >= max && j != max_i){
                    max = layer->input_backup[j + index];
                    max_i = j;
                    break;
                }
            }
            if(net->truth_label_index[b] == max_i) net->correct_num += 1;
        }
        //softmax_x_ent_cpu(layer->batch, layer->inputs, layer->output, net->truth_label_index, layer->delta, layer->loss);
        l2_cpu(layer->batch, layer->inputs, layer->output, net->truth_label_index, layer->delta, layer->loss);
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
        delta[i] += scale * layer->delta[i];
        //printf("%f \n", delta[i]);
    }
}

#ifdef GPU

void forward_softmax_layer_gpu(softmax_layer *layer, float *input_gpu, network *net)
{
    if(layer->label_specific_margin_bias < -0.01 && net->test == 0){    // 0: train, 1: valid
        cuda_mem_copy(layer->input_backup_gpu, input_gpu, layer->batch * layer->inputs);
        specific_margin_add_gpu(layer->batch, layer->inputs, layer->input_backup_gpu, layer->label_specific_margin_bias,
                                layer->margin_scale, net->truth_label_index_gpu);
    } else {
        layer->input_backup_gpu = input_gpu;
    }
    softmax_gpu_me(layer->input_backup_gpu, layer->inputs, layer->batch, layer->output_gpu);

    if(layer->is_last_layer &&  net->truth_label_index_gpu){
        cudaError_t status = cudaMemset(net->is_not_max_gpu, 0, sizeof(int) * layer->batch);
        check_error(status);
        is_max_gpu(layer->batch, layer->inputs, layer->output_gpu, net->truth_label_index_gpu, net->is_not_max_gpu);
        int *is_not_max_cpu = calloc(layer->batch, sizeof(int));
        cuda_pull_array_int(net->is_not_max_gpu, is_not_max_cpu, layer->batch);
        int correct_num = 0;
        for(int b = 0; b < layer->batch; ++b){
            if(is_not_max_cpu[b] == 0){
                correct_num += 1;
            }
        }
        net->correct_num += correct_num;
        /*printf("correct_num: %d\n", correct_num);

        float *input_temp = calloc(layer->inputs*layer->batch, sizeof(float));
        cuda_pull_array(layer->output_gpu, input_temp, layer->batch*layer->inputs);
        int correct_num1 = 0;
        for(int b = 0; b < layer->batch; ++b){
            int index = b * layer->inputs;
            int max_i = net->truth_label_index[b];
            double max = input_temp[index + net->truth_label_index[b]];
            for(int j = 0; j < net->classes; ++j){
                printf("%d %d %f\n", j, j == net->truth_label_index[b], input_temp[j]);
                if(input_temp[j + index] >= max && j != max_i){
                    max = input_temp[j + index];
                    max_i = j;
                    break;
                }
            }
            if(net->truth_label_index[b] == max_i) correct_num1 += 1;
        }
        printf("correct_num: %d\n", correct_num1);
        if(correct_num1 != correct_num) exit(-1);
        */
        //softmax_x_ent_gpu(layer->batch, layer->inputs, layer->output_gpu, net->truth_label_index_gpu, layer->delta_gpu, layer->loss_gpu);
        l2_gpu(layer->batch, layer->inputs, layer->output_gpu, net->truth_label_index_gpu, layer->delta_gpu, layer->loss_gpu);
        cuda_pull_array(layer->loss_gpu, layer->loss, layer->batch*layer->inputs);
        layer->cost[0] = sum_array(layer->loss, layer->batch*layer->inputs);
        net->loss = layer->cost[0];
    }
}

void backward_softmax_layer_gpu(const softmax_layer *layer, float *delta_gpu)
{
    float scale = 1.0F;
    if(layer->margin_scale > 0){
        scale = layer->margin_scale;
    }
    axpy_gpu(layer->batch*layer->inputs, scale, layer->delta_gpu, 1, delta_gpu, 1);
}

#endif
