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
            //printf("%f\n", layer->output[i + index]);
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

