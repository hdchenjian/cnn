#include "softmax_layer.h"

softmax_layer *make_softmax_layer(int inputs, int batch)
{
    fprintf(stderr, "Softmax:            %d inputs\n", inputs);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->inputs = inputs;
    layer->batch = batch;
    layer->output = calloc(batch * inputs, sizeof(float));
    layer->delta = calloc(batch * inputs, sizeof(float));
    return layer;
}

void forward_softmax_layer(const softmax_layer *layer, float *input)
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
		}
	}
}

void backward_softmax_layer(const softmax_layer *layer, float *delta)
{
    int element_num = layer->inputs * layer->batch;
    for(int i = 0; i < element_num; ++i){
        delta[i] = layer->delta[i];
    }
}

