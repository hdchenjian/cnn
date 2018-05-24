#include "network.h"

struct network *make_network(int n)
{
    struct network *net = calloc(1, sizeof(struct network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(void *));
    net->layers_type = calloc(net->n, sizeof(enum LAYER_TYPE));
    net->batch = 1;
    net->seen = 0;
    return net;
}

void forward_network(struct network *net, float *input)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            forward_convolutional_layer((convolutional_layer *)net->layers[i], input);
            input = layer->output;
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            forward_maxpool_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == AVGPOOL){
        	avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            forward_avgpool_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            forward_softmax_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == COST){
            cost_layer *layer = (cost_layer *)net->layers[i];
            forward_cost_layer(layer, input, net);
            input = layer->output;
        } else {
            printf("forward_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
}

void update_network(struct network *net, double step)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            update_convolutional_layer(layer, step, 0.9, .01);
        } else if(net->layers_type[i] == MAXPOOL){
        } else if(net->layers_type[i] == AVGPOOL){
        } else if(net->layers_type[i] == SOFTMAX){
        } else if(net->layers_type[i] == COST){
        } else {
        	printf("get_network_layer_data layers_type error, layer: %d\n", i);
        	exit(-1);
        }
    }
}

/* data_type: 0: output, 1: delta */
float *get_network_layer_data(struct network *net, int i, int data_type)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == AVGPOOL){
        avgpool_layer *layer = (avgpool_layer *)net->layers[i];
        return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == SOFTMAX){
        softmax_layer *layer = (softmax_layer *)net->layers[i];
        return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == COST){
        cost_layer *layer = (cost_layer *)net->layers[i];
        return data_type == 0 ? layer->output : layer->delta;
    } else {
        printf("get_network_layer_data layers_type error, layer: %d\n", i);
        exit(-1);
    }
}

float *get_network_output(struct network *net)
{
    return get_network_layer_data(net, net->n-1, 0);
}

void backward_network(struct network *net, float *input)
{
    float *prev_input;
    float *prev_delta;
    for(int i = net->n-1; i >= 0; --i){
        if(i == 0){
            prev_input = input;
            prev_delta = 0;
        }else{
            prev_input = get_network_layer_data(net, i-1, 0);
            prev_delta = get_network_layer_data(net, i-1, 1);
        }
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            learn_convolutional_layer(layer, prev_input);
            if(i != 0) backward_convolutional_layer(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            if(i != 0) backward_maxpool_layer(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            if(i != 0) backward_avgpool_layer(layer, prev_delta);
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            if(i != 0) backward_softmax_layer(layer, prev_delta);
        } else if(net->layers_type[i] == COST){
        	cost_layer *layer = (cost_layer *)net->layers[i];
            backward_cost_layer(layer, prev_delta);
        } else {
            printf("get_network_layer_data layers_type error, layer: %d\n", i);
        }

    }
}

void train_network_batch(struct network *net, batch b)
{
    for(int i = 0; i < b.n; ++i){
        //show_image(b.images[i], "Input");
    	net->truth = b.truth[i];
        forward_network(net, b.images[i].data);
        backward_network(net, b.images[i].data);
        update_network(net, .001);
    }
}

int get_network_output_size_layer(struct network *net, int i)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        image output = get_convolutional_image(layer);
        return output.h*output.w*output.c;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        image output = get_maxpool_image(layer);
        return output.h*output.w*output.c;
    }else if(net->layers_type[i] == AVGPOOL){
    	avgpool_layer *layer = (avgpool_layer *)net->layers[i];
        return layer->outputs;
    }else if(net->layers_type[i] == SOFTMAX){
        softmax_layer *layer = (softmax_layer *)net->layers[i];
        return layer->inputs;
    }else if(net->layers_type[i] == COST){
    	cost_layer *layer = (cost_layer *)net->layers[i];
        return layer->outputs;
    } else {
        printf("get_network_output_size_layer layers_type error, layer: %d\n", i);
        return 0;
    }
}

int get_network_output_size(struct network *net)
{
    int i = net->n-1;
    return get_network_output_size_layer(net, i);
}

image get_network_image_layer(struct network *net, int i)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        return get_convolutional_image(layer);
    }
    else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        return get_maxpool_image(layer);
    } else {
        printf("get_network_image_layer layers_type error, layer: %d\n", i);
        return make_empty_image(0,0,0);
    }
}

image get_network_image(struct network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    return make_empty_image(0,0,0);
}
