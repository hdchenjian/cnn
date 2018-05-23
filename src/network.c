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
            convolutional_layer layer = *(convolutional_layer *)net->layers[i];
            forward_convolutional_layer(layer, input);
            input = layer.output;
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net->layers[i];
            forward_maxpool_layer(layer, input);
            input = layer.output;
        } else if(net->layers_type[i] == AVGPOOL){
        	avgpool_layer layer = *(avgpool_layer *)net->layers[i];
            forward_avgpool_layer(layer, input);
            input = layer.output;
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net->layers[i];
            forward_softmax_layer(layer, input);
            input = layer.output;
        } else if(net->layers_type[i] == COST){
            cost_layer layer = *(cost_layer *)net->layers[i];
            forward_cost_layer(layer, input, net);
            input = layer.output;
    }
}

void update_network(struct network *net, double step)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net->layers[i];
            update_convolutional_layer(layer, step, 0.9, .01);
        }
        else if(net->layers_type[i] == MAXPOOL){
            //maxpool_layer layer = *(maxpool_layer *)net->layers[i];
        }
        else if(net->layers_type[i] == SOFTMAX){
            //maxpool_layer layer = *(maxpool_layer *)net->layers[i];
        }
    }
}

float *get_network_output_layer(struct network *net, int i)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net->layers[i];
        return layer.output;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net->layers[i];
        return layer.output;
    } else if(net->layers_type[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net->layers[i];
        return layer.output;
    }
    return 0;
}
float *get_network_output(struct network *net)
{
    return get_network_output_layer(net, net->n-1);
}

float *get_network_delta_layer(struct network *net, int i)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net->layers[i];
        return layer.delta;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net->layers[i];
        return layer.delta;
    } else if(net->layers_type[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net->layers[i];
        return layer.delta;
    }
    return 0;
}

float *get_network_delta(struct network *net)
{
    return get_network_delta_layer(net, net->n-1);
}

void learn_network(struct network *net, float *input)
{
    int i;
    float *prev_input;
    float *prev_delta;
    for(i = net->n-1; i >= 0; --i){
        if(i == 0){
            prev_input = input;
            prev_delta = 0;
        }else{
            prev_input = get_network_output_layer(net, i-1);
            prev_delta = get_network_delta_layer(net, i-1);
        }
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net->layers[i];
            learn_convolutional_layer(layer, prev_input);
            if(i != 0) backward_convolutional_layer(layer, prev_input, prev_delta);
        }
        else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net->layers[i];
            if(i != 0) backward_maxpool_layer(layer, prev_input, prev_delta);
        }
        else if(net->layers_type[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net->layers[i];
            if(i != 0) backward_softmax_layer(layer, prev_input, prev_delta);
        }
    }
}

void train_network_batch(struct network *net, batch b)
{
    int i,j;
    int k = get_network_output_size(net);
    int correct = 0;
    for(i = 0; i < b.n; ++i){
        //show_image(b.images[i], "Input");
    	net.truth = b.truth[i];
        forward_network(net, b.images[i].data);
        float *output = get_network_output(net);
        float *delta = get_network_delta(net);
        int max_k = 0;
        double max = 0;
        for(j = 0; j < k; ++j){
            delta[j] = b.truth[i][j]-output[j];
            if(output[j] > max) {
                max = output[j];
                max_k = j;
            }
        }
        if(b.truth[i][max_k]) ++correct;
        printf("%f\n", (double)correct/(i+1));
        learn_network(net, b.images[i].data);
        update_network(net, .001);
    }
    printf("Accuracy: %f\n", (double)correct/b.n);
}

int get_network_output_size_layer(struct network *net, int i)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net->layers[i];
        image output = get_convolutional_image(layer);
        return output.h*output.w*output.c;
    }
    else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net->layers[i];
        image output = get_maxpool_image(layer);
        return output.h*output.w*output.c;
    }else if(net->layers_type[i] == AVGPOOL){
    	avgpool_layer layer = *(avgpool_layer *)net->layers[i];
        return layer.outputs;
    }else if(net->layers_type[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net->layers[i];
        return layer.inputs;
    }else if(net->layers_type[i] == COST){
    	cost_layer layer = *(cost_layer *)net->layers[i];
        return layer.outputs;
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
        convolutional_layer layer = *(convolutional_layer *)net->layers[i];
        return get_convolutional_image(layer);
    }
    else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net->layers[i];
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
