#include "network.h"

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(void *));
    net->layers_type = calloc(net->n, sizeof(enum LAYER_TYPE));
    net->seen = 0;
    net->test = 0;
    net->batch_train = 0;
    net->epoch = 0;
    net->correct_num = 0;
    net->correct_num_count = 0;
    net->workspace_size = 0;
    net->workspace = NULL;
    return net;
}

void free_network(network *net)
{
    for(int i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    free(net->layers_type);
#ifdef GPU
    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
#endif
    free(net);
}

float update_current_learning_rate(network * net)
{
    switch (net->policy) {
        case STEPS:
            for(int i = 0; i < net->num_steps; ++i){
                if(net->steps[i] == net->batch_train){
                    net->learning_rate *= net->scales[i];
                }
            }
            return net->learning_rate;;
        default:
            //fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

void forward_network(network *net, float *input)
{
    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            forward_convolutional_layer(layer, input, net->workspace, net->test);
            input = layer->output;
        }else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            forward_connected_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            forward_maxpool_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            forward_avgpool_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            forward_dropout_layer(layer, input, net);
            input = layer->output;
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            forward_softmax_layer(layer, input, net);
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

void update_network(network *net)
{
    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            update_convolutional_layer(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            update_connected_layer(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == MAXPOOL){
        } else if(net->layers_type[i] == DROPOUT){
        } else if(net->layers_type[i] == AVGPOOL){
        } else if(net->layers_type[i] == SOFTMAX){
        } else if(net->layers_type[i] == COST){
        } else {
            printf("update_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
}

/* data_type: 0: output, 1: delta */
float *get_network_layer_data(network *net, int i, int data_type)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == CONNECTED){
        connected_layer *layer = (connected_layer *)net->layers[i];
        return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == DROPOUT){
        dropout_layer *layer = (dropout_layer *)net->layers[i];
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
        printf("get_network_layer_data layers_type error, layer: %d %d\n", i, net->layers_type[i]);
        exit(-1);
    }
}

void backward_network(network *net, float *input)
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
            backward_convolutional_layer(layer, prev_input, prev_delta, net->workspace, net->test);
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            backward_connected_layer(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            if(i != 0) backward_maxpool_layer(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            if(i != 0) backward_avgpool_layer(layer, prev_delta);
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            if(i != 0) backward_dropout_layer(layer, prev_delta);
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            if(i != 0) backward_softmax_layer(layer, prev_delta);
        } else if(net->layers_type[i] == COST){
            cost_layer *layer = (cost_layer *)net->layers[i];
            backward_cost_layer(layer, prev_delta);
        } else {
            printf("backward_network layers_type error, layer: %d\n", i);
            exit(-1);
        }

    }
}


#ifdef GPU

void forward_network_gpu(network *netp, float *input)
{
    network net = *netp;
    cuda_push_array(net.input_gpu, input, net->h * net->w * net->c * net->batch);
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.classes*net.batch);
    }
    for(int i = 0; i < net->n; ++i){
		if(net->layers_type[i] == CONVOLUTIONAL){
			convolutional_layer *layer = (convolutional_layer *)net->layers[i];
			forward_convolutional_layer(layer, input, net->workspace, net->test);
			input = layer->output;
		}else if(net->layers_type[i] == CONNECTED){
			connected_layer *layer = (connected_layer *)net->layers[i];
			forward_connected_layer(layer, input);
			input = layer->output;
		} else if(net->layers_type[i] == MAXPOOL){
			maxpool_layer *layer = (maxpool_layer *)net->layers[i];
			forward_maxpool_layer(layer, input);
			input = layer->output;
		} else if(net->layers_type[i] == AVGPOOL){
			avgpool_layer *layer = (avgpool_layer *)net->layers[i];
			forward_avgpool_layer(layer, input);
			input = layer->output;
		} else if(net->layers_type[i] == DROPOUT){
			dropout_layer *layer = (dropout_layer *)net->layers[i];
			forward_dropout_layer(layer, input, net);
			input = layer->output;
		} else if(net->layers_type[i] == SOFTMAX){
			softmax_layer *layer = (softmax_layer *)net->layers[i];
			forward_softmax_layer(layer, input, net);
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
    pull_network_output(netp);
    calc_network_cost(netp);
}

void backward_network_gpu(network *netp, float *input)
{
    int i;
    network net = *netp;
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
            backward_convolutional_layer(layer, prev_input, prev_delta, net->workspace, net->test);
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            backward_connected_layer(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            if(i != 0) backward_maxpool_layer(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            if(i != 0) backward_avgpool_layer(layer, prev_delta);
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            if(i != 0) backward_dropout_layer(layer, prev_delta);
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            if(i != 0) backward_softmax_layer(layer, prev_delta);
        } else if(net->layers_type[i] == COST){
            cost_layer *layer = (cost_layer *)net->layers[i];
            backward_cost_layer(layer, prev_delta);
        } else {
            printf("backward_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
}

void update_network_gpu(network *netp)
{
    network net = *netp;
    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            update_convolutional_layer(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            update_connected_layer(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == MAXPOOL){
        } else if(net->layers_type[i] == DROPOUT){
        } else if(net->layers_type[i] == AVGPOOL){
        } else if(net->layers_type[i] == SOFTMAX){
        } else if(net->layers_type[i] == COST){
        } else {
            printf("update_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
}

#endif

void train_network_batch(network *net, batch b)
{
    net->truth = b.truth;
#ifdef GPU
	forward_network_gpu(net, b.data);
	backward_network_gpu(net, b.data);
	update_network_gpu(net);
#else
    forward_network(net, b.data);
    backward_network(net, b.data);
    update_network(net);
#endif
    net->seen += net->batch;
    net->correct_num_count += net->batch;
    if(net->correct_num_count > 1000){
        net->correct_num_count = 0;
        net->correct_num = 0;
    }
    net->batch_train += 1;
}

void valid_network(network *net, batch b)
{
    for(int i = 0; i < b.n; ++i){
        //show_image(b.images[i], "Input");
        net->truth = b.truth;
        forward_network(net, b.data);
    }
    net->seen += net->batch;
    net->correct_num_count += net->batch;
    net->batch_train += 1;
}

int get_network_output_size_layer(network *net, int i)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        return layer->out_w * layer->out_h * layer->n;
    } else if(net->layers_type[i] == CONNECTED){
        connected_layer *layer = (connected_layer *)net->layers[i];
        return layer->outputs;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        image output = get_maxpool_image(layer, net->batch - 1);
        return output.h*output.w*output.c;
    }else if(net->layers_type[i] == AVGPOOL){
        avgpool_layer *layer = (avgpool_layer *)net->layers[i];
        return layer->c;
    }else if(net->layers_type[i] == DROPOUT){
        dropout_layer *layer = (dropout_layer *)net->layers[i];
        return layer->outputs;
    }else if(net->layers_type[i] == SOFTMAX){
        softmax_layer *layer = (softmax_layer *)net->layers[i];
        return layer->inputs;
    }else if(net->layers_type[i] == COST){
        cost_layer *layer = (cost_layer *)net->layers[i];
        return layer->outputs;
    } else {
        printf("get_network_output_size_layer layers_type error, layer: %d\n", i);
        exit(-1);
    }
}

image get_network_image_layer(network *net, int i)
{
    image im;
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        im.h = layer->out_h;
        im.w = layer->out_w;
        im.c = layer->n;
        im.data = layer->output;
        return im;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        return get_maxpool_image(layer, net->batch - 1);
    } else if(net->layers_type[i] == DROPOUT){
        dropout_layer *layer = (dropout_layer *)net->layers[i];
        return get_dropout_image(layer, net->batch - 1);
    } else {
        printf("get_network_image_layer layers_type error, layer: %d\n", i);
        exit(-1);
    }
}

void save_convolutional_weights(const convolutional_layer *l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    fwrite(l->biases, sizeof(float), l->n, fp);
    if (l->batch_normalize){
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l->weights, sizeof(float), l->n * l->size* l->size * l->c, fp);
}

void load_convolutional_weights(const convolutional_layer *l, FILE *fp)
{
    fread(l->biases, sizeof(float), l->n, fp);
    if (l.batch_normalize){
		fread(l.rolling_mean, sizeof(float), l.n, fp);
		fread(l.rolling_variance, sizeof(float), l.n, fp);
	}
    fread(l->weights, sizeof(float), l->n * l->size* l->size * l->c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void save_connected_weights(const connected_layer *l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l->biases, sizeof(float), l->outputs, fp);
    fwrite(l->weights, sizeof(float), l->outputs*l->inputs, fp);
}

void load_connected_weights(const connected_layer *l, FILE *fp)
{
    fread(l->biases, sizeof(float), l->outputs, fp);
    fread(l->weights, sizeof(float), l->outputs*l->inputs, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void save_weights(network *net, char *filename)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    printf("weights version info: major: %d, minor: %d, revision: %d, net->seen: %lu\n", major, minor, revision, net->seen);

    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(&(net->seen), sizeof(size_t), 1, fp);

    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            save_convolutional_weights((convolutional_layer *)net->layers[i], fp);
        } if(net->layers_type[i] == CONNECTED){
            save_connected_weights((connected_layer *)net->layers[i], fp);
        }
    }
    fprintf(stderr, "Saving weights Done!\n\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    fread(&net->seen, sizeof(size_t), 1, fp);
    printf("weights version info: major: %d, minor: %d, revision: %d, net->seen: %lu\n", major, minor, revision, net->seen);

    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            load_convolutional_weights((convolutional_layer *)net->layers[i], fp);
        } if(net->layers_type[i] == CONNECTED){
            load_connected_weights((connected_layer *)net->layers[i], fp);
        }
    }
    fprintf(stderr, "Loading weights Done!\n");
    fclose(fp);
}

