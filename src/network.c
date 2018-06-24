#include "network.h"

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->seen = 0;
    net->test = 0;
    net->batch_train = 0;
    net->epoch = 0;
    net->correct_num = 0;
    net->correct_num_count = 0;
    net->gpu_index = -1;

    net->layers = calloc(net->n, sizeof(void *));
    net->layers_type = calloc(net->n, sizeof(enum LAYER_TYPE));
    net->workspace_size = 0;
    net->workspace = NULL;
    net->workspace_gpu = NULL;
    net->input_gpu = NULL;
    net->truth_gpu = NULL;
    return net;
}

void free_network(network *net)
{
    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            if(layer->weights) free_ptr(layer->weights);
            if(layer->weight_updates) free_ptr(layer->weight_updates);
            if(layer->biases) free_ptr(layer->biases);
            if(layer->bias_updates) free_ptr(layer->bias_updates);
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
            if(layer->mean) free_ptr(layer->mean);
            if(layer->mean_delta) free_ptr(layer->mean_delta);
            if(layer->variance) free_ptr(layer->variance);
            if(layer->variance_delta) free_ptr(layer->variance_delta);
            if(layer->rolling_mean) free_ptr(layer->rolling_mean);
            if(layer->rolling_variance) free_ptr(layer->rolling_variance);
            if(layer->x) free_ptr(layer->x);

            if(layer->weights_gpu) cuda_free(layer->weights_gpu);
            if(layer->weight_updates_gpu) cuda_free(layer->weight_updates_gpu);
            if(layer->biases_gpu) cuda_free(layer->biases_gpu);
            if(layer->bias_updates_gpu) cuda_free(layer->bias_updates_gpu);
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            if(layer->mean_gpu) cuda_free(layer->mean_gpu);
            if(layer->mean_delta_gpu) cuda_free(layer->mean_delta_gpu);
            if(layer->variance_gpu) cuda_free(layer->variance_gpu);
            if(layer->variance_delta_gpu) cuda_free(layer->variance_delta_gpu);
            if(layer->rolling_mean_gpu) cuda_free(layer->rolling_mean_gpu);
            if(layer->rolling_variance_gpu) cuda_free(layer->rolling_variance_gpu);
            if(layer->x_gpu) cuda_free(layer->x_gpu);
            free_ptr(layer);
        }else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            if(layer->weights) free_ptr(layer->weights);
            if(layer->weight_updates) free_ptr(layer->weight_updates);
            if(layer->biases) free_ptr(layer->biases);
            if(layer->bias_updates) free_ptr(layer->bias_updates);
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);

            if(layer->weights_gpu) cuda_free(layer->weights_gpu);
            if(layer->weight_updates_gpu) cuda_free(layer->weight_updates_gpu);
            if(layer->biases_gpu) cuda_free(layer->biases_gpu);
            if(layer->bias_updates_gpu) cuda_free(layer->bias_updates_gpu);
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            free_ptr(layer);
        } else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);

            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            free_ptr(layer);
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
            if(layer->indexes) free_ptr(layer->indexes);

            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            if(layer->indexes_gpu) cuda_free_int(layer->indexes_gpu);
            free_ptr(layer);
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);

            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            free_ptr(layer);
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            if(layer->rand) free_ptr(layer->rand);
            if(layer->rand_gpu) cuda_free(layer->rand_gpu);
            free_ptr(layer);
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
            if(layer->loss) free_ptr(layer->loss);
            if(layer->cost) free_ptr(layer->cost);

            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            if(layer->loss_gpu) cuda_free(layer->loss_gpu);
            free_ptr(layer);
        } else if(net->layers_type[i] == COST){
            cost_layer *layer = (cost_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
            if(layer->cost) free_ptr(layer->cost);

            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            free_ptr(layer);
        } else {
            printf("forward_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
    free_ptr(net->layers);
    free_ptr(net->layers_type);
    if(net->workspace) free_ptr(net->workspace);

    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
    if(net->workspace_gpu) cuda_free(net->workspace_gpu);

    free_ptr(net);
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
        }else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            forward_route_layer(layer, net);
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
        } else if(net->layers_type[i] == ROUTE){
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
float *get_network_layer_data(network *net, int i, int data_type, int is_gpu)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        if(is_gpu)
            return data_type == 0 ? layer->output_gpu : layer->delta_gpu;
        else
            return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == CONNECTED){
        connected_layer *layer = (connected_layer *)net->layers[i];
        if(is_gpu)
            return data_type == 0 ? layer->output_gpu : layer->delta_gpu;
        else
            return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == ROUTE){
        route_layer *layer = (route_layer *)net->layers[i];
        if(is_gpu)
            return data_type == 0 ? layer->output_gpu : layer->delta_gpu;
        else
            return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        if(is_gpu)
            return data_type == 0 ? layer->output_gpu : layer->delta_gpu;
        else
            return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == DROPOUT){
        dropout_layer *layer = (dropout_layer *)net->layers[i];
        if(is_gpu)
            return data_type == 0 ? layer->output_gpu : layer->delta_gpu;
        else
            return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == AVGPOOL){
        avgpool_layer *layer = (avgpool_layer *)net->layers[i];
        if(is_gpu)
            return data_type == 0 ? layer->output_gpu : layer->delta_gpu;
        else
            return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == SOFTMAX){
        softmax_layer *layer = (softmax_layer *)net->layers[i];
        if(is_gpu)
            return data_type == 0 ? layer->output_gpu : layer->delta_gpu;
        else
            return data_type == 0 ? layer->output : layer->delta;
    } else if(net->layers_type[i] == COST){
        cost_layer *layer = (cost_layer *)net->layers[i];
        if(is_gpu)
            return data_type == 0 ? layer->output_gpu : layer->delta_gpu;
        else
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
            prev_input = get_network_layer_data(net, i-1, 0, 0);
            prev_delta = get_network_layer_data(net, i-1, 1, 0);
        }
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            backward_convolutional_layer(layer, prev_input, prev_delta, net->workspace, net->test);
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            backward_connected_layer(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            backward_route_layer(layer, net);
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

void forward_network_gpu(network *net, float *input)
{
    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            forward_convolutional_layer_gpu(layer, input, net->workspace_gpu, net->test);
            input = layer->output_gpu;
        }else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            forward_connected_layer_gpu(layer, input);
            input = layer->output_gpu;
        }else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            forward_route_layer_gpu(layer, net);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            forward_maxpool_layer_gpu(layer, input);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            forward_avgpool_layer_gpu(layer, input);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            forward_dropout_layer_gpu(layer, input, net);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            forward_softmax_layer_gpu(layer, input, net);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == COST){
            cost_layer *layer = (cost_layer *)net->layers[i];
            forward_cost_layer_gpu(layer, input, net);
            input = layer->output_gpu;
        } else {
            printf("forward_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
}

void backward_network_gpu(network *net, float *input)
{
    float *prev_input;
    float *prev_delta;
    for(int i = net->n-1; i >= 0; --i){
        if(i == 0){
            prev_input = input;
            prev_delta = 0;
        }else{
            prev_input = get_network_layer_data(net, i-1, 0, 1);
            prev_delta = get_network_layer_data(net, i-1, 1, 1);
        }
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            backward_convolutional_layer_gpu(layer, prev_input, prev_delta, net->workspace_gpu, net->test);
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            backward_connected_layer_gpu(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            backward_route_layer_gpu(layer, net);
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            if(i != 0) backward_maxpool_layer_gpu(layer, prev_delta);
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            if(i != 0) backward_avgpool_layer_gpu(layer, prev_delta);
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            if(i != 0) backward_dropout_layer_gpu(layer, prev_delta);
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            if(i != 0) backward_softmax_layer_gpu(layer, prev_delta);
        } else if(net->layers_type[i] == COST){
            cost_layer *layer = (cost_layer *)net->layers[i];
            backward_cost_layer_gpu(layer, prev_delta);
        } else {
            printf("backward_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
}

void update_network_gpu(network *net)
{
    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            update_convolutional_layer_gpu(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            update_connected_layer_gpu(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == ROUTE){
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
    cuda_push_array(net->input_gpu, b.data, net->h * net->w * net->c * net->batch);
    if(net->truth){
        cuda_push_array(net->truth_gpu, net->truth, net->classes*net->batch);
    }
    forward_network_gpu(net, net->input_gpu);
    backward_network_gpu(net, net->input_gpu);
    update_network_gpu(net);
#else
    forward_network(net, b.data);
    backward_network(net, b.data);
    update_network(net);
#endif
    net->seen += net->batch;
    net->correct_num_count += net->batch;
    if(net->correct_num_count > 1000 * net->batch){
        net->correct_num_count = 0;
        net->correct_num = 0;
    }
    net->batch_train += 1;
}

void valid_network(network *net, batch b)
{
    net->truth = b.truth;
#ifdef GPU
    cuda_push_array(net->input_gpu, b.data, net->h * net->w * net->c * net->batch);
    forward_network_gpu(net, net->input_gpu);
#else
    forward_network(net, b.data);
#endif
    net->correct_num_count += net->batch;
}

int get_network_output_size_layer(network *net, int i)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        return layer->out_w * layer->out_h * layer->n;
    } else if(net->layers_type[i] == CONNECTED){
        connected_layer *layer = (connected_layer *)net->layers[i];
        return layer->outputs;
    } else if(net->layers_type[i] == ROUTE){
        route_layer *layer = (route_layer *)net->layers[i];
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

void save_convolutional_weights(const convolutional_layer *l, FILE *fp, int gpu_index)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    fwrite(l->biases, sizeof(float), l->n, fp);
    if (l->batch_normalize){
        fwrite(l->rolling_mean, sizeof(float), l->n, fp);
        fwrite(l->rolling_variance, sizeof(float), l->n, fp);
    }
    fwrite(l->weights, sizeof(float), l->n * l->size* l->size * l->c, fp);
}

void load_convolutional_weights(const convolutional_layer *l, FILE *fp, int gpu_index)
{
    fread(l->biases, sizeof(float), l->n, fp);
    if (l->batch_normalize){
		fread(l->rolling_mean, sizeof(float), l->n, fp);
		fread(l->rolling_variance, sizeof(float), l->n, fp);
	}
    fread(l->weights, sizeof(float), l->n * l->size* l->size * l->c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void save_connected_weights(const connected_layer *l, FILE *fp, int gpu_index)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l->biases, sizeof(float), l->outputs, fp);
    fwrite(l->weights, sizeof(float), l->outputs*l->inputs, fp);
}

void load_connected_weights(const connected_layer *l, FILE *fp, int gpu_index)
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
            save_convolutional_weights((convolutional_layer *)net->layers[i], fp, net->gpu_index);
        } if(net->layers_type[i] == CONNECTED){
            save_connected_weights((connected_layer *)net->layers[i], fp, net->gpu_index);
        }
    }
    fprintf(stderr, "Saving weights Done!\n\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
#ifdef GPU
    if(net->gpu_index >= 0){
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
            load_convolutional_weights((convolutional_layer *)net->layers[i], fp, net->gpu_index);
        } if(net->layers_type[i] == CONNECTED){
            load_connected_weights((connected_layer *)net->layers[i], fp, net->gpu_index);
        }
    }
    fprintf(stderr, "Loading weights Done!\n");
    fclose(fp);
}

