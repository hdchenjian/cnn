#include "network.h"
network *parse_network_cfg(char *filename);

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->seen = 0;
    net->test = 0;
    net->batch_train = 0;
    net->epoch = 0;
    net->correct_num = 0;
    net->accuracy_count_max = 0;
    net->gpu_index = -1;

    net->layers = calloc(net->n, sizeof(void *));
    net->layers_type = calloc(net->n, sizeof(enum LAYER_TYPE));
    net->workspace_size = 0;
    net->workspace = NULL;
#ifdef GPU
    net->workspace_gpu = NULL;
    net->input_gpu = NULL;
#endif
    return net;
}

network *load_network(char *cfg, char *weights)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    return net;
}

void free_network(network *net)
{
    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            free_convolutional_layer(net->layers[i]);
        } else if(net->layers_type[i] == CONNECTED){
            free_connected_layer(net->layers[i]);
        } else if(net->layers_type[i] == RNN){
            free_rnn_layer(net->layers[i]);
        } else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
#ifdef GPU
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#endif
            free_ptr(layer);
        } else if(net->layers_type[i] == SHORTCUT){
            shortcut_layer *layer = (shortcut_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
#ifdef GPU
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#endif
            free_ptr(layer);
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
            if(layer->indexes) free_ptr(layer->indexes);
#ifdef GPU
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            if(layer->indexes_gpu) cuda_free(layer->indexes_gpu);
#endif
            free_ptr(layer);
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
#ifdef GPU
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#endif
            free_ptr(layer);
        } else if(net->layers_type[i] == NORMALIZE){
            normalize_layer *layer = (normalize_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
#ifdef GPU
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#endif
            free_ptr(layer);
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            if(layer->rand) free_ptr(layer->rand);
#ifdef GPU
            if(layer->rand_gpu) cuda_free(layer->rand_gpu);
#endif
            free_ptr(layer);
        } else if(net->layers_type[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
            if(layer->loss) free_ptr(layer->loss);
            if(layer->cost) free_ptr(layer->cost);
#ifdef GPU
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
            if(layer->loss_gpu) cuda_free(layer->loss_gpu);
#endif
            free_ptr(layer);
        } else if(net->layers_type[i] == COST){
            cost_layer *layer = (cost_layer *)net->layers[i];
            if(layer->output) free_ptr(layer->output);
            if(layer->delta) free_ptr(layer->delta);
            if(layer->cost) free_ptr(layer->cost);
#ifdef GPU
            if(layer->output_gpu) cuda_free(layer->output_gpu);
            if(layer->delta_gpu) cuda_free(layer->delta_gpu);
#endif
            free_ptr(layer);
        } else {
            printf("free_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
    free_ptr(net->layers);
    free_ptr(net->layers_type);
    if(net->workspace) free_ptr(net->workspace);
#ifdef GPU
    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_label_index_gpu) cuda_free(net->truth_label_index_gpu);
    if(net->is_not_max_gpu) cuda_free(net->is_not_max_gpu);
    if(net->workspace_gpu) cuda_free(net->workspace_gpu);
#endif

    free_ptr(net);
}

float update_current_learning_rate(network *net)
{
    switch (net->policy) {
        case STEPS:
            for(int i = 0; i < net->num_steps; ++i){
                if(net->steps[i] == net->batch_train){
                    net->learning_rate *= net->scales[i];
                }
            }
            if(net->learning_rate_init == net->learning_rate){
                for(int i = 0; i < net->num_steps; ++i){
                    if(net->batch_train >= net->steps[i]){
                        net->learning_rate *= net->scales[i];
                    }
                }
            }
            return net->learning_rate;;
        case POLY:
            net->learning_rate = net->learning_rate_init *
                pow(1 - (float)net->batch_train / (float)net->max_batches, net->learning_rate_poly_power);
            return net->learning_rate;
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

void forward_network(network *net, float *input)
{
    for(int i = 0; i < net->n && i <= net->output_layer; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            //memset(net->workspace, 0, net->workspace_size);
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            forward_convolutional_layer(layer, input, net->workspace, net->test);
            /*
            if(i == 0){
                int num = 5;
                for(int b = 0; b < num; ++b){
                    printf("%d %f\n", b, input[b]);
                }
            }
            if(i == 0){
                int num = 5;
                for(int b = 0; b < num; ++b){
                    printf("%d %f\n", b, layer->weights[b]);
                }
            }
            */
            
            input = layer->output;
            /*if(i == 0){
                int num = 5;
                for(int b = 0; b < num; ++b){
                    printf("%d %f\n", b, input[b]);
                }
            }
            */
            
        }else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            forward_connected_layer(layer, input, net->test);
            input = layer->output;
        }else if(net->layers_type[i] == RNN){
            rnn_layer *layer = (rnn_layer *)net->layers[i];
            forward_rnn_layer(layer, input, net->test);
            input = layer->output;
        }else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            forward_route_layer(layer, net);
            input = layer->output;
        }else if(net->layers_type[i] == SHORTCUT){
            shortcut_layer *layer = (shortcut_layer *)net->layers[i];
            forward_shortcut_layer(layer, input, net);
            input = layer->output;
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            forward_maxpool_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            forward_avgpool_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == NORMALIZE){
            normalize_layer *layer = (normalize_layer *)net->layers[i];
            forward_normalize_layer(layer, input);
            input = layer->output;
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            forward_dropout_layer(layer, input, net->test);
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
            /*
            if(i == 0){
                int num = 5;
                for(int b = 0; b < num; ++b){
                    printf("weight: %d %f\n", b, layer->weight_updates[b]);
                }
            }
            */
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            update_connected_layer(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == RNN){
            rnn_layer *layer = (rnn_layer *)net->layers[i];
            update_rnn_layer(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == ROUTE){
        } else if(net->layers_type[i] == SHORTCUT){
        } else if(net->layers_type[i] == MAXPOOL){
        } else if(net->layers_type[i] == DROPOUT){
        } else if(net->layers_type[i] == AVGPOOL){
        } else if(net->layers_type[i] == NORMALIZE){
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
    } else if(net->layers_type[i] == RNN){
        rnn_layer *layer = (rnn_layer *)net->layers[i];
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
    } else if(net->layers_type[i] == SHORTCUT){
        shortcut_layer *layer = (shortcut_layer *)net->layers[i];
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
    } else if(net->layers_type[i] == NORMALIZE){
        normalize_layer *layer = (normalize_layer *)net->layers[i];
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
            //memset(net->workspace, 0, net->workspace_size);
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            backward_convolutional_layer(layer, prev_input, prev_delta, net->workspace, net->test);

            /*
            if(i == 0){
                int num = 5;
                for(int b = 0; b < num; ++b){
                    printf("weight_updates: %d %.19ff\n", b, layer->weight_updates[b]);
                }
            }
            */
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            memset(prev_delta, 0, layer->batch * layer->inputs * sizeof(float));
            backward_connected_layer(layer, prev_input, prev_delta, net->test);
        } else if(net->layers_type[i] == RNN){
            rnn_layer *layer = (rnn_layer *)net->layers[i];
            backward_rnn_layer(layer, prev_input, prev_delta, net->test);
        } else if(net->layers_type[i] == LSTM){
            rnn_layer *layer = (rnn_layer *)net->layers[i];
            memset(prev_delta, 0, layer->batch * layer->inputs * sizeof(float));  // todo
            backward_rnn_layer(layer, prev_input, prev_delta, net->test);
        } else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            backward_route_layer(layer, net);
        } else if(net->layers_type[i] == SHORTCUT){
            shortcut_layer *layer = (shortcut_layer *)net->layers[i];
            backward_shortcut_layer(layer, prev_delta, net);
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            if(i != 0) backward_maxpool_layer(layer, prev_input, prev_delta);
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            if(i != 0) backward_avgpool_layer(layer, prev_delta);
        } else if(net->layers_type[i] == NORMALIZE){
            normalize_layer *layer = (normalize_layer *)net->layers[i];
            if(i != 0) backward_normalize_layer(layer, prev_delta);
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
    for(int i = 0; i < net->n && i <= net->output_layer; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            //cudaError_t status = cudaMemset(net->workspace_gpu, 0, net->workspace_size);
            //check_error(status);
            forward_convolutional_layer_gpu(layer, input, net->workspace_gpu, net->test);
            input = layer->output_gpu;
            /*
            if(i == 0){
                int num = 5;
                float *input_temp = calloc(num, sizeof(float));
                cuda_pull_array(input, input_temp, num);
                for(int b = 0; b < num; ++b){
                    printf("%d %f\n", b, input_temp[b]);
                }
            }
            */
        }else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            forward_connected_layer_gpu(layer, input, net->test);
            input = layer->output_gpu;
        }else if(net->layers_type[i] == RNN){
            rnn_layer *layer = (rnn_layer *)net->layers[i];
            forward_rnn_layer_gpu(layer, input, net->test);
            input = layer->output_gpu;
        }else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            forward_route_layer_gpu(layer, net);
            input = layer->output_gpu;
        }else if(net->layers_type[i] == SHORTCUT){
            shortcut_layer *layer = (shortcut_layer *)net->layers[i];
            forward_shortcut_layer_gpu(layer, input, net);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            forward_maxpool_layer_gpu(layer, input);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            forward_avgpool_layer_gpu(layer, input);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == NORMALIZE){
            normalize_layer *layer = (normalize_layer *)net->layers[i];
            forward_normalize_layer_gpu(layer, input);
            input = layer->output_gpu;
        } else if(net->layers_type[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net->layers[i];
            forward_dropout_layer_gpu(layer, input, net->test);
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
            printf("forward_network_gpu layers_type error, layer: %d\n", i);
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
            //cudaError_t status = cudaMemset(net->workspace_gpu, 0, net->workspace_size);
            //check_error(status);
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            backward_convolutional_layer_gpu(layer, prev_input, prev_delta, net->workspace_gpu, net->test);
        } else if(net->layers_type[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            backward_connected_layer_gpu(layer, prev_input, prev_delta, net->test);
        } else if(net->layers_type[i] == RNN){
            rnn_layer *layer = (rnn_layer *)net->layers[i];
            backward_rnn_layer_gpu(layer, prev_input, prev_delta, net->test);
        } else if(net->layers_type[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            backward_route_layer_gpu(layer, net);
        } else if(net->layers_type[i] == SHORTCUT){
            shortcut_layer *layer = (shortcut_layer *)net->layers[i];
            backward_shortcut_layer_gpu(layer, prev_delta, net);
        } else if(net->layers_type[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            if(i != 0) backward_maxpool_layer_gpu(layer, prev_delta);
        } else if(net->layers_type[i] == AVGPOOL){
            avgpool_layer *layer = (avgpool_layer *)net->layers[i];
            if(i != 0) backward_avgpool_layer_gpu(layer, prev_delta);
        } else if(net->layers_type[i] == NORMALIZE){
            normalize_layer *layer = (normalize_layer *)net->layers[i];
            if(i != 0) backward_normalize_layer_gpu(layer, prev_delta);
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
        } else if(net->layers_type[i] == RNN){
            rnn_layer *layer = (rnn_layer *)net->layers[i];
            update_rnn_layer_gpu(layer, net->learning_rate, net->momentum, net->decay);
        } else if(net->layers_type[i] == ROUTE){
        } else if(net->layers_type[i] == SHORTCUT){
        } else if(net->layers_type[i] == MAXPOOL){
        } else if(net->layers_type[i] == DROPOUT){
        } else if(net->layers_type[i] == AVGPOOL){
        } else if(net->layers_type[i] == NORMALIZE){
        } else if(net->layers_type[i] == SOFTMAX){
        } else if(net->layers_type[i] == COST){
        } else {
            printf("update_network layers_type error, layer: %d\n", i);
            exit(-1);
        }
    }
}

#endif

void train_network(network *net, float *input, int *truth_label_index)
{
    if(net->accuracy_count > net->accuracy_count_max){
        net->accuracy_count = 0;
        net->correct_num = 0;
    }

    net->truth_label_index = truth_label_index;
#ifdef GPU
    if(net->w == 0 || net->h == 0 || net->c == 0) {
        cuda_push_array(net->input_gpu, input, net->time_steps * net->batch * net->inputs);
    } else {
        cuda_push_array(net->input_gpu, input, net->h * net->w * net->c * net->batch);
    }
    cuda_push_array_int(net->truth_label_index_gpu, net->truth_label_index, net->batch);
    //forward_network(net, input);
    //backward_network(net, input);
    forward_network_gpu(net, net->input_gpu);
    backward_network_gpu(net, net->input_gpu);
    update_network_gpu(net);
#else
    forward_network(net, input);
    backward_network(net, input);
    update_network(net);
#endif
    net->seen += net->batch * net->time_steps;
    net->accuracy_count += net->batch;
    net->batch_train += 1;
}

void valid_network(network *net, float *input, int *truth_label_index)
{
    net->truth_label_index = truth_label_index;
#ifdef GPU
    if(net->w == 0 || net->h == 0 || net->c == 0) {
        cuda_push_array(net->input_gpu, input, net->time_steps * net->batch * net->inputs);
    } else {
        cuda_push_array(net->input_gpu, input, net->h * net->w * net->c * net->batch);
    }
    cuda_push_array_int(net->truth_label_index_gpu, net->truth_label_index, net->batch);
    forward_network_gpu(net, net->input_gpu);
#else
    forward_network(net, input);
#endif
    net->accuracy_count += net->batch;
}

float *forward_network_test(network *net, float *input)
{
#ifdef GPU
    if(net->w == 0 || net->h == 0 || net->c == 0) {
        cuda_push_array(net->input_gpu, input, net->time_steps * net->batch * net->inputs);
    } else {
        cuda_push_array(net->input_gpu, input, net->h * net->w * net->c * net->batch);
    }
    forward_network_gpu(net, net->input_gpu);
#else
    forward_network(net, input);
#endif

#ifndef GPU
    float *network_output = get_network_layer_data(net, net->output_layer, 0, 0);
#else
    int network_output_size = get_network_output_size_layer(net, net->output_layer);
    float *network_output_gpu = get_network_layer_data(net, net->output_layer, 0, 1);
    float *network_output = malloc(network_output_size * sizeof(float));
    cuda_pull_array(network_output_gpu, network_output, network_output_size);
#endif
    return network_output;
}

int get_network_output_size_layer(network *net, int i)
{
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        return layer->out_w * layer->out_h * layer->n;
    } else if(net->layers_type[i] == CONNECTED){
        connected_layer *layer = (connected_layer *)net->layers[i];
        return layer->outputs;
    } else if(net->layers_type[i] == RNN){
        rnn_layer *layer = (rnn_layer *)net->layers[i];
        return layer->outputs;
    } else if(net->layers_type[i] == ROUTE){
        route_layer *layer = (route_layer *)net->layers[i];
        return layer->outputs;
    } else if(net->layers_type[i] == SHORTCUT){
        shortcut_layer *layer = (shortcut_layer *)net->layers[i];
        return layer->outputs;
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        image output = get_maxpool_image(layer);
        return output.h*output.w*output.c;
    }else if(net->layers_type[i] == AVGPOOL){
        avgpool_layer *layer = (avgpool_layer *)net->layers[i];
        return layer->c;
    }else if(net->layers_type[i] == NORMALIZE){
        normalize_layer *layer = (normalize_layer *)net->layers[i];
        return layer->w * layer->h * layer->c;
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
    if(net->layers_type[i] == CONVOLUTIONAL){
        convolutional_layer *layer = (convolutional_layer *)net->layers[i];
        return get_convolutional_image(layer);
    } else if(net->layers_type[i] == MAXPOOL){
        maxpool_layer *layer = (maxpool_layer *)net->layers[i];
        return get_maxpool_image(layer);
    } else if(net->layers_type[i] == DROPOUT){
        dropout_layer *layer = (dropout_layer *)net->layers[i];
        return get_dropout_image(layer);
    } else if(net->layers_type[i] == SHORTCUT){
        shortcut_layer *layer = (shortcut_layer *)net->layers[i];
        return get_shortcut_image(layer);
    } else if(net->layers_type[i] == ROUTE){
        route_layer *layer = (route_layer *)net->layers[i];
        return get_route_image(layer);
    } else if(net->layers_type[i] == AVGPOOL){
        avgpool_layer *layer = (avgpool_layer *)net->layers[i];
        return get_avgpool_image(layer);
    } else if(net->layers_type[i] == CONNECTED){
        connected_layer *layer = (connected_layer *)net->layers[i];
        return get_connected_image(layer);
    } else if(net->layers_type[i] == RNN){
        rnn_layer *layer = (rnn_layer *)net->layers[i];
        return get_rnn_image(layer);
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
        fwrite(l->scales, sizeof(float), l->n, fp);
        fwrite(l->rolling_mean, sizeof(float), l->n, fp);
        fwrite(l->rolling_variance, sizeof(float), l->n, fp);
    }
    fwrite(l->weights, sizeof(float), l->n * l->size* l->size * l->c, fp);
}

void load_convolutional_weights(const convolutional_layer *l, FILE *fp, int gpu_index)
{
    fread(l->biases, sizeof(float), l->n, fp);
    if (l->batch_normalize){
        fread(l->scales, sizeof(float), l->n, fp);
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
    if(l->batch_normalize){
        fwrite(l->scales, sizeof(float), l->outputs, fp);
        fwrite(l->rolling_mean, sizeof(float), l->outputs, fp);
        fwrite(l->rolling_variance, sizeof(float), l->outputs, fp);
    }
}

void load_connected_weights(const connected_layer *l, FILE *fp, int gpu_index)
{
    fread(l->biases, sizeof(float), l->outputs, fp);
    fread(l->weights, sizeof(float), l->outputs*l->inputs, fp);
    if(l->batch_normalize){
        fread(l->scales, sizeof(float), l->outputs, fp);
        fread(l->rolling_mean, sizeof(float), l->outputs, fp);
        fread(l->rolling_variance, sizeof(float), l->outputs, fp);
    }
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
    fprintf(stderr, "weights version info: major: %d, minor: %d, revision: %d, net->seen: %lu\n",
            major, minor, revision, net->seen);

    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(&(net->seen), sizeof(size_t), 1, fp);

    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            save_convolutional_weights((convolutional_layer *)net->layers[i], fp, net->gpu_index);
        } else if(net->layers_type[i] == CONNECTED){
            save_connected_weights((connected_layer *)net->layers[i], fp, net->gpu_index);
        } else if(net->layers_type[i] == RNN){
            save_connected_weights((connected_layer *)((rnn_layer *)net->layers[i])->input_layer, fp, net->gpu_index);
            save_connected_weights((connected_layer *)((rnn_layer *)net->layers[i])->self_layer, fp, net->gpu_index);
            save_connected_weights((connected_layer *)((rnn_layer *)net->layers[i])->output_layer, fp, net->gpu_index);
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
    fprintf(stderr, "weights version info: major: %d, minor: %d, revision: %d, net->seen: %lu\n",
            major, minor, revision, net->seen);

    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == CONVOLUTIONAL){
            load_convolutional_weights((convolutional_layer *)net->layers[i], fp, net->gpu_index);
        } else if(net->layers_type[i] == CONNECTED){
            load_connected_weights((connected_layer *)net->layers[i], fp, net->gpu_index);
        } else if(net->layers_type[i] == RNN){
            load_connected_weights((connected_layer *)((rnn_layer *)net->layers[i])->input_layer, fp, net->gpu_index);
            load_connected_weights((connected_layer *)((rnn_layer *)net->layers[i])->self_layer, fp, net->gpu_index);
            load_connected_weights((connected_layer *)((rnn_layer *)net->layers[i])->output_layer, fp, net->gpu_index);
        }
    }
    fprintf(stderr, "Loading weights Done!\n");
    fclose(fp);
}

void reset_rnn_state(network *net, int b)
{
    for(int i = 0; i < net->n; ++i){
        if(net->layers_type[i] == RNN){
            rnn_layer *layer = (rnn_layer *)net->layers[i];
            fill_cpu(layer->outputs, 0, layer->state + layer->outputs*b, 1);
#ifdef GPU
            fill_gpu(layer->outputs, 0, layer->state_gpu + layer->outputs*b, 1);
#endif
        }
    }
}
