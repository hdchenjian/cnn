#include "parser.h"

struct section{
    char *type;
    struct list *options;
};

void free_section(struct section *s)
{
    free(s->type);
    struct node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        struct node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

convolutional_layer *parse_convolutional(struct list *options, network *net, int count)
{
    int h,w,c;
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    char *activation_s = option_find_str(options, "activation", "sigmoid");
    ACTIVATION activation = get_activation(activation_s);
    if (count == 0) {
        h = net->h;
        w = net->w;
        c = net->c;
    } else {
        image m = get_network_image_layer(net, count - 1);
        h = m.h;
        w = m.w;
        c = m.c;
        if (h == 0) error("Layer before convolutional layer must output image.");
    }
    int batch_normalize = option_find_int(options, "batch_normalize", 0);
    int pad = option_find_int(options, "pad", 0);
    convolutional_layer *layer = make_convolutional_layer(
        h, w, c, n, size, stride, net->batch, activation, &(net->workspace_size), batch_normalize, pad);
    return layer;
}

connected_layer *parse_connected(struct list *options, network *net, int count)
{
    char *activation_s = option_find_str(options, "activation", "sigmoid");
    ACTIVATION activation = get_activation(activation_s);
    int output = option_find_int(options, "output",1);
    int input;
    if(count == 0){
        input = option_find_int(options, "input",1);
    }else{
        input = get_network_output_size_layer(net, count-1);
    }
    int weight_normalize =option_find_int(options, "weight_normalize", 0);
    int bias_term = option_find_int(options, "bias_term", 1);
    connected_layer *layer = make_connected_layer(input, output, net->batch, activation, weight_normalize, bias_term);
    return layer;
}

route_layer *parse_route(struct list *options, network *net, int count)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    for(int i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }
    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(int i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = count + index;
        layers[i] = index;
        sizes[i] = get_network_output_size_layer(net, index);
    }

    route_layer *layer = make_route_layer(net->batch, n, layers, sizes, net);
    return layer;
}

shortcut_layer *parse_shortcut(struct list *options, network *net, int count)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if(index < 0) index = count + index;
    image shortcut_layer_output_image = get_network_image_layer(net, index);
    image previous_layer_output_image = get_network_image_layer(net, count - 1);
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    float prev_layer_weight = option_find_float(options, "prev_layer_weight", 1);
    float shortcut_layer_weight = option_find_float(options, "shortcut_layer_weight", 1);
    shortcut_layer *layer = make_shortcut_layer(
        net->batch, index, shortcut_layer_output_image.w, shortcut_layer_output_image.h,
        shortcut_layer_output_image.c,
        previous_layer_output_image.w, previous_layer_output_image.h, previous_layer_output_image.c,
        activation, prev_layer_weight, shortcut_layer_weight);
    return layer;
}

maxpool_layer *parse_maxpool(struct list *options, network *net, int count)
{
    int h,w,c;
    int stride = option_find_int(options, "stride",1);
    if(count == 0){
        h = net->h;
        w = net->w;
        c = net->c;
    }else{
        image m =  get_network_image_layer(net, count-1);
        h = m.h;
        w = m.w;
        c = m.c;
        if(h == 0) error("Layer before maxpool layer must output image.");
    }
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int(options, "padding", (size-1)/2);
    maxpool_layer *layer = make_maxpool_layer(h,w,c,size,stride,net->batch,padding);
    return layer;
}

normalize_layer *parse_normalize(struct list *options, network *net, int count)
{
    int h,w,c;
    if(count == 0){
        h = net->h;
        w = net->w;
        c = net->c;
    }else{
        image m =  get_network_image_layer(net, count-1);
        h = m.h;
        w = m.w;
        c = m.c;
        if(h == 0) error("Layer before normalize layer must output image.");
    }
    normalize_layer *layer = make_normalize_layer(h,w,c,net->batch);
    return layer;
}

avgpool_layer *parse_avgpool(struct list *options, network *net, int count)
{
    int w,h,c;
    image m =  get_network_image_layer(net, count-1);
    w = m.w;
    h = m.h;
    c = m.c;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");
    avgpool_layer *layer = make_avgpool_layer(net->batch,w,h,c);
    return layer;
}

dropout_layer *parse_dropout(struct list *options, network *net, int count)
{
    float probability = option_find_float(options, "probability", .5);
    int w,h,c;
    image m =  get_network_image_layer(net, count-1);
    w = m.w;
    h = m.h;
    c = m.c;
    int input;
    if(count == 0){
        input = option_find_int(options, "input",1);
    }else{
        input = get_network_output_size_layer(net, count-1);
    }
    dropout_layer *layer = make_dropout_layer(w, h, c, net->batch, input, probability);
    return layer;
}

softmax_layer *parse_softmax(struct list *options, network *net, int count, int is_last_layer)
{
    int input;
    if(count == 0){
        input = option_find_int(options, "input",1);
    }else{
        input =  get_network_output_size_layer(net, count-1);
    }
    float label_specific_margin_bias = option_find_float(options, "label_specific_margin_bias", 0);;
    int margin_scale = option_find_int(options, "margin_scale", 0);
    softmax_layer *layer = make_softmax_layer(input ,net->batch, is_last_layer, label_specific_margin_bias, margin_scale);
    return layer;
}

cost_layer *parse_cost(struct list *options, network *net, int count)
{
    char *type_s = option_find_str(options, "type", "sse");
    enum COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float(options, "scale", 1);
    int inputs =  get_network_output_size_layer(net, count-1);
    cost_layer *layer = make_cost_layer(net->batch, inputs, type, scale);
    return layer;
}

int read_option(char *s, struct list *options)
{
    int i;
    int len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

struct list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    struct list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

struct list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0){
        fprintf(stderr, "Couldn't open file: %s\n", filename);
        exit(-1);
    }
    char *line;
    int nu = 0;
    struct list *sections = make_list();
    struct section *current = 0;
    while((line=fgetl(file)) != 0){
        ++nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(struct section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    printf("Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}

enum learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(struct list *options, network *net)
{
    // get output from the net->output_layer layer
    net->output_layer = option_find_float(options, "output_layer", net->n - 1);
    net->classes = option_find_int(options, "classes", 0);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    net->w = option_find_int(options, "width", 0);
    net->h = option_find_int(options, "height", 0);
    net->c = option_find_int(options, "channels", 0);
    if(net->w == 0 || net->h == 0 || net->c == 0) {
        fprintf(stderr, "Input image size error!\n");
        exit(-1);
    }
    net->saturation = option_find_float(options, "saturation", 1);
    net->exposure = option_find_float(options, "exposure", 1);
    net->hue = option_find_float(options, "hue", 0);
    net->flip = option_find_float(options, "flip", 0);
    net->mean_value = option_find_float(options, "mean_value", 1);
    net->mean_value /= 255.0F;  // scale image to [0, 1] when load image
    net->scale = option_find_float(options, "scale", 0);

    net->max_batches = option_find_int(options, "max_batches", 0);
    net->max_epoch = option_find_int(options, "max_epoch", 0);
    net->batch = option_find_int(options, "batch", 0);
    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    if (net->policy == STEPS){
        char *steps_str = option_find(options, "steps");
        char *scales_str = option_find(options, "scales");
        if(!steps_str || !scales_str) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(steps_str);
        int n = 1;
        int scales_n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (steps_str[i] == ',') ++n;
        }
        for(i = 0; i < strlen(scales_str); ++i){
            if (scales_str[i] == ',') ++scales_n;
        }
        if(n != scales_n){
            fprintf(stderr, "error: steps not match scales: %d, %d\n", n, scales_n);
            exit(-1);
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(steps_str);
            float scale = atof(scales_str);
            steps_str = strchr(steps_str, ',')+1;
            scales_str = strchr(scales_str, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if(net->policy == POLY){
        net->learning_rate_poly_power = option_find_int(options, "learning_rate_poly_power", 4);
    }
    net->learning_rate_init = net->learning_rate;
}

network *parse_network_cfg(char *filename)
{
    struct list *sections = read_cfg(filename);
    network *net = make_network(sections->size - 1);
    struct node *n = sections->front;
    struct section *s = (struct section *)n->val;
    if(!(strcmp(s->type, "[network]")==0)) error("First section must be [network]");
    struct list *options = s->options;
    parse_net_options(options, net);

    float total_bflop = 0;
    n = n->next;
    int count = 0;
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){
        struct section *s = (struct section *)n->val;
        struct list *options = s->options;
        fprintf(stderr, "%3d: ", count);
        if(strcmp(s->type, "[convolutional]")==0){
            convolutional_layer *layer = parse_convolutional(options, net, count);
            total_bflop += layer->bflop;
            net->layers_type[count] = CONVOLUTIONAL;
            net->layers[count] = layer;
        } else if(strcmp(s->type, "[connected]")==0){
            connected_layer *layer = parse_connected(options, net, count);
            net->layers_type[count] = CONNECTED;
            net->layers[count] = layer;
        } else if(strcmp(s->type, "[route]")==0){
            route_layer *layer = parse_route(options, net, count);
            net->layers_type[count] = ROUTE;
            net->layers[count] = layer;
        } else if(strcmp(s->type, "[shortcut]")==0){
            shortcut_layer *layer = parse_shortcut(options, net, count);
            net->layers_type[count] = SHORTCUT;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[softmax]")==0){
            softmax_layer *layer = parse_softmax(options, net, count, sections->size - 1 - 1 == count);
            net->layers_type[count] = SOFTMAX;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[maxpool]")==0){
            maxpool_layer *layer = parse_maxpool(options, net, count);
            net->layers_type[count] = MAXPOOL;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[normalize]")==0){
            normalize_layer *layer = parse_normalize(options, net, count);
            net->layers_type[count] = NORMALIZE;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[dropout]")==0){
            dropout_layer *layer = parse_dropout(options, net, count);
#ifndef GPU
            layer->output = get_network_layer_data(net, count - 1, 0, 0);  // reuse previous layer output and delta
            layer->delta = get_network_layer_data(net, count - 1, 1, 0);
#else
            layer->output_gpu = get_network_layer_data(net, count - 1, 0, 1);
            layer->delta_gpu = get_network_layer_data(net, count - 1, 1, 1);
#endif
            net->layers_type[count] = DROPOUT;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[cost]")==0){
            cost_layer *layer = parse_cost(options, net, count);
            net->layers_type[count] = COST;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[avgpool]")==0){
            avgpool_layer *layer = parse_avgpool(options, net, count);
            net->layers_type[count] = AVGPOOL;
            net->layers[count] = layer;
        }else{
            fprintf(stderr, "layer type not recognized: %s\n", s->type);
            exit(-1);
        }
        option_unused(options);
        free_section(s);
        ++count;
        n = n->next;
    }
#ifdef GPU
    net->input_gpu = cuda_make_array(0, net->h * net->w * net->c * net->batch);
    net->truth_label_index_gpu = cuda_make_int_array(0, net->batch);
    net->is_not_max_gpu = cuda_make_int_array(0, net->batch);
    net->gpu_index = cuda_get_device();
#endif
    if(net->workspace_size){
#ifdef GPU
        if(net->gpu_index >= 0){
            net->workspace_gpu = cuda_make_array(0, (net->workspace_size-1)/sizeof(float)+1);
        }else {
            net->workspace = calloc(1, net->workspace_size);
        }
#else
        net->workspace = calloc(1, net->workspace_size);
#endif
    }
    if(net->workspace_gpu){
        printf("net->workspace_gpu is not null, calloc for net->workspace just for test!!!\n\n\n");
        net->workspace = calloc(1, net->workspace_size);
    }
    free_list(sections);
    fprintf(stderr, "\nnetwork total_bflop: %5.3f BFLOPs\n", total_bflop);;
    return net;
}
