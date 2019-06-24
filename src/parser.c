#include "parser.h"
#include <assert.h>

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
    char *activation_s = option_find_str(options, "activation", "linear");
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
    float lr_mult = option_find_float(options, "lr_mult", 1);
    float lr_decay_mult = option_find_float(options, "lr_decay_mult", 1);
    float bias_mult = option_find_float(options, "bias_mult", 1);
    float bias_decay_mult = option_find_float(options, "bias_decay_mult", 0);
    char *weight_filler_str = option_find_str(options, "weight_filler", "xavier");
    int weight_filler = 1;
    if(strcmp(weight_filler_str, "xavier") == 0){
        weight_filler = 1;
    } else if(strcmp(weight_filler_str, "gaussian") == 0){
        weight_filler = 2;
    } else{
        weight_filler = 1;
    }
    float sigma = option_find_float(options, "weight_filler_std", 1);
    convolutional_layer *layer = make_convolutional_layer(h, w, c, n, size, stride, net->batch, activation,
                                                          &(net->workspace_size), batch_normalize, pad,
                                                          lr_mult, lr_decay_mult, bias_mult, bias_decay_mult,
                                                          weight_filler, sigma, net->subdivisions, net->test);
    return layer;
}

batchnorm_layer *parse_batchnorm(struct list *options, network *net, int count)
{
    int h,w,c;
    if (count == 0) {
        h = net->h;
        w = net->w;
        c = net->c;
    } else {
        image m = get_network_image_layer(net, count - 1);
        h = m.h;
        w = m.w;
        c = m.c;
        if (h == 0) error("Layer before batchnorm layer must output image.");
    }
    batchnorm_layer *layer = make_batchnorm_layer(net->batch, net->subdivisions, w, h, c, net->test);
    return layer;
}

rnn_layer *parse_rnn(struct list *options, network *net, int count)
{
    int outputs = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int(options, "batch_normalize", 0);
    int inputs = 0;
    if(count == 0){
        inputs = net->inputs;
    }else{
        inputs = get_network_output_size_layer(net, count-1);
    }
    rnn_layer *l = make_rnn_layer(net->batch, inputs, outputs, net->time_steps, activation, batch_normalize);
    return l;
}

lstm_layer *parse_lstm(struct list *options, network *net, int count)
{
    int outputs = option_find_int(options, "output",1);
    int batch_normalize = option_find_int(options, "batch_normalize", 0);
    int inputs = 0;
    if(count == 0){
        inputs = net->inputs;
    }else{
        inputs = get_network_output_size_layer(net, count-1);
    }
    lstm_layer *l = make_lstm_layer(net->batch, inputs, outputs, net->time_steps, batch_normalize);
    return l;
}

gru_layer *parse_gru(struct list *options, network *net, int count)
{
    int outputs = option_find_int(options, "output",1);
    int batch_normalize = option_find_int(options, "batch_normalize", 0);
    int inputs = 0;
    if(count == 0){
        inputs = net->inputs;
    }else{
        inputs = get_network_output_size_layer(net, count-1);
    }
    gru_layer *l = make_gru_layer(net->batch, inputs, outputs, net->time_steps, batch_normalize);
    return l;
}

connected_layer *parse_connected(struct list *options, network *net, int count)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    int output = option_find_int(options, "output",1);
    int input = 0;
    if(count == 0){
        input = option_find_int(options, "input",1);
    }else{
        input = get_network_output_size_layer(net, count-1);
    }
    int weight_normalize =option_find_int(options, "weight_normalize", 0);
    int bias_term = option_find_int(options, "bias_term", 1);

    float lr_mult = option_find_float(options, "lr_mult", 1);
    float lr_decay_mult = option_find_float(options, "lr_decay_mult", 0);
    float bias_mult = option_find_float(options, "bias_mult", 2);
    float bias_decay_mult = option_find_float(options, "bias_decay_mult", 0);
    char *weight_filler_str = option_find_str(options, "weight_filler", "xavier");
    int weight_filler = 1;
    if(strcmp(weight_filler_str, "xavier") == 0){
        weight_filler = 1;
    } else if(strcmp(weight_filler_str, "gaussian") == 0){
        weight_filler = 2;
    } else{
        weight_filler = 1;
    }
    float sigma = option_find_float(options, "weight_filler_std", 1);
    int batch_normalize = option_find_int(options, "batch_normalize", 0);
    connected_layer *layer = make_connected_layer(input, output, net->batch, activation, weight_normalize, bias_term,
                                                  lr_mult, lr_decay_mult, bias_mult, bias_decay_mult, weight_filler,
                                                  sigma, batch_normalize, net->subdivisions, net->test);
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

    route_layer *layer = make_route_layer(net->batch, n, layers, sizes, net, net->test);
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
        activation, prev_layer_weight, shortcut_layer_weight, net->test);
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
    int size = option_find_int(options, "size", stride);
    //int padding = option_find_int(options, "padding", size-1);
    int padding = option_find_int(options, "pad", 0);
    maxpool_layer *layer = make_maxpool_layer(h,w,c,size,stride,net->batch,padding, net->test);
    return layer;
}

upsample_layer *parse_upsample(struct list *options, network *net, int count)
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
        if(h == 0) error("Layer before upsample layer must output image.");
    }
    upsample_layer *layer = make_upsample_layer(net->batch, w, h, c, stride, net->test);
    return layer;
}

int *parse_yolo_mask(const char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}

yolo_layer *make_yolo_snpe(int c, int h, int w, const char *mask_str, const char *anchors)
{
    int count = 0;
    int total = 6;
    int batch = 1;
    int classes = 1;
    int num = 0;
    int *mask = parse_yolo_mask(mask_str, &num);
    yolo_layer *l = make_yolo_layer(batch, w, h, num, total, mask, classes, count);
    assert(l->outputs == w * h * c);

    l->ignore_thresh = 0.7;
    l->truth_thresh = 1;
    if(anchors){
        int len = strlen(anchors);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (anchors[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(anchors);
            l->biases[i] = bias;
            anchors = strchr(anchors, ',')+1;
        }
    }
    return l;
}

yolo_layer *parse_yolo(struct list *options, network *net, int count)
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
        if(h == 0) error("Layer before yolo layer must output image.");
    }
    int total = option_find_int(options, "total", 1);
    int num = 0;
    char *mask_str = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(mask_str, &num);
    yolo_layer *l = make_yolo_layer(net->batch, w, h, num, total, mask, net->classes, count);
    assert(l->outputs == w * h * c);

    l->ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l->truth_thresh = option_find_float(options, "truth_thresh", 1);
    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l->biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
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
    normalize_layer *layer = make_normalize_layer(h,w,c,net->batch, net->test);
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

enum COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "sse")==0) return SSE;
    if (strcmp(s, "masked")==0) return MASKED;
    if (strcmp(s, "smooth")==0) return SMOOTH;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
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

struct list *read_cfg(const char *filename)
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
                //printf("here 0 %p %s\n", current, current->type);
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
                //printf("here 0 %p %s %s\n", current->options, ((kvp *)current->options->front->val)->key, ((kvp *)current->options->front->val)->val);
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
    net->output_layer = option_find_float(options, "output_layer", net->n - 1);
    net->classes = option_find_int(options, "classes", 0);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    net->w = option_find_int(options, "width", 0);
    net->h = option_find_int(options, "height", 0);
    net->c = option_find_int(options, "channels", 0);
    net->inputs = option_find_int(options, "inputs", 0);    // for rnn layer
    if((net->w == 0 || net->h == 0 || net->c == 0) && net->inputs == 0) {
        fprintf(stderr, "parse_net_options: network input size error!\n");
        exit(-1);
    }
    net->saturation = option_find_float(options, "saturation", 1);
    net->exposure = option_find_float(options, "exposure", 1);
    net->hue = option_find_float(options, "hue", 0);
    net->jitter = option_find_float(options, "jitter", 0);
    net->flip = option_find_float(options, "flip", 0);
    net->mean_value = option_find_float(options, "mean_value", 0);
    //net->mean_value /= 255.0F;  // scale image to [0, 1] when load image
    net->scale = option_find_float(options, "scale", 0);

    net->max_batches = option_find_int(options, "max_batches", 0);
    net->max_epoch = option_find_int(options, "max_epoch", 0);
    net->batch = option_find_int(options, "batch", 0);
    net->subdivisions = option_find_int(options, "subdivisions", 1);
    net->batch /= net->subdivisions;
    net->accuracy_count_max = option_find_int(options, "accuracy_count_max", 2000);
    net->time_steps = option_find_int(options, "time_steps", 1);
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
        net->learning_rate_poly_power = option_find_float(options, "learning_rate_poly_power", 4.0);
    }
    net->learning_rate_init = net->learning_rate;
}

network *parse_network_cfg(const char *filename, int test)
{
    struct list *sections = read_cfg(filename);
    network *net = make_network(sections->size - 1);
    net->test = test;
    struct node *n = sections->front;
    struct section *s = (struct section *)n->val;
    if(!(strcmp(s->type, "[network]")==0)) error("First section must be [network]");
    struct list *options = s->options;
    parse_net_options(options, net);
    free_section(s);

    float total_bflop = 0;
    n = n->next;
    int count = 0;
    fprintf(stderr, "layer                    input                 filters                          output\n");
    while(n){
        struct section *s = (struct section *)n->val;
        struct list *options = s->options;
        fprintf(stderr, "%3d: ", count);
        if(strcmp(s->type, "[convolutional]")==0){
            convolutional_layer *layer = parse_convolutional(options, net, count);
            total_bflop += layer->bflop;
            net->layers_type[count] = CONVOLUTIONAL;
            net->layers[count] = layer;
        } else if(strcmp(s->type, "[batchnorm]")==0){
            batchnorm_layer *layer = parse_batchnorm(options, net, count);
            net->layers_type[count] = BATCHNORM;
            net->layers[count] = layer;
        } else if(strcmp(s->type, "[connected]")==0){
            connected_layer *layer = parse_connected(options, net, count);
            net->layers_type[count] = CONNECTED;
            net->layers[count] = layer;
        } else if(strcmp(s->type, "[rnn]")==0){
            rnn_layer *layer = parse_rnn(options, net, count);
            net->layers_type[count] = RNN;
            net->layers[count] = layer;
        } else if(strcmp(s->type, "[lstm]")==0){
            lstm_layer *layer = parse_lstm(options, net, count);
            net->layers_type[count] = LSTM;
            net->layers[count] = layer;
        } else if(strcmp(s->type, "[gru]")==0){
            gru_layer *layer = parse_gru(options, net, count);
            net->layers_type[count] = GRU;
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
        }else if(strcmp(s->type, "[upsample]")==0){
            upsample_layer *layer = parse_upsample(options, net, count);
            net->layers_type[count] = UPSAMPLE;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[yolo]")==0){
            yolo_layer *layer = parse_yolo(options, net, count);
            net->layers_type[count] = YOLO;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[normalize]")==0){
            normalize_layer *layer = parse_normalize(options, net, count);
            net->layers_type[count] = NORMALIZE;
            net->layers[count] = layer;
        }else if(strcmp(s->type, "[dropout]")==0){
            dropout_layer *layer = parse_dropout(options, net, count);
            layer->output = get_network_layer_data(net, count - 1, 0, 0);  // reuse previous layer output and delta
            layer->delta = get_network_layer_data(net, count - 1, 1, 0);
#ifdef GPU
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
            fprintf(stderr, "parse_network_cfg: layer type not recognized: %s\n", s->type);
            exit(-1);
        }
        option_unused(options);
        free_section(s);
        ++count;
        n = n->next;
    }

    net->input = (float *)malloc(net->h * net->w * net->c * net->batch * sizeof(float));
    net->max_boxes = 30;
    net->truth = calloc(1, net->max_boxes * 5 * net->batch * sizeof(float));
#ifdef GPU
    if(net->w == 0 || net->h == 0 || net->c == 0) {
        net->input_gpu = cuda_make_array(0, net->time_steps * net->batch * net->inputs);
    } else {
        net->input_gpu = cuda_make_array(0, net->h * net->w * net->c * net->batch);
    }
    net->truth_gpu = cuda_make_array(0, net->max_boxes * 5 * net->batch);
    net->truth_label_index_gpu = cuda_make_int_array(0, net->batch);
    net->is_not_max_gpu = cuda_make_int_array(0, net->batch);
    net->gpu_index = cuda_get_device();
#elif defined(OPENCL)
    net->input_cl = cl_make_array(0, net->h * net->w * net->c * net->batch);
    net->truth_cl = cl_make_array(0, net->max_boxes * 5 * net->batch);
    net->truth_label_index_cl = cl_make_int_array(0, net->batch);
    net->is_not_max_cl = cl_make_int_array(0, net->batch);
    net->gpu_index = -1;
#endif
    if(net->workspace_size){
#ifdef GPU
        if(net->gpu_index >= 0){
            net->workspace_gpu = cuda_make_array(0, net->workspace_size / sizeof(float));
        }else {
            printf("net->gpu_index < 0!\n");
            exit(-1);
        }
#elif defined(OPENCL)
        //net->workspace_cl = cl_make_share_array(0, net->workspace_size / sizeof(float));
        net->workspace_cl = cl_make_array(0, net->workspace_size / sizeof(float));
#endif
        net->workspace = calloc(1, net->workspace_size);
    }
    if(net->workspace_gpu){
        //printf("net->workspace_gpu is not null, calloc for net->workspace just for test!!!\n\n\n");
        //net->workspace = calloc(1, net->workspace_size);
    }
    /*
    struct node *start_node = sections->front;
    struct node *next;
    while(start_node) {
        next = start_node->next;

        printf("here 0\n");
        printf("here 0 %p\n", start_node->val);
        struct section *section_local = start_node->val;
        printf("%s\n", section_local->type);
        free(section_local->type);
        struct node *start_node_inner = section_local->options->front;
        struct node *next_inner;
        while(start_node_inner) {
            next_inner = start_node_inner->next;
            kvp *kvp_val = start_node_inner->val;
            printf("%s %s\n", kvp_val->key, kvp_val->val);
            free(kvp_val->key);
            free(start_node_inner);
            start_node_inner = next_inner;
            printf("well %p\n", start_node_inner);
        }
        free(section_local->options);
        free(start_node);
        printf("here 1\n");
        start_node = next;
        printf("here 2 %p\n", next);
    }
    free(sections);
    */
    free_list(sections);
    fprintf(stderr, "\nnetwork total_bflop: %5.3f BFLOPs\n", total_bflop);;
    return net;
}
