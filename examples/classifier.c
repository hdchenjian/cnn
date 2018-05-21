#include "darknet.h"
#include "utils.h"

#include <sys/time.h>
#include <assert.h>

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("train data base path: %s\n", base);
    printf("the number of GPU: %d\n", ngpus);
    struct network **nets = calloc(ngpus, sizeof(struct network*));

    srand(time(0));
    int seed = rand();
    for(int i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    struct network *net = nets[0];

    int imgs = net->batch * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    struct list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    struct list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("train data size: %d\n", plist->size);
    int train_set_size = plist->size;
    double time;

    struct load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;

    printf("input size: [%d, %d]\n", args.min, args.max);
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = train_set_size;
    args.labels = labels;

    struct data train;
    struct data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;
    int epoch = (*net->seen) / train_set_size;
    printf("the number of image net has seen: %d, max_batches of net: %d\n", *net->seen, net->max_batches);

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(count++%40 == 0){
            int dim = (rand() % 11 + 4) * 32;
            args.w = dim;
            args.h = dim;
            args.size = dim;
            printf("resize_network: %d\n", dim);
            printf("input size: [%d %d]\n", args.min, args.max);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            net = nets[0];
        }
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        time = what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("batch: %d, epoch: %.3f, loss: %f, avg_loss: %f avg, learning_rate: %f, %lf seconds, seen %d images\n",
        		get_current_batch(net), (float)(*net->seen) / train_set_size, loss, avg_loss,
				get_current_learning_rate(net), what_time_is_it_now()-time, *net->seen);

        free_data(train);
        if(get_current_batch(net) % 1000 == 0) {
            epoch = *net->seen / train_set_size;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory, base, epoch);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}
/*
void validate_classifier_single(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    struct network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    struct list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(0, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    struct list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        struct image im = load_image_color(paths[i], 0, 0);
        struct image resized = resize_min(im, net->w);
        struct image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

if(i == m-1){
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
}
    }
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
	struct network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    struct list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        struct image im = load_image_color(input, 0, 0);
        struct image r = letterbox_image(im, net->w, net->h);
        //resize_network(net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}
*/
void run_classifier(int argc, char **argv)
{
    double time_start = what_time_is_it_now();;
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/predict/valid] [data cfg] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "predict")){
    	;//predict_classifier(data, cfg, weights, filename, top);
    }
    else if(0==strcmp(argv[2], "train")){
    	train_classifier(data, cfg, weights, gpus, ngpus, clear);
    }
    else if(0==strcmp(argv[2], "valid")){
    	;//validate_classifier_single(data, cfg, weights);
    }
    printf("total %.2lf seconds\n", what_time_is_it_now() - time_start);
}
