#include <unistd.h>
#include <sys/time.h>
#include <assert.h>

#include "utils.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"
#include "network.h"

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    char *base = basecfg(cfgfile);   // get projetc name by cfgfile, cifar.cfg -> cifar
    //fprintf(stderr, "train data base name: %s\n", base);
    //fprintf(stderr, "the number of GPU: %d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(struct network*));  // todo: free

    srand(time(0));
    int seed = rand();
    for(int i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);;
    }
    network *net = nets[0];
    if(weightfile && weightfile[0] != 0){
        load_weights(net, weightfile);
    }
    struct list *options = read_data_cfg(datacfg);
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char **labels = get_labels(label_list);
    char *train_list = option_find_str(options, "train", "data/train.list");
    net->classes = option_find_int(options, "classes", 2);

    int train_set_size = 0;
    int batch_num = 0;
    char **paths = NULL;
    struct list *plist = NULL;
    int train_data_type = option_find_int(options, "train_data_type", 1);    //  0: csv, 1: load to memory
    batch *all_train_data = NULL;
    if(0 == train_data_type) {
        train_set_size = option_find_int(options, "train_num", 0);
        all_train_data = load_csv_image_to_memory(train_list, net->batch, labels, net->classes,
            train_set_size, &batch_num, net->w, net->h, net->c, net->hue, net->saturation, net->exposure);
    } else if(1 == train_data_type){
        plist = get_paths(train_list);
        paths = (char **)list_to_array(plist);
        train_set_size = plist->size;
        train_set_size = option_find_int(options, "train_num", train_set_size);
        all_train_data = load_image_to_memory(paths, net->batch, labels, net->classes, train_set_size, &batch_num,
                                              net->w, net->h, net->c, net->hue, net->saturation, net->exposure);
    } else {
        plist = get_paths(train_list);
        paths = (char **)list_to_array(plist);
        train_set_size = plist->size;
    }
    double time;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    printf("image net has seen: %lu, train_set_size: %d, max_batches of net: %d, net->classes: %d, net->batch: %d\n\n",
            net->seen, train_set_size, net->max_batches, net->classes, net->batch);

    float avg_loss = -1;
    while(net->epoch < net->max_epoch || net->max_epoch == 0){
        time = what_time_is_it_now();
        batch train;
        if(0 == train_data_type) {
            int index = rand() % batch_num;
            //index = 1;
            train = all_train_data[index];
            for(int i = 0; i < net->classes * net->batch; i++) {
                //if(train.truth[i] > 0.1) printf("input class: %d %d %f\n", batch_num, i, train.truth[i]);
            }
            train_network_batch(net, train);
        } else if(1 == train_data_type) {
            int index = rand() % batch_num;
            train = all_train_data[index];
            train_network_batch(net, train);
        } else {
            train = random_batch(paths, net->batch, labels, net->classes, train_set_size,
                    net->w, net->h, net->c, net->hue, net->saturation, net->exposure);
            for(int i = 0; i < net->classes * net->batch; i++) {
                //if(train.truth[i] > 0.1) printf("input class: %d %f\n", i, train.truth[i]);
            }
            image tmp;
            tmp.w = train.w;
            tmp.h = train.h;
            tmp.c = train.c;
            tmp.data = train.data;
            save_image_png(tmp, "input.jpg");
            //tmp.data = train.data + tmp.w * tmp.h * tmp.c;
            //save_image_png(tmp, "input0.jpg");

            train_network_batch(net, train);
            free_batch(&train);
        }
        int epoch_old = net->epoch;
        net->epoch = net->seen / train_set_size;
        float loss = 0;
        if(avg_loss == -1) avg_loss = loss;
        loss = net->loss;
        if(loss > 999999 || loss < -999999 || loss != loss || (loss + 1.0 == loss)) {  // NaN ≠ NaN, Inf + 1 = Inf
            fprintf(stderr, "\n\nloss too large: %f, exit\n", loss);
            //continue;
            exit(-1);
        }
        avg_loss = avg_loss*.9 + loss*.1;
        net->learning_rate = update_current_learning_rate(net);
        printf("epoch: %d, batch: %d, accuracy: %.3f, loss: %f, avg_loss: %f avg, learning_rate: %f, %lf seconds, "
                "seen %lu images\n", net->epoch, net->batch_train, net->correct_num / (net->correct_num_count + 0.00001F),
                loss, avg_loss, net->learning_rate, what_time_is_it_now()-time, net->seen);
        if(epoch_old != net->epoch){
            char buff[256];
            sprintf(buff, "%s/%s_%06d.weights", backup_directory, base, net->epoch);
            save_weights(net, buff);
        }
        //sleep(3);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    free_network(net);
    if(all_train_data){
        for(int i = 0; i < batch_num; i++){
            free_batch(all_train_data + i);
        }
        free(all_train_data);
    }
    free_ptrs((void**)labels, net->classes);
    if(paths) free_ptr(paths);
    if(plist){
        free_list_contents(plist);
        free_list(plist);
    }
    free(base);
}

void validate_classifier(char *datacfg, char *cfgfile, char *weightfile)
{
    srand(time(0));
    int seed = rand();
    srand(seed);

    network *net = parse_network_cfg(cfgfile);
    if(weightfile && weightfile[0] != 0){
        load_weights(net, weightfile);
    }
    net->test = 1;
#ifdef GPU
    cuda_set_device(net->gpu_index);
#endif

    struct list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char **labels = get_labels(label_list);
    char *valid_list = option_find_str(options, "valid", "data/valid.list");
    net->classes = option_find_int(options, "classes", 2);

    int valid_set_size = 0;
    int batch_num = 0;
    char **paths = NULL;
    struct list *plist = NULL;
    int train_data_type = option_find_int(options, "train_data_type", 1);    //  0: csv, 1: load to memory
    batch *all_valid_data = NULL;
    if(0 == train_data_type) {
        valid_set_size = option_find_int(options, "valid_num", 0);
        all_valid_data = load_csv_image_to_memory(valid_list, net->batch, labels, net->classes, valid_set_size,
                                                  &batch_num, net->w, net->h, net->c, net->hue, net->saturation, net->exposure);
    } else if(1 == train_data_type){
        plist = get_paths(valid_list);
        paths = (char **)list_to_array(plist);
        valid_set_size = plist->size;
        valid_set_size = option_find_int(options, "valid_num", valid_set_size);
        all_valid_data = load_image_to_memory(paths, net->batch, labels, net->classes, valid_set_size, &batch_num,
                                              net->w, net->h, net->c, net->hue, net->saturation, net->exposure);
    } else {
        plist = get_paths(valid_list);
        paths = (char **)list_to_array(plist);
        valid_set_size = plist->size;
    }

    fprintf(stderr, "valid_set_size: %d, net->classes: %d\n", valid_set_size, net->classes);

    float avg_loss = -1;
    int count = 0;
    while(count < valid_set_size){
        batch train;
        if(0 == train_data_type) {
            int index = rand() % batch_num;
            train = all_valid_data[index];
            valid_network(net, train);
        } else if(1 == train_data_type) {
            int index = rand() % batch_num;
            train = all_valid_data[index];
            valid_network(net, train);
        } else {
            train = random_batch(paths, net->batch, labels, net->classes, valid_set_size,
                    net->w, net->h, net->c, net->hue, net->saturation, net->exposure);
            valid_network(net, train);
            free_batch(&train);
        }
        float loss = 0;
        if(avg_loss == -1) avg_loss = loss;
        loss = net->loss;
        if(loss > 999999 || loss < -999999 || loss != loss || (loss + 1.0 == loss)) {  // NaN ≠ NaN, Inf + 1 = Inf
            fprintf(stderr, "\n\nloss too large: %f, exit\n", loss);
            //exit(-1);
        }
        avg_loss = avg_loss*.9 + loss*.1;
        if(count == valid_set_size - 1){
            printf("count: %d, accuracy: %.3f, loss: %f, avg_loss: %f\n",
                   count, net->correct_num / (net->correct_num_count + 0.00001F), loss, avg_loss);
        }
        count += 1;
    }
    free_network(net);
    free_ptrs((void**)labels, net->classes);
    if(paths) free_ptrs((void**)paths, plist->size);
    if(plist){
        free_list_contents(plist);
        free_list(plist);
    }
}

void run_classifier(int argc, char **argv)
{
    double time_start = what_time_is_it_now();;
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/predict/valid] [data cfg] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "predict")){
        //char *filename = (argc > 6) ? argv[6]: 0;
        //int top = find_int_arg(argc, argv, "-t", 0);
        ;//predict_classifier(data, cfg, weights, filename, top);
    }
    else if(0==strcmp(argv[2], "train")){
        train_classifier(data, cfg, weights, gpus, ngpus, clear);
    }
    else if(0==strcmp(argv[2], "valid")){
        validate_classifier(data, cfg, weights);
    }
    fprintf(stderr, "total %.2lf seconds\n\n\n", what_time_is_it_now() - time_start);
    printf("\n\n");
}
