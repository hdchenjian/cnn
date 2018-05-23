#include "utils.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"

#include <sys/time.h>
#include <assert.h>

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    float avg_loss = -1;
    char *base = basecfg(cfgfile);   // get projetc name by cfgfile, cifar.cfg -> cifar
    printf("train data base name: %s\n", base);
    printf("the number of GPU: %d\n", ngpus);
    struct network **nets = calloc(ngpus, sizeof(struct network*));  // todo: free

    srand(time(0));
    int seed = rand();
    for(int i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);;
    }
    srand(time(0));
    struct network *net = nets[0];
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
    printf("the number of image net has seen: %d, max_batches of net: %d\n", net->seen, net->max_batches);

    while(net->seen < net->max_batches || net->max_batches == 0){
    	batch train = random_batch(train_list, train_set_size, labels, classes);
    	train_network_batch(net, train);
    	free_batch(train);
    	printf("Round %d\n", net->seen);
        time = what_time_is_it_now();
        float loss = 0;
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("batch: %d, epoch: %.3f, loss: %f, avg_loss: %f avg, learning_rate: %f, %lf seconds, seen %d images\n",
        		net->seen, (float)(net->seen) / train_set_size, loss, avg_loss,
				net->learning_rate, what_time_is_it_now()-time, net->seen);
        net->seen += 1;
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    //save_weights(net, buff);
    //free_network(net);
    free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
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
