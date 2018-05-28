#include "utils.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"

#include <unistd.h>

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
    struct network *net = nets[0];
    struct list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char **labels = get_labels(label_list);
    char *train_list = option_find_str(options, "train", "data/train.list");
    net->classes = option_find_int(options, "classes", 2);

    int train_set_size = 0;
	int train_data_type = 0;	//  0: csv, load to memory
	char **paths = NULL;
	struct list *plist = NULL;
	batch *all_train_data;
	if(0 == train_data_type) {
		train_set_size = option_find_int(options, "train_num", 0);
		all_train_data = load_csv_image_to_memory(train_list, 1, labels, net->classes, train_set_size);
	} else {
		plist = get_paths(train_list);
		paths = (char **)list_to_array(plist);
		train_set_size = plist->size;
	}
    double time;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    printf("image net has seen: %d, train_set_size: %d, max_batches of net: %d, net->classes: %d\n\n\n",
    		net->seen, train_set_size, net->max_batches, net->classes);

    while(net->seen < net->max_batches || net->max_batches == 0){
    	batch train;
    	if(0 == train_data_type) {
            int index = rand() % train_set_size;
    		train = all_train_data[index];
    		/*
        	for(int i = 0; i < net->classes; i++) {
        		if(train.truth[0][i] > 0.1) printf("input class: %d %f\n", i, train.truth[0][i]);
        	}*/
        	train_network_batch(net, train);
    		//save_image_png(train.images[0], "input.jpg");
        	//sleep(1);
    	} else {
        	train = random_batch(paths, 1, labels, net->classes, train_set_size);
        	train_network_batch(net, train);
        	free_batch(train);
    	}
        time = what_time_is_it_now();
        float loss = 0;
        if(avg_loss == -1) avg_loss = loss;
    	cost_layer *layer = (cost_layer *)net->layers[net->n - 1];
        loss = layer->cost[0];
        if(loss > 999999 || loss < -999999 || loss != loss || (loss + 1.0 == loss)) {  // NaN ≠ NaN, Inf + 1 = Inf
			printf("\n\nloss too large: %f, exit\n", loss);
			exit(-1);
		}
        avg_loss = avg_loss*.9 + loss*.1;
        net->learning_rate = get_current_learning_rate(net);
        fprintf(stderr, "batch: %d, accuracy: %.3f, loss: %f, avg_loss: %f avg, learning_rate: %f, %lf seconds, seen %d images\n",
        		net->seen, net->correct_num / (net->correct_num_count + 0.00001F), loss, avg_loss,
				net->learning_rate, what_time_is_it_now()-time, net->seen);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    //save_weights(net, buff);
    //free_network(net);
    free_ptrs((void**)labels, net->classes);
    if(paths) free_ptrs((void**)paths, plist->size);
    if(plist){
    	free_list_contents(plist);
        free_list(plist);
    }
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
