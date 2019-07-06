//#include <unistd.h>
//#include <sys/time.h>
#include <assert.h>
//#include <pthread.h>

#include "utils.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"
#include "network.h"

#ifdef USE_LINUX
#include <unistd.h>
#include <pthread.h>

pthread_mutex_t mutex;
int load_over;
batch train_global;

typedef struct load_args{
    char **paths;
    char **labels;
    int classes, train_set_size, w, h, c, batch_size, flip, test;
    float hue, saturation, exposure, mean_value, scale;
} load_args;

void *load_data_in_thread(void *args_point)
{
    load_args args = *(load_args *)args_point;
    while(1){
        pthread_mutex_lock(&mutex);
        if(load_over == 0){
            train_global = random_batch(
                args.paths, args.batch_size, args.labels, args.classes,
                args.train_set_size, args.w, args.h, args.c,
                args.hue, args.saturation, args.exposure, args.flip, args.mean_value, args.scale,
                args.test);
            //printf("load_over\n");
            load_over = 1;
            pthread_mutex_unlock(&mutex);
        } else {
            pthread_mutex_unlock(&mutex);
        }
        usleep(10000);
    }
}

void train_classifier(char *datacfg, char *cfgfile, char *weightfile)
{
    char *base = basecfg(cfgfile);   // get projetc name by cfgfile, cifar.cfg -> cifar
    //fprintf(stderr, "train data base name: %s\n", base);
    //fprintf(stderr, "the number of GPU: %d\n", ngpus);
    srand(time(0));
    network *net = load_network(cfgfile, weightfile, 0);
    net->output_layer = net->n - 1;
    struct list *options = read_data_cfg(datacfg);
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char **labels = get_labels(label_list);
    char *train_list = option_find_str(options, "train", "data/train.list");

    int train_set_size = 0;
    int batch_num = 0;
    char **paths = NULL;
    struct list *plist = NULL;
    int train_data_type = option_find_int(options, "train_data_type", 1);    //  0: csv, 1: load to memory
    batch *all_train_data = NULL;
    if(0 == train_data_type) {
        train_set_size = option_find_int(options, "train_num", 0);
        all_train_data = load_csv_image_to_memory(train_list, net->batch * net->subdivisions, labels, net->classes, train_set_size,
                                                  &batch_num, net->w, net->h, net->c, net->hue, net->saturation,
                                                  net->exposure, net->test);
    } else if(1 == train_data_type){
        plist = get_paths(train_list);
        paths = (char **)list_to_array(plist);
        train_set_size = plist->size;
        train_set_size = option_find_int(options, "train_num", train_set_size);
        all_train_data = load_image_to_memory(paths, net->batch * net->subdivisions, labels, net->classes, train_set_size, &batch_num,
                                              net->w, net->h, net->c, net->hue, net->saturation, net->exposure,
                                              net->flip, net->mean_value, net->scale, net->test);
    } else {
        plist = get_paths(train_list);
        paths = (char **)list_to_array(plist);
        train_set_size = plist->size;
        train_set_size = option_find_int(options, "train_num", train_set_size);
    }
    double time;
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int max_epoch = (int)net->max_batches * net->batch * net->subdivisions / train_set_size;
    fprintf(stderr, "image net has seen: %lu, train_set_size: %d, max_batches of net: %d, net->classes: %d,"
           "net->batch: %d, max_epoch: %d\n\n",
           net->seen, train_set_size, net->max_batches, net->classes, net->batch, max_epoch);

    net->batch_train = net->seen / net->batch / net->subdivisions;
    net->epoch = net->seen / train_set_size;
    float avg_loss = -1;
    float max_accuracy = -1;
    int max_accuracy_batch = 0;
    pthread_t load_data_thread_id;
    if(0 != train_data_type && 1 != train_data_type){
        load_args args = {0};
        args.paths = paths;
        args.batch_size = net->batch * net->subdivisions;
        args.labels = labels;
        args.classes =  net->classes;
        args.train_set_size = train_set_size;
        args.w = net->w;
        args.h = net->h;
        args.c = net->c;
        args.hue = net->hue;
        args.saturation = net->saturation;
        args.exposure=net->exposure;
        args.flip= net->flip;
        args.mean_value = net->mean_value;
        args.scale = net->scale;
        args.test = net->test;
        pthread_mutex_init(&mutex, NULL);
        pthread_create(&load_data_thread_id, NULL, load_data_in_thread, &args);
        usleep(1000000);
    }
    while(net->batch_train < net->max_batches){
        batch train;
        time = what_time_is_it_now();
        update_current_learning_rate(net);
        if(0 == train_data_type) {
            int index = rand() % batch_num;
            train = all_train_data[index];
            /*printf("class: %d\n", train.truth_label_index[0]);
            image tmp;
            tmp.w = train.w;
            tmp.h = train.h;
            tmp.c = train.c;
            tmp.data = train.data;
            float max = -FLT_MAX, min = FLT_MAX;
            for(int i = 0; i < train.w * train.h * train.c; ++i){
                if(train.data[i] > max) max = train.data[i];
                if(train.data[i] < min) min = train.data[i];
            }
            printf("input image max: %f, min: %f\n", max, min);
            save_image_png(tmp, "input.jpg");*/
            train_network(net, train.data, train.truth_label_index);
        } else if(1 == train_data_type) {
            int index = rand() % batch_num;
            train = all_train_data[index];
            train_network(net, train.data, train.truth_label_index);
        } else {
            while(1){
                /*
                train = random_batch(paths, net->batch * net->subdivisions, labels, net->classes,
                                     train_set_size, net->w, net->h, net->c, net->hue, net->saturation, net->exposure,
                                     net->flip, net->mean_value, net->scale, net->test);
                printf("train_data_type: %d, spend %f \n", train_data_type, what_time_is_it_now() - time);
                break;
                */
                pthread_mutex_lock(&mutex);
                if(load_over == 1){
                    train = train_global;
                    load_over = 0;
                    pthread_mutex_unlock(&mutex);
                    break;
                }
                pthread_mutex_unlock(&mutex);
                printf("wait load_over\n");
                usleep(5000);
            }
            /*
            image tmp;
            tmp.w = train.w;
            tmp.h = train.h;
            tmp.c = train.c;
            tmp.data = train.data;
            float max = -FLT_MAX, min = FLT_MAX;
            for(int i = 0; i < train.w * train.h * train.c; ++i){
                if(train.data[i] > max) max = train.data[i];
                if(train.data[i] < min) min = train.data[i];
            }
            printf("input image max: %f, min: %f , class: %d\n", max, min, train.truth_label_index[0]);
            save_image_png(tmp, "input.jpg");
            sleep(3);
            */
            //printf("train_data_type: %d, spend %f \n", train_data_type, what_time_is_it_now() - time);
            //double train_start_time = what_time_is_it_now();
            train_network(net, train.data, train.truth_label_index);
            //printf("train spend %f \n", what_time_is_it_now() - train_start_time);
            free_batch(&train);
        }
        int epoch_old = net->epoch;
        net->epoch = net->seen / train_set_size;
        float loss = net->loss;
        if(loss > 999999 || loss < -999999 || loss != loss || (loss + 1.0 == loss)) {  // NaN ≠ NaN, Inf + 1 = Inf
            fprintf(stderr, "\n\nloss too large: %f, exit\n", loss);
            exit(-1);
        }
        if(avg_loss < 0){
            avg_loss = loss;
        } else {
            avg_loss = avg_loss*.9 + loss*.1;
        }
        if(net->correct_num / (net->accuracy_count + 0.00001F) > max_accuracy){
            max_accuracy = net->correct_num / (net->accuracy_count + 0.00001F);
            max_accuracy_batch = net->batch_train;
        }
        printf("epoch:%d, batch:%d, accuracy: %.4f, loss: %.2f, avg_loss:%.2f, learning_rate:%f, %.3fs, "
               "seen %lu image, max_accuracy: %.4f\n", net->epoch+1, net->batch_train,
               net->correct_num / (net->accuracy_count + 0.00001F),
               loss, avg_loss, net->learning_rate, what_time_is_it_now() - time, net->seen,  max_accuracy);
        if(epoch_old != net->epoch){
            int save_weight_times = 20;
            int save_weight_interval = max_epoch / save_weight_times;
            if(save_weight_interval <= 1 || net->epoch % save_weight_interval == 0){
                char buff[256];
                sprintf(buff, "%s/%s_%06d.weights", backup_directory, base, net->epoch);
                save_weights(net, buff);
            }
        } else {
            if(net->batch_train > 1000 && net->batch_train % 3000 == 0){
                char buff[256];
                sprintf(buff, "%s/%s.backup",backup_directory,base);
                save_weights(net, buff);
            }
        }
        //sleep(30);
        //exit(-1);
    }
    printf("max_accuracy_batch: %d\n", max_accuracy_batch);
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
    if(paths) free_ptr((void *)&paths);
    if(plist){
        free_list_contents(plist);
        free_list(plist);
    }
    free(base);
}
#endif

void validate_classifier(char *datacfg, char *cfgfile, char *weightfile)
{
    srand(time(0));
    network *net = load_network(cfgfile, weightfile, 1);
    struct list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels_test", "data/labels.list");
    int label_num = 0;
    char **labels = get_labels_and_num(label_list, &label_num);
    char *valid_list = option_find_str(options, "valid", "data/valid.list");

    int valid_set_size = 0;
    int batch_num = 0;
    char **paths = NULL;
    struct list *plist = NULL;
    int train_data_type = option_find_int(options, "train_data_type", 1);    //  0: csv, 1: load to memory

    batch *all_valid_data = NULL;
    if(0 == train_data_type) {
        valid_set_size = option_find_int(options, "valid_num", 0);
        all_valid_data = load_csv_image_to_memory(valid_list, net->batch, labels, net->classes, valid_set_size,
                                                  &batch_num, net->w, net->h, net->c, net->hue, net->saturation,
                                                  net->exposure, net->test);
    } else if(1 == train_data_type){
        plist = get_paths(valid_list);
        paths = (char **)list_to_array(plist);
        valid_set_size = plist->size;
        valid_set_size = option_find_int(options, "valid_num", valid_set_size);
        all_valid_data = load_image_to_memory(paths, net->batch, labels, net->classes, valid_set_size, &batch_num,
                                              net->w, net->h, net->c, net->hue, net->saturation, net->exposure,
                                              net->flip, net->mean_value, net->scale, net->test);
    } else {
        plist = get_paths(valid_list);
        paths = (char **)list_to_array(plist);
        valid_set_size = plist->size;
        batch_num = valid_set_size;
    }

    fprintf(stderr, "valid_set_size: %d, net->classes: %d, net->batch: %d, batch_num: %d, train_data_type: %d\n",
            valid_set_size, net->classes, net->batch, batch_num, train_data_type);
    // when net->batch != 1, train.data may can not match in last batch, such as net->batch = 2, valid_set_size = 11
    if(net->batch != 1){
        printf("\n\nerror: net->batch != 1\n\n");
        exit(-1);
    }

    float avg_loss = -1;
    int count = 0;

    FILE *fp = fopen("features.txt", "w");
    if(!fp) file_error("features.txt");
    while(count < batch_num){
        batch train;
        if(0 == train_data_type) {
            train = all_valid_data[count];
            valid_network(net, train.data, train.truth_label_index);
        } else if(1 == train_data_type) {
            train = all_valid_data[count];
            valid_network(net, train.data, train.truth_label_index);
        } else {
            train = random_batch(paths, net->batch, labels, net->classes, valid_set_size, net->w, net->h, net->c,
                                 net->hue, net->saturation, net->exposure, net->flip, net->mean_value, net->scale,
                                 net->test);
            //for(int i = 0; i < train.w * train.h * train.c; ++i) train.data[i] = -0.98828125;
            valid_network(net, train.data, train.truth_label_index);
            free_batch(&train);
        }

        int network_output_size = get_network_output_size_layer(net, net->output_layer);
#ifndef GPU
        float *network_output = get_network_layer_data(net, net->output_layer, 0, 0);
#else
        float *network_output_gpu = get_network_layer_data(net, net->output_layer, 0, 1);
        float *network_output = malloc(network_output_size * sizeof(float));
        cuda_pull_array(network_output_gpu, network_output, network_output_size);

        /*
        char cuda_compare_error_string[128] = {0};
        sprintf(cuda_compare_error_string, "\n%s", "validate_classifier output");
        float *network_output_cpu = get_network_layer_data(net, net->output_layer, 0, 0);
        cuda_compare(network_output_gpu, network_output_cpu, net->batch * network_output_size,
                     cuda_compare_error_string);
        */
#endif
        for(int i = 0; i < network_output_size; i++){
            fprintf(fp, "%f ", network_output[i]);
            //if(i < 10) printf("%f\n", network_output[i]);
        }
        //break;
        fprintf(fp, "\n");

        float loss = net->loss;
        if(loss > 999999 || loss < -999999 || loss != loss || (loss + 1.0 == loss)) {  // NaN ≠ NaN, Inf + 1 = Inf
            fprintf(stderr, "\n\nloss too large: %f, exit\n", loss);
            exit(-1);
        }
        if(avg_loss < 0){
            avg_loss = loss;
        } else {
            avg_loss = avg_loss*.9 + loss*.1;
        }
        if(count % 100 == 0 || count == valid_set_size - 1){
            printf("count: %d, accuracy: %.3f, loss: %f, avg_loss: %f\n",
                   count, net->correct_num / (net->accuracy_count + 0.00001F), loss, avg_loss);
        }
        count += 1;
        //sleep(3);
    }
    fclose(fp);
    free_network(net);
    if(all_valid_data){
        for(int i = 0; i < batch_num; i++){
            free_batch(all_valid_data + i);
        }
        free(all_valid_data);
    }
    free_ptrs((void**)labels, label_num);
    if(paths) free_ptr((void *)&paths);
    if(plist){
        free_list_contents(plist);
        free_list(plist);
    }
}

network *net_recognition = NULL;
void init_recognition(const char *cfgfile, const char *weightfile)
{
    if(net_recognition != NULL){
        printf("error: has call init_recognition already\n");
        return;
    }
    srand(time(0));
    net_recognition = load_network(cfgfile, weightfile, 1);
    fprintf(stderr, "net->classes: %d, net->batch: %d\n", net_recognition->classes, net_recognition->batch);
#ifdef FORWARD_GPU
    free_network_weight_bias_cpu(net_recognition);
#endif
}

void run_recognition(float *image_data, int face_num, float *feature)
{
    if(net_recognition == NULL){
        printf("error: please call init_recognition first\n");
        return;
    }
    int net_batch = 5;
    int *truth_label_index = calloc(face_num, sizeof(int));
    int network_output_size = get_network_output_size_layer(net_recognition, net_recognition->output_layer);
#if defined(GPU)  || defined(OPENCL)
    float *network_output = malloc(net_batch * network_output_size * sizeof(float));
#endif
    float *network_output_tmp = malloc(net_batch * network_output_size * sizeof(float));

    if(face_num <= net_batch){
        net_recognition->batch = face_num;
    } else {
        net_recognition->batch = net_batch;
    }
    valid_network(net_recognition, image_data, truth_label_index);

#ifdef GPU
    float *network_output_gpu = get_network_layer_data(net_recognition, net_recognition->output_layer, 0, 1);
    cuda_pull_array(network_output_gpu, network_output, network_output_size * net_recognition->batch);
    l2normalize_cpu(network_output, net_recognition->batch, network_output_size, 1, network_output_tmp);
#elif defined(OPENCL)
    cl_mem network_output_cl = get_network_layer_data_cl(net_recognition, net_recognition->output_layer, 0);
    cl_read_array(network_output_cl, network_output_for_gpu, network_output_size * net_recognition->batch);
#else
    float *network_output = get_network_layer_data(net_recognition, net_recognition->output_layer, 0, 0);
    l2normalize_cpu(network_output, net_recognition->batch, network_output_size, 1, network_output_tmp);
#endif
    memcpy(feature, network_output, net_recognition->batch * network_output_size * sizeof(float));

#if defined(GPU)  || defined(OPENCL)
    free(network_output);
#endif
    free(truth_label_index);
    free(network_output_tmp);
}

void uninit_recognition()
{
    if(net_recognition) free_network(net_recognition);
    else printf("error: please call init_recognition first\n");
}

network *net_spoofing = NULL;
void uninit_spoofing()
{
    if(net_spoofing) free_network(net_spoofing);
    else printf("error: please call init_spoofing first\n");
}

void init_spoofing(const char *cfgfile, const char *weightfile)
{
    if(net_spoofing != NULL){
        printf("error: has call init_spoofing already\n");
        return;
    }
    srand(time(0));
    net_spoofing = load_network(cfgfile, weightfile, 1);
    fprintf(stderr, "net->classes: %d, net->batch: %d\n", net_spoofing->classes, net_spoofing->batch);
    if(net_spoofing->batch != 1){
        printf("\nerror: net->batch != 1\n");
        //uninit_spoofing();
        //exit(-1);
    }
#ifdef FORWARD_GPU
    free_network_weight_bias_cpu(net_spoofing);
#endif
}

void run_spoofing(float *image_data, int face_num, int *is_real_face)
{
    if(net_spoofing == NULL){
        printf("error: please call init_spoofing first\n");
        return;
    }
    int net_batch = 5;
    int forward_times = 1;
    if(face_num <= net_batch) {
        forward_times = 1;
    } else {
        if(face_num % net_batch == 0) forward_times = face_num / net_batch;
        else forward_times = face_num / net_batch + 1;
    }
    int *truth_label_index = calloc(face_num, sizeof(int));
    int *truth_label_index_bak = truth_label_index;
#if defined(GPU)  || defined(OPENCL)
    int network_output_size = get_network_output_size_layer(net_spoofing, net_spoofing->output_layer);
    float *network_output = malloc(net_batch * network_output_size * sizeof(float));
#endif
    int real_face_index = 0;
    for(int i = 0; i < forward_times; i++){
        if(face_num <= net_batch){
            net_spoofing->batch = face_num;
        } else {
            if(face_num % net_batch == 0) {
                net_spoofing->batch = net_batch;
            } else {
                if(i < forward_times - 1) net_spoofing->batch = net_batch;
                else net_spoofing->batch = face_num % net_batch;
            }
        }
        valid_network(net_spoofing, image_data, truth_label_index_bak);
        image_data += net_spoofing->batch * net_spoofing->w * net_spoofing->h * net_spoofing->c;
        truth_label_index_bak += net_spoofing->batch;
#ifdef GPU
        float *network_output_gpu = get_network_layer_data(net_spoofing, net_spoofing->output_layer, 0, 1);
        cuda_pull_array(network_output_gpu, network_output, network_output_size * net_spoofing->batch);
#elif defined(OPENCL)
        cl_mem network_output_cl = get_network_layer_data_cl(net_spoofing, net_spoofing->output_layer, 0);
        cl_read_array(network_output_cl, network_output, network_output_size * net_spoofing->batch);

#else
        float *network_output = get_network_layer_data(net_spoofing, net_spoofing->output_layer, 0, 0);
#endif
        for(int j = 0; j < net_spoofing->batch; j++){
            printf("network_output %d %f %f\n", real_face_index, network_output[2*j], network_output[2*j + 1]);
            if(network_output[2*j] >= network_output[2*j + 1]) is_real_face[real_face_index] = 1;
            else is_real_face[real_face_index] = 0;
            real_face_index += 1;
        }
    }
#if defined(GPU)  || defined(OPENCL)
    free(network_output);
#endif
    free(truth_label_index);
}

network *net_mtcnn = NULL;
void uninit_mtcnn()
{
    if(net_mtcnn) free_network(net_mtcnn);
    else printf("error: please call init_mtcnn first\n");
}

void init_mtcnn(const char *cfgfile, const char *weightfile)
{
    if(net_mtcnn != NULL){
        printf("error: has call init_mtcnn already\n");
        return;
    }
    srand(time(0));
    net_mtcnn = load_network(cfgfile, weightfile, 1);
    fprintf(stderr, "net->classes: %d, net->batch: %d\n", net_mtcnn->classes, net_mtcnn->batch);
    if(net_mtcnn->batch != 1){
        printf("\nerror: net->batch != 1\n");
    }
#ifdef FORWARD_GPU
    free_network_weight_bias_cpu(net_mtcnn);
#endif
}

void run_mtcnn(float *image_data, int face_num, float *landmark)
{
    if(net_mtcnn == NULL){
        printf("error: please call init_mtcnn first\n");
        return;
    }
    int net_batch = 5;
    int *truth_label_index = calloc(face_num, sizeof(int));
    int network_output_size = get_network_output_size_layer(net_mtcnn, net_mtcnn->output_layer);
#if defined(GPU)  || defined(OPENCL)
    float *network_output = malloc(net_batch * network_output_size * sizeof(float));
#endif
    net_mtcnn->batch = face_num > net_batch ? net_batch : face_num;
    valid_network(net_mtcnn, image_data, truth_label_index);

#ifdef GPU
    float *network_output_gpu = get_network_layer_data(net_mtcnn, net_mtcnn->output_layer, 0, 1);
    cuda_pull_array(network_output_gpu, network_output, network_output_size * net_mtcnn->batch);
#elif defined(OPENCL)
    cl_mem network_output_cl = get_network_layer_data_cl(net_mtcnn, net_mtcnn->output_layer, 0);
    cl_read_array(network_output_cl, network_output, network_output_size * net_mtcnn->batch);
#else
    float *network_output = get_network_layer_data(net_mtcnn, net_mtcnn->output_layer, 0, 0);
#endif

    memcpy(landmark, network_output, network_output_size * net_mtcnn->batch * sizeof(float));

#if defined(GPU) || defined(OPENCL)
    free(network_output);
#endif
    free(truth_label_index);
}

void run_classifier(int argc, char **argv)
{
    double time_start = what_time_is_it_now();;
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/valid] [data cfg] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "train")){
        #ifdef USE_LINUX
        train_classifier(data, cfg, weights);
        #endif
    } else if(0==strcmp(argv[2], "valid")){
        validate_classifier(data, cfg, weights);
    } else {
        fprintf(stderr, "usage: %s %s [train/valid] [data cfg] [cfg] [weights (optional)]\n", argv[0], argv[1]);
    }
    fprintf(stderr, "\n\ntotal %.2lf seconds\n\n\n", what_time_is_it_now() - time_start);
}
