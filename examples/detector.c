//#include <sys/time.h>
#include <assert.h>

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
batch_detect train_global;

typedef struct load_args{
    char **paths;
    int classes, train_set_size, w, h, test, batch, subdivisions, max_boxes;
    float hue, saturation, exposure, jitter;
} load_args;

void *load_detect_data_thread(void *args_point)
{
    load_args args = *(load_args *)args_point;
    while(1){
        pthread_mutex_lock(&mutex);
        if(load_over == 0){
            train_global = load_data_detection(args.batch * args.subdivisions, args.paths, args.train_set_size, args.w, args.h, args.max_boxes,
                                               args.classes, args.jitter, args.hue, args.saturation, args.exposure, args.test);
            load_over = 1;
            pthread_mutex_unlock(&mutex);
        } else {
            pthread_mutex_unlock(&mutex);
        }
        usleep(10000);
    }
}

void train_detector(char *datacfg, char *cfgfile, char *weightfile)
{
    char *base = basecfg(cfgfile);
    srand(time(0));
    network *net = load_network(cfgfile, weightfile, 0);

    struct list *options = read_data_cfg(datacfg);
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");

    int train_set_size = 0;
    char **paths = NULL;
    struct list *plist = NULL;
    plist = get_paths(train_list);
    paths = (char **)list_to_array(plist);
    train_set_size = plist->size;
    train_set_size = option_find_int(options, "train_num", train_set_size);
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
    int burn_in = 1000;
    batch_detect train;
    load_args args = {0};
    args.paths = paths;
    args.batch = net->batch;
    args.subdivisions = net->subdivisions;
    args.classes =  net->classes;
    args.train_set_size = train_set_size;
    args.w = net->w;
    args.h = net->h;
    args.max_boxes = net->max_boxes;
    args.jitter=net->jitter;
    args.hue = net->hue;
    args.saturation = net->saturation;
    args.exposure=net->exposure;
    args.test = net->test;
    pthread_mutex_init(&mutex, NULL);
    pthread_t load_data_thread_id;
    pthread_create(&load_data_thread_id, NULL, load_detect_data_thread, &args);
    usleep(1000000);
    while(net->batch_train < net->max_batches){
        time = what_time_is_it_now();
        update_current_learning_rate(net);
        if(net->batch_train < burn_in) net->learning_rate = net->learning_rate_init * pow((float)net->batch_train / burn_in, 4);
        else if(net->batch_train == burn_in) net->learning_rate = net->learning_rate_init;
        while(1){
            /*
            train = load_data_detection(net->batch * net->subdivisions, paths, train_set_size, net->w, net->h, net->max_boxes,
                                        net->classes, net->jitter, net->hue, net->saturation, net->exposure, net->test);
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
        printf("Loaded: %lf seconds\n", what_time_is_it_now() - time);
        int zz;
        for(zz = 0; zz < train.X.rows; ++zz){
            image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
            int k;
            for(k = 0; k < net->max_boxes; ++k){
                box b = float_to_box(train.y.vals[zz] + k*5, 1);
                if(!b.x) break;
                printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
                draw_bbox(im, b, 1, 1, 0, 0);
            }
            save_image_png(im, "truth11");
        }
        */
        printf("load data spend %f \n", what_time_is_it_now() - time);
        train_network_detect(net, train);
        free_batch_detect(train);
        //sleep(1.5);

        int epoch_old = net->epoch;
        net->epoch = net->seen / train_set_size;
        float loss = net->loss;
        if(loss > 999999 || loss < -999999 || loss != loss || (loss + 1.0 == loss)) {  // NaN ≠ NaN, Inf + 1 = Inf
            //fprintf(stderr, "\n\nloss too large: %f, exit\n", loss);
            //exit(-1);
        }
        if(avg_loss < 0){
            avg_loss = loss;
        } else {
            avg_loss = avg_loss*.9 + loss*.1;
        }
        if(net->correct_num / (net->accuracy_count + 0.00001F) > max_accuracy){
            max_accuracy = net->correct_num / (net->accuracy_count + 0.00001F);
        }
        printf("epoch: %d, batch: %d, accuracy: %.4f, loss: %f, avg_loss: %.2f, learning_rate: %.8f, %.4f s, "
               "seen %lu images\n", net->epoch+1, net->batch_train,
               net->correct_num / (net->accuracy_count + 0.00001F),
               loss, avg_loss, net->learning_rate, what_time_is_it_now()-time, net->seen);
        if(epoch_old != net->epoch){
            int save_weight_times = 20;
            int save_weight_interval = max_epoch / save_weight_times;
            if(save_weight_interval <= 1 || net->epoch % save_weight_interval == 0){
                char buff[256];
                sprintf(buff, "%s/%s_%06d.weights", backup_directory, base, net->epoch);
                save_weights(net, buff);
            }
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    free_network(net);
    if(paths) free_ptr((void *)&paths);
    if(plist){
        free_list_contents(plist);
        free_list(plist);
    }
    free(base);
}
#endif

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;
        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j])
                fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j], xmin/w, ymin/h, xmax/w, ymax/h);
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile)
{
    srand(time(0));
    network *net = load_network(cfgfile, weightfile, 1);

    struct list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "names", "data/names.list");
    char **labels = get_labels(label_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    struct list *plist = get_paths(valid_list);
    char **paths = (char **)list_to_array(plist);
    int valid_set_size = plist->size;
    fprintf(stderr, "valid_set_size: %d, net->classes: %d, net->batch: %d\n", valid_set_size, net->classes, net->batch);
    if(net->batch != 1){
        printf("\nerror: net->batch != 1\n");
        exit(-1);
    }

    FILE **fps = calloc(net->classes, sizeof(FILE *));
    char *outfile = "comp4_det_test_";
    char *prefix = "results";
    char buff[1024];
    for(int j = 0; j < net->classes; ++j){
        snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, labels[j]);
        fps[j] = fopen(buff, "w");
        if(0 == fps[j]){
            printf("file %s not exist\n", buff);
            exit(-1);
        }
    }
    float thresh = .3;
    float nms = .45;
    int *map = 0;
    double start = what_time_is_it_now();
    for(int i = 0; i < valid_set_size; i++){
        int image_original_w, image_original_h;
        image train = load_data_detection_valid(paths[i], net->w, net->h, &image_original_w, &image_original_h);
        double start_forward = what_time_is_it_now();
        forward_network_test(net, train.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, image_original_w, image_original_h, thresh, map, 0, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, net->classes, nms);
        fprintf(stderr, "%d loss: %f, %f Seconds\n", i, net->loss, what_time_is_it_now() - start_forward);
        print_detector_detections(fps, paths[i], dets, nboxes, net->classes, image_original_w, image_original_h);
        free_image(train);
        free_ptr((void *)&dets);
    }
    for(int j = 0; j < net->classes; ++j){
        if(fps) fclose(fps[j]);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
    free_network(net);
    free_ptrs((void**)labels, net->classes);
    if(paths) free_ptr((void *)&paths);
    if(plist){
        free_list_contents(plist);
        free_list(plist);
    }
}

network *net_detect = NULL;


void uninit_detector()
{
    if(net_detect) free_network(net_detect);
    else printf("error: please call init_detector first\n");
}

void init_detector(const char *cfgfile, const char *weightfile)
{
    if(net_detect != NULL){
        printf("error: has call init_detector already\n");
        return;
    }
    srand(time(0));
    net_detect = load_network(cfgfile, weightfile, 1);
    //fprintf(stderr, "net->classes: %d, net->batch: %d\n", net->classes, net->batch);
    if(net_detect->batch != 1){
        printf("\nerror: net->batch != 1\n");
        uninit_detector();
        exit(-1);
    }
#ifdef FORWARD_GPU
    free_network_weight_bias_cpu(net_detect);
#endif
}

void run_detection(float *image_data, int width, int height, int channel, int image_original_w, int image_original_h,
                   int *detection_bbox, int max_bbox_num, int *total_bbox_num)
{
    if(net_detect == NULL){
        printf("error: please call init_detector first\n");
        return;
    }
    float thresh = .6;
    float nms = .45;
    int *map = 0;
    image input;
    input.data = image_data;
    input.w = width;
    input.h = height;
    input.c = channel;
    image train = make_empty_image(net_detect->w, net_detect->h, input.c);
    train.data = malloc(train.h * train.w * train.c * sizeof(float));
    fill_image(train, 0.5f);
    embed_image(input, train, (net_detect->w - input.w) / 2, (net_detect->h - input.h) / 2);

    forward_network_test(net_detect, train.data);
    free_image(train);
    int nboxes = 0;
    detection *dets = get_network_boxes(net_detect, image_original_w, image_original_h, thresh, map, 0, &nboxes);
    if (nms) do_nms_sort(dets, nboxes, net_detect->classes, nms);

    int bbox_num = 0;
    for(int i = 0; i < nboxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;
        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > image_original_w) xmax = image_original_w;
        if (ymax > image_original_h) ymax = image_original_h;
        for(int j = 0; j < net_detect->classes; ++j){
            if (dets[i].prob[j] > thresh && bbox_num < max_bbox_num){
                detection_bbox[bbox_num * 4] = xmin;
                detection_bbox[bbox_num * 4 + 1] = ymin;
                detection_bbox[bbox_num * 4 + 2] = xmax;
                detection_bbox[bbox_num * 4 + 3] = ymax;
                bbox_num += 1;
            }
        }
    }
    *total_bbox_num = bbox_num;
    for(int i = 0; i < nboxes; ++i){
        free_ptr((void *)&(dets[i].prob));
    }
    free_ptr((void *)&dets);
}

void run_detector(int argc, char **argv)
{
    double time_start = what_time_is_it_now();;
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/valid] [data cfg] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if(0==strcmp(argv[2], "train")){
        #ifdef USE_LINUX
        train_detector(datacfg, cfg, weights);
        #endif
    } else if(0==strcmp(argv[2], "valid")){
        validate_detector(datacfg, cfg, weights);
    } else {
        fprintf(stderr, "usage: %s %s [train/valid] [data cfg] [cfg] [weights (optional)]\n", argv[0], argv[1]);
    }
    fprintf(stderr, "\n\ntotal %.2lf seconds\n\n\n", what_time_is_it_now() - time_start);

}
