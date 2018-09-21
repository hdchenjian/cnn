#include <unistd.h>
#include <sys/time.h>
#include <assert.h>

#include "utils.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"
#include "network.h"

void train_detector(char *datacfg, char *cfgfile, char *weightfile)
{
    char *base = basecfg(cfgfile);
    srand(time(0));
    network *net = load_network(cfgfile, weightfile);
    net->output_layer = net->n - 1;

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
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    int max_epoch = (int)net->max_batches * net->batch / train_set_size;
    printf("image net has seen: %lu, train_set_size: %d, max_batches of net: %d, net->classes: %d,"
           "net->batch: %d, max_epoch: %d\n\n",
           net->seen, train_set_size, net->max_batches, net->classes, net->batch, max_epoch);

    net->batch_train = net->seen / net->batch / net->subdivisions;
    net->epoch = net->seen / train_set_size;
    float avg_loss = -1;
    float max_accuracy = -1;
    while(net->batch_train < net->max_batches){
        time = what_time_is_it_now();
        update_current_learning_rate(net);
        batch_detect train;
        train = load_data_detection(net->batch * net->subdivisions, paths, train_set_size, net->w, net->h, net->max_boxes,
                                    net->classes, net->jitter, net->hue, net->saturation, net->exposure, net->test);
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
               "seen %lu images, max_accuracy: %.4f\n", net->epoch+1, net->batch_train,
               net->correct_num / (net->accuracy_count + 0.00001F),
               loss, avg_loss, net->learning_rate, what_time_is_it_now()-time, net->seen,  max_accuracy);
        if(epoch_old != net->epoch){
            int save_weight_times = 15;
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
    if(paths) free_ptr(paths);
    if(plist){
        free_list_contents(plist);
        free_list(plist);
    }
    free(base);
}
/*
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
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin/w, ymin/h, xmax/w, ymax/h);
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    char *outfile = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .3;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            print_detector_detections(fps, path, dets, nboxes, classes, w, h);
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}
*/
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
        train_detector(datacfg, cfg, weights);
    } else if(0==strcmp(argv[2], "valid")){
        //validate_detector(datacfg, cfg, weights);
    } else if(0==strcmp(argv[2], "recall")){
        //validate_detector_recall(cfg, weights);
    } else {
        fprintf(stderr, "usage: %s %s [train/valid] [data cfg] [cfg] [weights (optional)]\n", argv[0], argv[1]);
    }
    fprintf(stderr, "\n\ntotal %.2lf seconds\n\n\n", what_time_is_it_now() - time_start);

}
