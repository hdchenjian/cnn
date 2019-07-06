#include <unistd.h>
#include <sys/time.h>
#include <assert.h>

#include <locale.h>
#include <wchar.h>
#include <stdio.h>

#include "utils.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"
#include "network.h"

size_t rand_size_t() {
    return  ((size_t)(rand()&0xff) << 56) | 
        ((size_t)(rand()&0xff) << 48) |
        ((size_t)(rand()&0xff) << 40) |
        ((size_t)(rand()&0xff) << 32) |
        ((size_t)(rand()&0xff) << 24) |
        ((size_t)(rand()&0xff) << 16) |
        ((size_t)(rand()&0xff) << 8) |
        ((size_t)(rand()&0xff) << 0);
}

typedef struct {
    float *x;
    int *y;
} float_pair;

float_pair get_rnn_data(wint_t *text, size_t *offsets, int inputs, size_t train_set_size, int batch, int steps)
{
    float *x = calloc(batch * steps * inputs, sizeof(float));
    int *y = calloc(batch * steps, sizeof(int));
//#pragma omp parallel for
    for(int j = 0; j < steps; ++j){
        for(int i = 0; i < batch; ++i){
            //offsets[i] = 0;
            int curr = (int)text[offsets[i] % train_set_size];
            int next = (int)text[(offsets[i] + 1) % train_set_size];
            x[(j*batch + i)*inputs + curr] = 1;
            y[j*batch + i] = next;
            //printf("%d %d %lc %d %lc", i, curr, curr, next, next);
            offsets[i] = (offsets[i] + 1) % train_set_size;
        }
    }
    float_pair p;
    p.x = x;
    p.y = y;
    return p;
}

wint_t *parse_tokens(char *filename, size_t *n)
{
    size_t count = 0;
    setlocale(LC_ALL, "");
    FILE *fp = fopen(filename, "r");
    wint_t c;
    //int min = 99999999;
    //int max = -1;
    while((c = fgetwc(fp)) != WEOF){
        //int temp = c;
        //if((int)c < min) min = c;
        //if((int)c > max) max = c;
        //if(c == '\n') break;
        //printf("%lc %d %d %d\n",c, c, temp, max);
        ++count;
    }
    //printf("min %d, max %d\n", min, max);
    fclose(fp);
    *n = count;

    wint_t *text = malloc(sizeof(wint_t) * count);
    int index = 0;
    fp = fopen(filename, "r");
    while((c = fgetwc(fp)) != WEOF){
        text[index] = (int)c;
        index += 1;
    }
    fclose(fp);
    return text;
}

void train_char_rnn(char *cfgfile, char *weightfile, char *filename)
{
    srand(time(0));
    size_t train_set_size = 0;
    wint_t *text = parse_tokens(filename, &train_set_size);
    char *backup_directory = "backup/";
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    net->output_layer = net->n - 1;
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g, Inputs: %d, batch: %d, time_steps: %d, classes: %d\n",
            net->learning_rate, net->momentum, net->decay, net->inputs, net->batch, net->time_steps, net->classes);
    int max_epoch = (int)net->max_batches * net->batch / train_set_size;
    int save_epoch = 1;
    if(max_epoch / 10 > 1) save_epoch = max_epoch / 20;
    fprintf(stderr, "%s: train data size %lu, max_batches: %d, max epoch: %d\n",
            base, train_set_size, net->max_batches, max_epoch);
    size_t *offsets = calloc(net->batch, sizeof(size_t));
    for(int j = 0; j < net->batch; ++j){
        offsets[j] = rand_size_t() % train_set_size;
    }

    net->batch_train = net->seen / (net->batch * net->time_steps);
    net->epoch = net->seen / train_set_size;
    clock_t time;
    float avg_loss = -1;
    float max_accuracy = -1;
    int max_accuracy_batch = 0;
    while(net->batch_train < net->max_batches){
        update_current_learning_rate(net);
        time=clock();
        float_pair p = get_rnn_data(text, offsets, net->inputs, train_set_size, net->batch, net->time_steps);
        //for(int i = 0; i < 10; ++i) printf("%d %f %d\n", i, p.x[i], p.y[i]);
        train_network(net, p.x, p.y);
        free(p.x);
        free(p.y);
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
        int epoch_old = net->epoch;
        net->epoch = net->seen / train_set_size;
        if(net->correct_num / (net->accuracy_count + 0.00001F) > max_accuracy){
            max_accuracy = net->correct_num / (net->accuracy_count + 0.00001F);
            max_accuracy_batch = net->batch_train;
        }

        printf("epoch: %d, batch: %d: accuracy: %.4f loss: %.4f, avg_loss: %.4f, "
                "learning_rate: %.8f, %.3lfs, seen: %lu, max_accuracy: %.4f\n",
                net->epoch+1, net->batch_train, net->correct_num / (net->accuracy_count + 0.00001F),
                loss, avg_loss, net->learning_rate, sec(clock()-time), net->seen, max_accuracy);

        for(int j = 0; j < net->batch; ++j){
            if(rand()%64 == 0){
                offsets[j] = rand_size_t() % train_set_size;
                reset_rnn_state(net, j);
            }
        }
        if(epoch_old != net->epoch && (net->epoch) % save_epoch == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%06d.weights", backup_directory, base, net->epoch);
            save_weights(net, buff);
        }
    }
    printf("max_accuracy_batch: %d\n", max_accuracy_batch);
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    free_ptr((void *)&offsets);
    free_ptr((void *)&text);
}

void test_char_rnn(char *cfgfile, char *weightfile, int num, char *seed)
{
    setlocale(LC_ALL, "");
    srand(time(0));
    network *net = load_network(cfgfile, weightfile, 1);
    //net->test = 1;      // 0: train, 1: valid

    int c = 0;
    int len = strlen(seed);
    float *input = calloc(net->inputs, sizeof(float));

    printf("seed string:");
    for(int i = 0; i < len-1; ++i){
        c = seed[i];
        input[c] = 1;
        forward_network_test(net, input);
        input[c] = 0;
        printf("%lc", c);
    }
    if(len) c = seed[len-1];
    printf("%lc", c);
    printf("\nseed string over, generate start:\n");
    for(int i = 0; i < num; ++i){
        input[c] = 1;
        forward_network_test(net, input);
#ifndef GPU
        float *out = get_network_layer_data(net, net->output_layer, 0, 0);
#else
        int network_output_size = get_network_output_size_layer(net, net->output_layer);
        float *network_output_gpu = get_network_layer_data(net, net->output_layer, 0, 1);
        float *out = malloc(network_output_size * sizeof(float));
        cuda_pull_array(network_output_gpu, out, network_output_size);
#endif
        float max = -FLT_MAX;
        float min = FLT_MAX;
        for(int j = 0; j < net->classes; ++j){
            if(out[j] > max) max = out[j];
            if(out[j] < min) min = out[j];
            //printf("%d %lc %f\n",j, j, out[j]);
        }
        //printf("test_char_rnn max: %.10f, min: %.10f\n", max, min);

        input[c] = 0;
        if(1){
            for(int j = 0; j < net->inputs; ++j){
                if (out[j] < .001) out[j] = 0;
            }
            c = sample_array(out, net->inputs);
        } else {
            float max = -FLT_MAX;
            int max_index = -1;
            for(int j = 0; j < net->inputs; ++j){
                if (out[j] > max){
                    max = out[j];
                    max_index = j;
                }
            }
            c = max_index;
        }
        printf("%lc", c);
#ifdef GPU
        free_ptr((void *)&out);
#endif
    }
    printf("\n");
    free_ptr((void *)&input);
}

void run_char_rnn(int argc, char **argv)
{
    double time_start = what_time_is_it_now();;
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/generate] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *filename = find_char_arg(argc, argv, "-data", "data/shakespeare.txt");
    char *seed = find_char_arg(argc, argv, "-seed", "\n\n");
    int len = find_int_arg(argc, argv, "-len", 20);

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")){
        train_char_rnn(cfg, weights, filename);
    } else if(0==strcmp(argv[2], "generate")){
        test_char_rnn(cfg, weights, len, seed);
    } else {
        fprintf(stderr, "usage: %s %s [train/generate] [cfg] [weights (optional)]\n", argv[0], argv[1]);
    }
    fprintf(stderr, "\n\ntotal %.2lf seconds\n\n\n", what_time_is_it_now() - time_start);
}
