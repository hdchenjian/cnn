#include <unistd.h>
#include <sys/time.h>
#include <assert.h>

#include "utils.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"
#include "network.h"

typedef struct {
    float *x;
    int *y;
} float_pair;

float_pair get_rnn_data(unsigned char *text, size_t *offsets, int inputs, size_t len, int batch, int steps)
{
    float *x = calloc(batch * steps * inputs, sizeof(float));
    int *y = calloc(batch * steps, sizeof(int));
#pragma omp parallel for
    for(int j = 0; j < steps; ++j){
        for(int i = 0; i < batch; ++i){
            unsigned char curr = text[offsets[i] % len];
            unsigned char next = text[(offsets[i] + 1) % len];
            x[(j*batch + i)*inputs + curr] = 1;
            y[j*batch + i] = next;
            offsets[i] = (offsets[i] + 1) % len;
        }
    }
    float_pair p;
    p.x = x;
    p.y = y;
    return p;
}

void train_char_rnn(char *cfgfile, char *weightfile, char *filename)
{
    srand(time(0));
    unsigned char *text = read_file(filename);
    size_t train_set_size = strlen((const char*)text);
    char *backup_directory = "backup/";
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile);
    net->output_layer = net->n - 1;
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g, Inputs: %d batch: %d, time_steps: %d, classes: %d\n",
            net->learning_rate, net->momentum, net->decay, net->inputs, net->batch, net->time_steps, net->classes);
    fprintf(stderr, "%s: train data size %lu\n", base, train_set_size);

    size_t *offsets = calloc(net->batch, sizeof(size_t));
    for(int j = 0; j < net->batch; ++j){
        offsets[j] = rand_size_t() % train_set_size;
    }

    clock_t time;
    float avg_loss = -1;
    float max_accuracy = -1;
    int max_accuracy_batch = 0;
    while(net->batch_train < net->max_batches){
        update_current_learning_rate(net);
        time=clock();
        float_pair p = get_rnn_data(text, offsets, net->inputs, train_set_size, net->batch, net->time_steps);

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
        net->epoch = net->seen / train_set_size;
        if(net->correct_num / (net->correct_num_count + 0.00001F) > max_accuracy){
            max_accuracy = net->correct_num / (net->correct_num_count + 0.00001F);
            max_accuracy_batch = net->batch_train;
        }

        fprintf(stderr, "epoch: %d, batch: %d: accuracy: %.4f loss: %f, avg_loss: %f, "
                "learning_rate: %.8f, %lf s, seen: %lu, max_accuracy: %.4f\n",
                net->epoch+1, net->batch_train, net->correct_num / (net->correct_num_count + 0.00001F),
                loss, avg_loss, net->learning_rate, sec(clock()-time), net->seen, max_accuracy);

        for(int j = 0; j < net->batch; ++j){
            if(rand()%64 == 0){
                offsets[j] = rand_size_t() % train_set_size;
                reset_rnn_state(net, j);
            }
        }
        if((net->epoch + 1) % 100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%06d.weights", backup_directory, base, net->epoch);
            save_weights(net, buff);
        }
    }
    printf("max_accuracy_batch: %d\n", max_accuracy_batch);
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void print_symbol(int n, char **tokens){
    if(tokens){
        printf("%s ", tokens[n]);
    } else {
        printf("%c", n);
    }
}

char **read_tokens(char *filename, size_t *read)
{
    size_t size = 512;
    size_t count = 0;
    FILE *fp = fopen(filename, "r");
    char **d = calloc(size, sizeof(char *));
    char *line;
    while((line=fgetl(fp)) != 0){
        ++count;
        if(count > size){
            size = size*2;
            d = realloc(d, size*sizeof(char *));
        }
        if(0==strcmp(line, "<NEWLINE>")) line = "\n";
        d[count-1] = line;
    }
    fclose(fp);
    d = realloc(d, count*sizeof(char *));
    *read = count;
    return d;
}

void test_char_rnn(char *cfgfile, char *weightfile, int num, char *seed, float temp, char *token_file)
{
    char **tokens = 0;
    if(token_file){
        size_t n;
        tokens = read_tokens(token_file, &n);
    }

    srand(time(0));
    char *base = basecfg(cfgfile);
    fprintf(stderr, "%s\n", base);

    network *net = load_network(cfgfile, weightfile);
    int inputs = net->inputs;

    int i, j;
    int c = 0;
    int len = strlen(seed);
    float *input = calloc(inputs, sizeof(float));

    for(i = 0; i < len-1; ++i){
        c = seed[i];
        input[c] = 1;
        forward_network_test(net, input);
        input[c] = 0;
        print_symbol(c, tokens);
    }
    if(len) c = seed[len-1];
    print_symbol(c, tokens);
    for(i = 0; i < num; ++i){
        input[c] = 1;
        float *out = forward_network_test(net, input);
        input[c] = 0;
        for(j = 32; j < 127; ++j){
            //printf("%d %c %f\n",j, j, out[j]);
        }
        for(j = 0; j < inputs; ++j){
            if (out[j] < .0001) out[j] = 0;
        }
        c = sample_array(out, inputs);
        print_symbol(c, tokens);
    }
    printf("\n");
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
    int len = find_int_arg(argc, argv, "-len", 1000);
    float temp = find_float_arg(argc, argv, "-temp", .7);

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    if(0==strcmp(argv[2], "train")){
        train_char_rnn(cfg, weights, filename);
    } else if(0==strcmp(argv[2], "generate")){
        char *tokens = find_char_arg(argc, argv, "-tokens", 0);
        test_char_rnn(cfg, weights, len, seed, temp, tokens);
    } else {
        fprintf(stderr, "usage: %s %s [train/generate] [cfg] [weights (optional)]\n", argv[0], argv[1]);
    }
    fprintf(stderr, "\n\ntotal %.2lf seconds\n\n\n", what_time_is_it_now() - time_start);
}
