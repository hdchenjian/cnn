#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "network.h"

typedef struct {
	float scale;
    int batch;
    int inputs;
    int outputs;
    float *delta;
    float *output;
    enum COST_TYPE cost_type;
    float *cost;
} cost_layer;

enum COST_TYPE get_cost_type(char *s);
char *get_cost_string(enum COST_TYPE a);
cost_layer *make_cost_layer(int batch, int inputs, enum COST_TYPE type, float scale);
void forward_cost_layer(const cost_layer *l, network_state state);
void backward_cost_layer(const cost_layer *l, network_state state);
void resize_cost_layer(cost_layer *l, int inputs);

#ifdef GPU
void forward_cost_layer_gpu(cost_layer l, network_state state);
void backward_cost_layer_gpu(const cost_layer l, network_state state);
#endif

#endif
