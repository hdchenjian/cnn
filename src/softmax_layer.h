#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

typedef struct {
    int inputs;
    float *delta;
    float *output;
} softmax_layer;

softmax_layer *make_softmax_layer(int inputs);
void forward_softmax_layer(const softmax_layer layer, float *input);
void backward_softmax_layer(const softmax_layer layer, float *input, float *delta);

#endif
