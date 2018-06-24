#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "network.h"

dropout_layer *make_dropout_layer(int w, int h, int c, int batch, int inputs, float probability);
void resize_dropout_layer(dropout_layer *l, int inputs);

#endif
