#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <float.h>

#include "network.h"
#include "blas.h"

softmax_layer *make_softmax_layer(int inputs, int batch, int is_last_layer, float label_specific_margin_bias, int margin_scale);

#endif
