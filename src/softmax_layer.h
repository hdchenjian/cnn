#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include "network.h"

softmax_layer *make_softmax_layer(int inputs, int batch, int is_last_layer);

#endif
