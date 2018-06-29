#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H

#include "network.h"
#include "blas.h"

route_layer *make_route_layer(int batch, int n, int *input_layers, int *input_size, network *net);

#endif
