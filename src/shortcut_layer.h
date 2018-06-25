#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include <assert.h>

#include "network.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

shortcut_layer *make_shortcut_layer(int batch, int index, int w, int h, int c, int out_w,int out_h,int out_c,
                                    ACTIVATION activation, float prev_layer_weight, float shortcut_layer_weight);

#endif
