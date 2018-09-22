#ifndef COST_LAYER_H
#define COST_LAYER_H

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "utils.h"
#include "network.h"

enum COST_TYPE get_cost_type(char *s);
char *get_cost_string(enum COST_TYPE a);

#endif
