#ifndef PARSER_H
#define PARSER_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "network.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "softmax_layer.h"
#include "cost_layer.h"
#include "list.h"
#include "option_list.h"
#include "utils.h"

struct list *read_data_cfg(char *filename);
struct network *parse_network_cfg(char *filename);
struct list *read_data_cfg(char *filename);

#endif
