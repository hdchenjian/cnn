#ifndef PARSER_H
#define PARSER_H
#include "network.h"

struct list *read_data_cfg(char *filename);
struct network *parse_network_cfg(char *filename);
struct list *read_data_cfg(char *filename);

#endif
