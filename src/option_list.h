#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


void option_insert(struct list *l, char *key, char *val);
char *option_find(struct list *l, char *key);
char *option_find_str(struct list *l, char *key, char *def);
int option_find_int(struct list *l, char *key, int def);
float option_find_float(struct list *l, char *key, float def);
void option_unused(struct list *l);

#endif
