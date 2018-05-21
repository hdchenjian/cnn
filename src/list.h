#ifndef LIST_H
#define LIST_H
#include "darknet.h"

struct list *make_list();
int list_find(struct list *l, void *val);

void list_insert(struct list *, void *);


void free_list_contents(struct list *l);

#endif
