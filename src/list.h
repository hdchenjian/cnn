#ifndef LIST_H
#define LIST_H

struct node{
    void *val;
    struct node *next;
    struct node *prev;
};

struct list{
    int size;
    struct node *front;
    struct node *back;
};

void **list_to_array(struct list *l);
void free_list(struct list *l);

struct list *make_list();
int list_find(struct list *l, void *val);

void list_insert(struct list *, void *);


void free_list_contents(struct list *l);

#endif
