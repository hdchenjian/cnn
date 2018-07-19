#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"

void option_insert(struct list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

void option_unused(struct list *l)
{
    struct node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

char *option_find(struct list *l, char *key)
{
    struct node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
char *option_find_str(struct list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    //if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

int option_find_int(struct list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    //fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

float option_find_float(struct list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    //fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}
