#ifndef TREE_H
#define TREE_H

#include "list.h"

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;

int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
float get_hierarchy_probability(float *x, tree *hier, int c, int stride);

#endif
