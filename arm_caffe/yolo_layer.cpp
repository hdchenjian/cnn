#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "yolo_layer.h"

void free_ptr(void **ptr)
{
    free(*ptr);
    (*ptr) = NULL;
}

int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    /*
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness < dets[k].objectness){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;
    */
    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

static int entry_index(const yolo_layer *l, int batch, int location, int entry)
{
    int n =   location / (l->w*l->h);
    int loc = location % (l->w*l->h);
    return batch*l->outputs + n*l->w*l->h*(4+l->classes+1) + entry*l->w*l->h + loc;
}

int yolo_num_detections(const yolo_layer *l, float thresh)
{
    int i, n;
    int count = 0;
    //printf("yolo_num_detections %f %f\n", l->output[0], thresh);
    for(i = 0; i < l->w*l->h; ++i){
        for(n = 0; n < l->n; ++n){
            int obj_index  = entry_index(l, 0, n*l->w*l->h + i, 4);
            if(l->output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

void avg_flipped_yolo(yolo_layer *l)
{
    int i,j,n,z;
    float *flip = l->output + l->outputs;
    for(j = 0; j < l->h; ++j) {
        for(i = 0; i < l->w/2; ++i) {
            for(n = 0; n < l->n; ++n) {
                for(z = 0; z < l->classes + 4 + 1; ++z){
                    int i1 = z*l->w*l->h*l->n + n*l->w*l->h + j*l->w + i;
                    int i2 = z*l->w*l->h*l->n + n*l->w*l->h + j*l->w + (l->w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l->outputs; ++i){
        l->output[i] = (l->output[i] + flip[i])/2.;
    }
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for(i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int get_yolo_detections(yolo_layer *l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l->output;
    if (l->batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for(i = 0; i < l->w*l->h; ++i){
        //printf("%d %d\n", i, l->w*l->h);
        int row = i / l->w;
        int col = i % l->w;
        for(n = 0; n < l->n; ++n){
            int obj_index  = entry_index(l, 0, n*l->w*l->h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            //printf("get_yolo_detections %d %f %f\n", i, l->output[0], thresh);
            int box_index  = entry_index(l, 0, n*l->w*l->h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l->biases, l->mask[n], box_index, col, row, l->w, l->h, netw, neth, l->w*l->h);
            dets[count].objectness = objectness;
            dets[count].classes = l->classes;
            for(j = 0; j < l->classes; ++j){
                int class_index = entry_index(l, 0, n*l->w*l->h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

yolo_layer *make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int layer_index)
{
    yolo_layer *l = (yolo_layer *)calloc(1, sizeof(yolo_layer));;
    l->n = n;
    l->total = total;
    l->batch = batch;
    l->h = h;
    l->w = w;
    l->c = n*(classes + 4 + 1);
    l->out_w = l->w;
    l->out_h = l->h;
    l->out_c = l->c;
    l->classes = classes;
    l->layer_index = layer_index;
    l->biases = (float *)calloc(total*2, sizeof(float));
    for(int i = 0; i < total*2; ++i){
        l->biases[i] = .5;
    }
    l->bias_updates = (float *)calloc(n*2, sizeof(float));
    if(mask) l->mask = mask;
    l->outputs = h*w*n*(classes + 4 + 1);
    l->inputs = l->outputs;
    l->max_boxes = 30;
    l->truths = l->max_boxes * (4 + 1);
    l->delta = (float *)calloc(batch*l->outputs, sizeof(float));
    l->output = (float *)calloc(batch*l->outputs, sizeof(float));
    l->input_cpu = (float *)calloc(batch*l->outputs, sizeof(float));
#ifdef GPU
    l->output_gpu = cuda_make_array(l->output, batch*l->outputs);
    l->delta_gpu = cuda_make_array(l->delta, batch*l->outputs);
#elif defined(OPENCL)
    l->output_cl = cl_make_array(l->output, batch*l->outputs);
    l->delta_cl = cl_make_array(l->delta, batch*l->outputs);
#endif
    fprintf(stderr, "make yolo\n");
    return l;
}

int *parse_yolo_mask(const char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        mask = (int *)calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}

yolo_layer *make_yolo_snpe(int c, int h, int w, const char *mask_str, const char *anchors)
{
    int count = 0;
    int total = 6;
    int batch = 1;
    int classes = 1;
    int num = 0;
    int *mask = parse_yolo_mask(mask_str, &num);
    yolo_layer *l = make_yolo_layer(batch, w, h, num, total, mask, classes, count);
    assert(l->outputs == w * h * c);

    l->ignore_thresh = 0.7;
    l->truth_thresh = 1;
    if(anchors){
        int len = strlen(anchors);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (anchors[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(anchors);
            l->biases[i] = bias;
            anchors = strchr(anchors, ',')+1;
        }
    }
    return l;
}

static inline float logistic_activate(float x){return 1./(1. + expf(-x));}

void activate_array(float *x, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = logistic_activate(x[i]);
    }
}

void forward_yolo_layer(const yolo_layer *l, void *net, float *input, int test)
{
    memset(l->delta, 0, l->outputs * l->batch * sizeof(float));
    memcpy(l->output, input, l->outputs * l->batch * sizeof(float));

#if !defined(GPU) && !defined(OPENCL)
    for(int b = 0; b < l->batch; ++b){
        for(int n = 0; n < l->n; ++n){
            int index = entry_index(l, b, n*l->w*l->h, 0);
            activate_array(l->output + index, 2*l->w*l->h);
            index = entry_index(l, b, n*l->w*l->h, 4);
            activate_array(l->output + index, (1+l->classes)*l->w*l->h);
        }
    }
#endif

    if(0 != test) return;    // 0: train, 1: valid
}
