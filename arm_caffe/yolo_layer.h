typedef struct {
    int h,w,c, out_h, out_w, out_c, n, batch, total, classes, inputs, outputs, truths, max_boxes, layer_index;
    int *mask;
    float ignore_thresh, truth_thresh;
    float *biases, *bias_updates, *delta, *output, *input_cpu;
    float *output_gpu, *delta_gpu;
#ifdef OPENCL
    cl_mem output_cl, delta_cl;
#endif
} yolo_layer;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

