#include "convolutional_layer.h"

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride,
		ACTIVATION activation, size_t *workspace_size)
{
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->size = size;
    layer->stride = stride;
    layer->weights = calloc(c*n*size*size, sizeof(float));
    float scale = 1.0F/(size*size*c);
    for(int i = 0; i < c*n*size*size; ++i) layer->weights[i] = scale*(rand_uniform(0, 1));
    layer->weight_updates = calloc(c*n*size*size, sizeof(float));
    layer->weight_momentum = calloc(c*n*size*size, sizeof(float));
    layer->biases = calloc(n, sizeof(float));
    layer->bias_updates = calloc(n, sizeof(float));
    layer->bias_momentum = calloc(n, sizeof(float));
    layer->out_h = (layer->h-1)/layer->stride + 1;
    layer->out_w = (layer->w-1)/layer->stride + 1;
    fprintf(stderr, "Convolutional:      %d x %d x %d inputs, %d weights size: %d -> %d x %d x %d outputs\n",
            h,w,c, n,size, layer->out_h, layer->out_w, n);
    layer->output = calloc(layer->out_h * layer->out_w * n, sizeof(float));
    layer->delta  = calloc(layer->out_h * layer->out_w * n, sizeof(float));
    layer->activation = activation;
    size_t workspace_size_local = (size_t)(layer->out_h*layer->out_w*size*size*c*sizeof(float));
    if (workspace_size_local > *workspace_size) *workspace_size = workspace_size_local;
    return layer;
}

//From Berkeley Vision's Caffe!
void im2col_cpu(float* data_im, const int channels,
        const int height, const int width, const int ksize, const int stride, float* data_col)
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for ( c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for ( h = 0; h < height_col; ++h) {
            for ( w = 0; w < width_col; ++w) {
                data_col[(c * height_col + h) * width_col + w] =
                    data_im[(c_im * height + h * stride + h_offset) * width
                    + w * stride + w_offset];
            }
        }
    }
}

void col2im_cpu(float* data_col, const int channels,
        const int height, const int width, const int ksize, const int stride,
        float* data_im)
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for ( c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for ( h = 0; h < height_col; ++h) {
            for ( w = 0; w < width_col; ++w) {
                data_im[(c_im * height + h * stride + h_offset) * width
                    + w * stride + w_offset]+= data_col[(c * height_col + h) * width_col + w];
            }
        }
    }
}

void forward_convolutional_layer(const convolutional_layer *layer, float *in, float *workspace)
{
    int m = layer->n;
    int n = ((layer->h-layer->size)/layer->stride + 1) * ((layer->w-layer->size)/layer->stride + 1);
    int k = layer->size*layer->size*layer->c;
    memset(layer->output, 0, m*n*sizeof(float));
    float *a = layer->weights;
    float *b = workspace;
    float *c = layer->output;
    im2col_cpu(in,  layer->c,  layer->h,  layer->w,  layer->size,  layer->stride, b);
    gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
    for(int i = 0; i < m*n; ++i) layer->output[i] = activate(layer->output[i], layer->activation);
}

void backward_convolutional_layer(const convolutional_layer *layer, float *input, float *delta, float *workspace)
{
    for(int i = 0; i < layer->out_h * layer->out_w * layer->n; ++i){
    	layer->delta[i] *= gradient(layer->output[i], layer->activation);
    }
    for(int i = 0; i < layer->n; ++i){
        layer->bias_updates[i] += sum_array(layer->delta + layer->out_h * layer->out_w * i, layer->out_h * layer->out_w);
    }

    int m = layer->n;
    int n = layer->size*layer->size*layer->c;
    int k = layer->out_w * layer->out_h;
    float *a = layer->delta;
    float *b = workspace;
    float *c = layer->weight_updates;
    if(layer->size == 1){
        b = input;
    } else {
        im2col_cpu(input, layer->c, layer->h, layer->w, layer->size, layer->stride, b);
    }
    gemm(0,1,m,n,k,1,a,k,b,k,0,c,n);

    if (delta) {  // not first layer
        memset(delta, 0, layer->h * layer->w * layer->c * sizeof(float));
    	m = layer->size*layer->size*layer->c;
    	n = layer->out_w * layer->out_h;
    	k = layer->n;
        a = layer->weights;
        b = layer->delta;
        c = workspace;
        if (layer->size == 1) {
            c = delta;
        }
        gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);
        if (layer->size != 1) {
            col2im_cpu(workspace, layer->c, layer->h, layer->w, layer->size, layer->stride, delta);
        }
    }
}

void update_convolutional_layer(const convolutional_layer *layer, float learning_rate, float momentum, float decay)
{
    int weight_pixel = layer->c*layer->size*layer->size;
    for( int i = 0; i < layer->n; ++i){
        layer->bias_momentum[i] = learning_rate*(layer->bias_updates[i]) + momentum*layer->bias_momentum[i];
        layer->biases[i] += layer->bias_momentum[i];
        layer->bias_updates[i] = 0;
        for(int j = 0; j < weight_pixel; ++j){
        	int index = i * weight_pixel + j;
            layer->weight_momentum[index] = learning_rate*(layer->weight_updates[index] - decay*layer->weights[index])
            		+ momentum*layer->weight_momentum[index];
            layer->weights[index] += layer->weight_momentum[index];
        }
    }
    memset(layer->weight_updates, 0, layer->n * weight_pixel * sizeof(float));
}
