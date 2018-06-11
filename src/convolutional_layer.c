#include "convolutional_layer.h"
#include <float.h>

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, int batch,
        ACTIVATION activation, size_t *workspace_size)
{
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->size = size;
    layer->stride = stride;
    layer->batch = batch;
    layer->weights = calloc(c*n*size*size, sizeof(float));
    float scale = sqrt(2.0F/(size*size*c));
    for(int i = 0; i < c*n*size*size; ++i) layer->weights[i] = scale*rand_normal();
    //scale = 1.0F/(size*size*c);
    //for(int i = 0; i < c*n*size*size; ++i) layer->weights[i] = scale*rand_uniform(0, 1);

    layer->weight_updates = calloc(c*n*size*size, sizeof(float));
    layer->biases = calloc(n, sizeof(float));
    layer->bias_updates = calloc(n, sizeof(float));
    layer->out_h = (layer->h-1)/layer->stride + 1;
    layer->out_w = (layer->w-1)/layer->stride + 1;
    fprintf(stderr, "Convolutional:      %d x %d x %d inputs, %d weights size: %d -> %d x %d x %d outputs\n",
            h,w,c, n,size, layer->out_h, layer->out_w, n);
    layer->output = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
    layer->delta  = calloc(batch * layer->out_h * layer->out_w * n, sizeof(float));
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
    int n = layer->out_h * layer->out_w;
    int k = layer->size*layer->size*layer->c;
    memset(layer->output, 0, layer->batch * m*n*sizeof(float));
    for(int i = 0; i < layer->batch; ++i){
		float *a = layer->weights;
		float *b = workspace;
		float *c = layer->output + i * m * n;
		im2col_cpu(in,  layer->c,  layer->h,  layer->w,  layer->size,  layer->stride, b);
		gemm(0,0,m,n,k,1,a,k,b,n,0,c,n);
    }

	for(int b = 0; b < layer->batch; ++b){
		for(int i = 0; i < layer->n; ++i){
			for(int j = 0; j < n; ++j){
				layer->output[(b*layer->n + i)*n + j] += layer->biases[i];
			}
		}
	}
    for(int i = 0; i < layer->batch * m*n; ++i) layer->output[i] = activate(layer->output[i], layer->activation);
    float max = -FLT_MAX;
    float min = FLT_MAX;
    for(int i = 0; i < layer->batch * m*n; ++i){
    	if(layer->output[i] > max) max = layer->output[i];
    	if(layer->output[i] < min) min = layer->output[i];
    }
    //printf("forward_convolutional_layer max: %f, min: %f\n", max, min);
}

void backward_convolutional_layer(const convolutional_layer *layer, float *input, float *delta, float *workspace)
{
	int outputs = layer->batch * layer->out_h * layer->out_w * layer->n;
    for(int i = 0; i < outputs; ++i){
        layer->delta[i] *= gradient(layer->output[i], layer->activation);
    }
	for(int j = 0; j < layer->batch; ++j){
		for(int i = 0; i < layer->n; ++i){
			layer->bias_updates[i] += sum_array(layer->delta + layer->out_h * layer->out_w * (i + j*layer->n),
					layer->out_h * layer->out_w);
		}
		int m = layer->n;
		int n = layer->size*layer->size*layer->c;
		int k = layer->out_w * layer->out_h;
		float *a = layer->delta + j * m * k;
		float *b = workspace;
		float *c = layer->weight_updates;
		float *im  = input + j*layer->c*layer->h*layer->w;
		if(layer->size == 1){
			b = im;
		} else {
			im2col_cpu(im, layer->c, layer->h, layer->w, layer->size, layer->stride, b);
		}
		gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

		if (delta) {  // not first layer
			memset(delta + j * layer->h * layer->w * layer->c, 0, layer->h * layer->w * layer->c * sizeof(float));
			m = layer->size*layer->size*layer->c;
			n = layer->out_w * layer->out_h;
			k = layer->n;
			a = layer->weights;
			b = layer->delta + j * m * k;
			c = workspace;
			if (layer->size == 1) {
				c = delta + j * layer->h * layer->w * layer->c;
			}
			gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);
			if (layer->size != 1) {
				col2im_cpu(workspace, layer->c, layer->h, layer->w, layer->size, layer->stride,
						delta + j * layer->h * layer->w * layer->c);
			}
		}
	}
}

void update_convolutional_layer(const convolutional_layer *layer, float learning_rate, float momentum, float decay)
{
    for(int i = 0; i < layer->n; i ++){
    	layer->biases[i] += learning_rate / layer->batch * layer->bias_updates[i];
    	layer->bias_updates[i] *= momentum;
    }

    int size = layer->size*layer->size*layer->c*layer->n;
    for(int i = 0; i < size; i ++){
    	layer->weight_updates[i] += -decay*layer->batch*layer->weights[i];
    	layer->weights[i] += learning_rate / layer->batch * layer->weight_updates[i];
    	layer->weight_updates[i] *= momentum;
    }
}
