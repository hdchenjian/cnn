#include "blas.h"
#include "math.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    if(batch * spatial == 1) scale = 1.0f;
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void backward_l2normalize_cpu(int batch, int filters, int spatial, float *norm_data, float *output, float *delta, float *previous_delta)
{
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < spatial; ++i){
            float a = 0;
            for(int j = 0; j < filters; j++){
                int index = b * filters * spatial + i + j * spatial;
                a += output[index] * delta[index];
            }
            float norm_data_tmp = norm_data[b * spatial + i];
            for(int f = 0; f < filters; ++f){
                int index_delta = (b * filters + f) * spatial + i;
                previous_delta[index_delta] += (delta[index_delta] - output[index_delta] * a) / norm_data_tmp;
            }
        }
    }
}

void l2normalize_cpu(float *x, int batch, int filters, int spatial, float *norm_data)
{
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < spatial; ++i){
            float sum = 1e-6;
            for(int f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                sum += powf(x[index], 2);
            }
            sum = sqrtf(sum);
            for(int f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                x[index] /= sum;
            }
            norm_data[b * spatial + i] = sum;
        }
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/sqrtf(variance[f] + .00002f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    for(int i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    if(INCX == 1 && ALPHA == 0){
        memset(X, 0, N * sizeof(float));
    } else {
        for(int i = 0; i < N; ++i) X[i*INCX] = ALPHA;
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    if(INCX == 1 && INCY == 1){
        memcpy(Y, X, N * sizeof(float));
    } else {
        for(int i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
    }
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabsf(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? -1 : 1;
        }
    }
}

void l2_cpu(int batch, int n, float *pred, int *truth_label_index, float *delta, float *error)
{
    float diff = 0.0F;
    for(int b = 0; b < batch; ++b){
        int index = b * n;
        for(int i = 0; i < n; ++i){
            if(truth_label_index[b] == i){
                diff = 1.0F - pred[i + index];
            } else {
                diff = 0.0F - pred[i + index];
            }
            error[i + index] = diff * diff;
            delta[i + index] = diff;
            //printf("%d %d %f %f\n", i, i == truth_label_index[b], pred[i + index], delta[i + index]);
        }
    }
    //printf("\n");
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax_x_ent_cpu(int batch, int n, float *pred, int *truth, float *delta, float *error)
{
    for(int b = 0; b < batch; ++b){
        int index = b * n;
        for(int i = 0; i < n; ++i){
            float t;
            if(i == truth[b]){
                t = 1.0F;
            } else {
                t = 0.0F;
            }
            float p = pred[i + index];
            error[i + index] = (t > 0.01) ? -log(p) : 0;
            delta[i + index] = t-p;
            //printf("truth: %d %f %f %f\n", i, truth[i], pred[i], error[i]);
        }
    }
}

void weighted_delta_cpu(int num, float *state, float *h, float *z, float *delta_state, float *delta_h, float *delta_z, float *delta)
{
    for(int i = 0; i < num; ++i){
        if(delta_state) delta_state[i] = delta[i] * (1 - z[i]);
        delta_h[i] = delta[i] * z[i];
        delta_z[i] = delta[i] * (h[i] - state[i]);
    }
}

void mult_add_into_cpu(int num, float *a, float *b, float *c)
{
    for(int i = 0; i < num; ++i){
        c[i] += a[i]*b[i];
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}


void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrtf(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters,
                         int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta,
                         int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrtf(variance[f] + .00001f)) +
                    variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}
