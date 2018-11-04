#define TS 16

__kernel void axpy_cl(int N, float ALPHA, __global float *X, int INCX, __global float *Y, int INCY)
{
    int i = get_global_id(0);
    Y[i*INCY] += ALPHA*X[i*INCX];
}

__kernel void scal_cl(int N, float ALPHA, __global float *X, int INCX)
{
    int i = get_global_id(0);
    X[i*INCX] *= ALPHA;
}

__kernel void mask_cl(int n, __global float *x, __global float *mask, int mod)
{
    int i = get_global_id(0);
    x[i] = (i%mod && !mask[(i/mod)*mod]) ? 0 : x[i];
}

__kernel void copy_cl(int N, __global float *X, int INCX, __global float *Y, int INCY)
{
    int i = get_global_id(0);
    Y[i*INCY] = X[i*INCX];
}

__kernel void im2col_cl(__global float *data_im, int offset, int height, int width, int ksize, int pad, int stride,
                        int height_col, int width_col, __global float *data_col)
{
    int index = get_global_id(0);
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    int data_col_index = (channel_out * height_col + h_out) * width_col + w_out;
    int data_im_index = offset + (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            int h = h_in + i;
            int w = w_in + j;
            data_col[data_col_index] = (h >= 0 && w >= 0 && h < height && w < width) ? data_im[data_im_index + i * width + j] : 0;
            data_col_index += height_col * width_col;
        }
    }
}

__kernel void convolutional_bias_cl(int n, int size, __global float *biases, __global float *output)
{
    int id = get_global_id(0);
    int batch = get_global_id(1);
    int filter = id/size;
    output[batch*n*size + id] += biases[filter];
}

__kernel void gemm_tn(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int a_off, int lda, 
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc)
{
    A += a_off;
    B += b_off;
    C += c_off;
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    int col = get_global_id(0);
    int row = get_global_id(1);

    int col_block = get_group_id(0);
    int row_block = get_group_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float val = 0;
    float orig = C[row*ldc + col];

    for(i = 0; i < K; i += TS){
        
        int arow = y + i;
        int acol = x + row_block*TS;

        int brow = y + i;
        int bcol = col;

        arow = (arow < K) ? arow : K-1;
        acol = (acol < M) ? acol : M-1;
        brow = (brow < K) ? brow : K-1;
        
        int aind = arow*lda + acol;
        int bind = brow*ldb + bcol;
        
        Asub[x][y] = A[aind];
        Bsub[y][x] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(j = 0; j < TS && i+j<K; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

__kernel void gemm_nt(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int a_off, int lda, 
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc)
{
    A += a_off;
    B += b_off;
    C += c_off;
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    
    int col = get_global_id(0);
    int row = get_global_id(1);

    int col_block = get_group_id(0);
    int row_block = get_group_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float val = 0;
    float orig = C[row*ldc + col];

    for(i = 0; i < K; i += TS){
        
        int arow = row;
        int acol = x + i;

        int brow = col_block*TS + y;
        int bcol = x + i;

        brow = (brow < N) ? brow : N-1;
        acol = (acol < K) ? acol : K-1;
        bcol = (bcol < K) ? bcol : K-1;
        
        int aind = arow*lda + acol;
        int bind = brow*ldb + bcol;
        
        Asub[y][x] = A[aind];
        Bsub[x][y] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(j = 0; j < TS && i+j<K; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

__kernel void gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int a_off, int lda, 
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc)
{
    A += a_off;
    B += b_off;
    C += c_off;
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    int col = get_global_id(0);
    int row = get_global_id(1);

    col = (col < N) ? col : N - 1;
    row = (row < M) ? row : M - 1;

    int x = get_local_id(0);
    int y = get_local_id(1);

    int i,j;

    float orig = C[row*ldc+col];
    float val = 0;
    
    for(i = 0; i < K; i += TS){
        
        int arow = row;
        int acol = x + i;

        int brow = y + i;
        int bcol = col;

        acol = (acol < K) ? acol : K-1;
        brow = (brow < K) ? brow : K-1;
        
        int aind = arow*lda + acol;
        int bind = brow*ldb + bcol;
        
        Asub[y][x] = A[aind];
        Bsub[y][x] = B[bind];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(j = 0; j < TS && i+j<K; ++j){
            val += Asub[y][j]*Bsub[j][x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row*ldc+col] = ALPHA*val + BETA*orig;
}

__kernel void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    __global float *A, int a_off, int lda, 
                    __global float *B, int b_off, int ldb,
                    float BETA,
                    __global float *C, int c_off, int ldc)
{
    A += a_off;
    B += b_off;
    C += c_off;
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float val = 0;
    
    int row_block = get_group_id(1);
    int col_block = get_group_id(0);

    int sub_row = get_local_id(1);
    int sub_col = get_local_id(0);

    int row = row_block*TS + sub_row;
    int col = col_block*TS + sub_col;

    int i,j;
    for(i = 0; i < K; i += TS){
        int arow = row_block*TS + sub_row;
        int acol = i + sub_col;

        int brow = i + sub_row;
        int bcol = col_block*TS + sub_col;

        if(arow < M && acol < K)Asub[sub_row][sub_col] = TA ? A[arow + acol*lda] : A[arow*lda + acol];
        if(brow < K && bcol < N)Bsub[sub_row][sub_col] = TB ? B[brow + bcol*ldb] : B[brow*ldb + bcol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(j = 0; j < TS && i+j<K; ++j){
            val += Asub[sub_row][j]*Bsub[j][sub_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(row < M && col < N){
        C[row*ldc+col] = ALPHA*val + BETA*C[row*ldc+col];
    }
}

typedef enum{
    LOGISTIC, RELU, PRELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;

float linear_activate(float x){return x;}
float logistic_activate(float x){return 1./(1. + exp(-x));}
float relu_activate(float x){return x*(x>0);}
float ramp_activate(float x){return x*(x>0)+.1*x;}
float leaky_activate(float x){return (x>0) ? x : .1*x;}

float linear_gradient(float x){return 1;}
float logistic_gradient(float x){return (1-x)*x;}
float relu_gradient(float x){return (x>0);}
float ramp_gradient(float x){return (x>0)+.1;}
float leaky_gradient(float x){return (x>0) ? 1 : .1;}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case RELU:
            return relu_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        default:
            return relu_activate(x);
    }
    return 0;
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case RELU:
            return relu_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        default:
            return relu_gradient(x);
    }
    return 0;
}

__kernel void activate_array_cl(__global float *x, int n, ACTIVATION a)
{
    int i = get_global_id(0);
    x[i] = activate(x[i], a);
}

__kernel void activate_array_with_offset_cl(__global float *x, int offset, int n, ACTIVATION a)
{
    int i = get_global_id(0);
    int index = offset + i;
    x[index] = activate(x[index], a);
}

__kernel void gradient_array_cl(__global float *x, int n, ACTIVATION a, __global float *delta)
{
    int i = get_global_id(0);
    delta[i] *= gradient(x[i], a);
}

__kernel void scale_bias_cl(__global float *output, __global float * biases, int batch, int n, int size){
    int i = get_global_id(0);
    int batch_local = i / (n * size);
    int tmp = i % (n * size);
    int c = tmp / size;
    int index = tmp % size;
    output[(batch_local*n + c)*size + index] *= biases[c];
}

__kernel void normalize_cl(__global float *x, __global float *mean, __global float *variance, int batch, int filters, int spatial){
    int index = get_global_id(0);
    int f = (index/spatial)%filters;
    x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
}

__kernel void activate_prelu_array_cl(__global float *x, __global float *slope_cl, int filters, int spatial){
    int i = get_global_id(0);
    int cc = (i / spatial) % filters;
    x[i] = max(x[i], 0.0F) + slope_cl[cc] * min(x[i], 0.0F);
}

__kernel void shortcut_cl(int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1,
                          __global float *add, int w2, int h2, int c2, float s1, float s2, __global float *out){
    int id = get_global_id(0);
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = s1*out[out_index] + s2*add[add_index];
}

__kernel void forward_maxpool_layer_cl(int in_h, int in_w, int in_c, int stride, int size, int pad,
                                       __global float *input, __global float *output, __global int *indexes)
{
    int id = get_global_id(0);
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;

    int j = id % w;  // width
    id /= w;
    int i = id % h;  // height
    id /= h;
    int k = id % c;  // channel
    id /= c;
    int b = id;      // batch

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                         cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

__kernel void upsample_cl(__global float *x, int w, int h, int c, int batch, int stride, int forward,
                          float scale, __global float *out)
{
    int i = get_global_id(0);
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;
    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;
    if(forward) out[out_index] = scale * x[in_index];
    //else atomicAdd(x+in_index, scale * out[out_index]);
}
