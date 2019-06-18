#define TS 16
#define T_WIDTH 8
#define T_LOCAL_WIDTH 4
#define LOCAL_SIZE_WIDTH 16

// M is the height of matrix a
__kernel void gemm_with_local_image(int M, int N, int K, __global float *a, __read_only image2d_t b, __global float *c)
{
    local float4 a_sm[T_LOCAL_WIDTH * LOCAL_SIZE_WIDTH];  // 4 * 16
    local float4 b_sm[T_LOCAL_WIDTH * LOCAL_SIZE_WIDTH];
    float c_r[T_LOCAL_WIDTH*T_LOCAL_WIDTH] = {0};
    float4 a_r;
    float4 b_r;

    int group_width = N / T_LOCAL_WIDTH / LOCAL_SIZE_WIDTH;
    int a_off = (get_group_id(0) / group_width) * LOCAL_SIZE_WIDTH + get_local_id(0);
    int b_off = (get_group_id(0) % group_width) * LOCAL_SIZE_WIDTH + get_local_id(0);

    int k, b_index;
    for(k = 0; k < K; k += T_LOCAL_WIDTH ) {
        if(get_local_id(0) < 64) {
            b_index = (b_off + get_local_id(0) / LOCAL_SIZE_WIDTH * (N / T_LOCAL_WIDTH - LOCAL_SIZE_WIDTH)) << 2;
            b_sm[get_local_id(0)] = read_imagef(b, (int2)((b_index % N) / 4 , b_index / N)); //(b_off * 8 / 4, k)
        }
        if(get_local_id(0) < 64) {
            a_sm[get_local_id(0)] = ((__global float4 const *)(a))[a_off + get_local_id(0) / LOCAL_SIZE_WIDTH * (M / T_LOCAL_WIDTH - LOCAL_SIZE_WIDTH)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        a_r = a_sm[get_local_id(0) / LOCAL_SIZE_WIDTH];
        b_r = b_sm[get_local_id(0) % LOCAL_SIZE_WIDTH];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s1*b_r.s0;
        c_r[5] += a_r.s1*b_r.s1;
        c_r[6] += a_r.s1*b_r.s2;
        c_r[7] += a_r.s1*b_r.s3;
        c_r[8] += a_r.s2*b_r.s0;
        c_r[9] += a_r.s2*b_r.s1;
        c_r[10] += a_r.s2*b_r.s2;
        c_r[11] += a_r.s2*b_r.s3;
        c_r[12] += a_r.s3*b_r.s0;
        c_r[13] += a_r.s3*b_r.s1;
        c_r[14] += a_r.s3*b_r.s2;
        c_r[15] += a_r.s3*b_r.s3;
        a_r = a_sm[get_local_id(0) / LOCAL_SIZE_WIDTH + 16];
        b_r = b_sm[get_local_id(0) % LOCAL_SIZE_WIDTH + 16];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s1*b_r.s0;
        c_r[5] += a_r.s1*b_r.s1;
        c_r[6] += a_r.s1*b_r.s2;
        c_r[7] += a_r.s1*b_r.s3;
        c_r[8] += a_r.s2*b_r.s0;
        c_r[9] += a_r.s2*b_r.s1;
        c_r[10] += a_r.s2*b_r.s2;
        c_r[11] += a_r.s2*b_r.s3;
        c_r[12] += a_r.s3*b_r.s0;
        c_r[13] += a_r.s3*b_r.s1;
        c_r[14] += a_r.s3*b_r.s2;
        c_r[15] += a_r.s3*b_r.s3;
        a_r = a_sm[get_local_id(0) / LOCAL_SIZE_WIDTH + 32];
        b_r = b_sm[get_local_id(0) % LOCAL_SIZE_WIDTH + 32];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s1*b_r.s0;
        c_r[5] += a_r.s1*b_r.s1;
        c_r[6] += a_r.s1*b_r.s2;
        c_r[7] += a_r.s1*b_r.s3;
        c_r[8] += a_r.s2*b_r.s0;
        c_r[9] += a_r.s2*b_r.s1;
        c_r[10] += a_r.s2*b_r.s2;
        c_r[11] += a_r.s2*b_r.s3;
        c_r[12] += a_r.s3*b_r.s0;
        c_r[13] += a_r.s3*b_r.s1;
        c_r[14] += a_r.s3*b_r.s2;
        c_r[15] += a_r.s3*b_r.s3;
        a_r = a_sm[get_local_id(0) / LOCAL_SIZE_WIDTH + 48];
        b_r = b_sm[get_local_id(0) % LOCAL_SIZE_WIDTH + 48];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s1*b_r.s0;
        c_r[5] += a_r.s1*b_r.s1;
        c_r[6] += a_r.s1*b_r.s2;
        c_r[7] += a_r.s1*b_r.s3;
        c_r[8] += a_r.s2*b_r.s0;
        c_r[9] += a_r.s2*b_r.s1;
        c_r[10] += a_r.s2*b_r.s2;
        c_r[11] += a_r.s2*b_r.s3;
        c_r[12] += a_r.s3*b_r.s0;
        c_r[13] += a_r.s3*b_r.s1;
        c_r[14] += a_r.s3*b_r.s2;
        c_r[15] += a_r.s3*b_r.s3;

        a_off += M;  // K / T_LOCAL_WIDTH * T_LOCAL_WIDTH;
        b_off += N;  // K / T_LOCAL_WIDTH * T_LOCAL_WIDTH;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    a_off = ((get_group_id(0) / group_width)*LOCAL_SIZE_WIDTH + (get_local_id(0) / LOCAL_SIZE_WIDTH)) * N +
        ((get_group_id(0) % group_width)*LOCAL_SIZE_WIDTH + (get_local_id(0) % LOCAL_SIZE_WIDTH));

    for(k = 0; k < T_LOCAL_WIDTH; ++k ) {
        switch(k) {
        case 0:
            b_r.s0 = c_r[0];
            b_r.s1 = c_r[1];
            b_r.s2 = c_r[2];
            b_r.s3 = c_r[3];
            break;
        case 1:
            b_r.s0 = c_r[4];
            b_r.s1 = c_r[5];
            b_r.s2 = c_r[6];
            b_r.s3 = c_r[7];
            break;
        case 2:
            b_r.s0 = c_r[8];
            b_r.s1 = c_r[9];
            b_r.s2 = c_r[10];
            b_r.s3 = c_r[11];
            break;
        case 3:
            b_r.s0 = c_r[12];
            b_r.s1 = c_r[13];
            b_r.s2 = c_r[14];
            b_r.s3 = c_r[15];
            break;
        }
        ((global float4 *)c)[a_off] = b_r;
        a_off += N / T_LOCAL_WIDTH;
    }
}

// M is the height of matrix a
__kernel void gemm_with_local(int M, int N, int K, __global float *a, __global float *b, __global float *c)
{
    local float4 a_sm[T_LOCAL_WIDTH * LOCAL_SIZE_WIDTH];  // 4 * 16
    local float4 b_sm[T_LOCAL_WIDTH * LOCAL_SIZE_WIDTH];
    float c_r[T_LOCAL_WIDTH*T_LOCAL_WIDTH] = {0};
    float4 a_r;
    float4 b_r;

    int group_width = N / T_LOCAL_WIDTH / LOCAL_SIZE_WIDTH;
    int a_off = (get_group_id(0) / group_width) * LOCAL_SIZE_WIDTH + get_local_id(0);
    int b_off = (get_group_id(0) % group_width) * LOCAL_SIZE_WIDTH + get_local_id(0);

    int k;
    for(k = 0; k < K; k += T_LOCAL_WIDTH ) {
        if(get_local_id(0) < 64) {
            a_sm[get_local_id(0)] = ((__global float4 const *)(a))[a_off + get_local_id(0) / LOCAL_SIZE_WIDTH * (M / T_LOCAL_WIDTH - LOCAL_SIZE_WIDTH)];
        }
        if(get_local_id(0) < 64) {
            b_sm[get_local_id(0)] = ((__global float4 const *)(b))[b_off + get_local_id(0) / LOCAL_SIZE_WIDTH * (N / T_LOCAL_WIDTH - LOCAL_SIZE_WIDTH)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        a_r = a_sm[get_local_id(0) / LOCAL_SIZE_WIDTH];
        b_r = b_sm[get_local_id(0) % LOCAL_SIZE_WIDTH];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s1*b_r.s0;
        c_r[5] += a_r.s1*b_r.s1;
        c_r[6] += a_r.s1*b_r.s2;
        c_r[7] += a_r.s1*b_r.s3;
        c_r[8] += a_r.s2*b_r.s0;
        c_r[9] += a_r.s2*b_r.s1;
        c_r[10] += a_r.s2*b_r.s2;
        c_r[11] += a_r.s2*b_r.s3;
        c_r[12] += a_r.s3*b_r.s0;
        c_r[13] += a_r.s3*b_r.s1;
        c_r[14] += a_r.s3*b_r.s2;
        c_r[15] += a_r.s3*b_r.s3;
        a_r = a_sm[get_local_id(0) / LOCAL_SIZE_WIDTH + 16];
        b_r = b_sm[get_local_id(0) % LOCAL_SIZE_WIDTH + 16];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s1*b_r.s0;
        c_r[5] += a_r.s1*b_r.s1;
        c_r[6] += a_r.s1*b_r.s2;
        c_r[7] += a_r.s1*b_r.s3;
        c_r[8] += a_r.s2*b_r.s0;
        c_r[9] += a_r.s2*b_r.s1;
        c_r[10] += a_r.s2*b_r.s2;
        c_r[11] += a_r.s2*b_r.s3;
        c_r[12] += a_r.s3*b_r.s0;
        c_r[13] += a_r.s3*b_r.s1;
        c_r[14] += a_r.s3*b_r.s2;
        c_r[15] += a_r.s3*b_r.s3;
        a_r = a_sm[get_local_id(0) / LOCAL_SIZE_WIDTH + 32];
        b_r = b_sm[get_local_id(0) % LOCAL_SIZE_WIDTH + 32];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s1*b_r.s0;
        c_r[5] += a_r.s1*b_r.s1;
        c_r[6] += a_r.s1*b_r.s2;
        c_r[7] += a_r.s1*b_r.s3;
        c_r[8] += a_r.s2*b_r.s0;
        c_r[9] += a_r.s2*b_r.s1;
        c_r[10] += a_r.s2*b_r.s2;
        c_r[11] += a_r.s2*b_r.s3;
        c_r[12] += a_r.s3*b_r.s0;
        c_r[13] += a_r.s3*b_r.s1;
        c_r[14] += a_r.s3*b_r.s2;
        c_r[15] += a_r.s3*b_r.s3;
        a_r = a_sm[get_local_id(0) / LOCAL_SIZE_WIDTH + 48];
        b_r = b_sm[get_local_id(0) % LOCAL_SIZE_WIDTH + 48];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s1*b_r.s0;
        c_r[5] += a_r.s1*b_r.s1;
        c_r[6] += a_r.s1*b_r.s2;
        c_r[7] += a_r.s1*b_r.s3;
        c_r[8] += a_r.s2*b_r.s0;
        c_r[9] += a_r.s2*b_r.s1;
        c_r[10] += a_r.s2*b_r.s2;
        c_r[11] += a_r.s2*b_r.s3;
        c_r[12] += a_r.s3*b_r.s0;
        c_r[13] += a_r.s3*b_r.s1;
        c_r[14] += a_r.s3*b_r.s2;
        c_r[15] += a_r.s3*b_r.s3;

        a_off += M;  // K / T_LOCAL_WIDTH * T_LOCAL_WIDTH;
        b_off += N;  // K / T_LOCAL_WIDTH * T_LOCAL_WIDTH;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    a_off = ((get_group_id(0) / group_width)*LOCAL_SIZE_WIDTH + (get_local_id(0) / LOCAL_SIZE_WIDTH)) * N +
        ((get_group_id(0) % group_width)*LOCAL_SIZE_WIDTH + (get_local_id(0) % LOCAL_SIZE_WIDTH));

    for(k = 0; k < T_LOCAL_WIDTH; ++k ) {
        switch(k) {
        case 0:
            b_r.s0 = c_r[0];
            b_r.s1 = c_r[1];
            b_r.s2 = c_r[2];
            b_r.s3 = c_r[3];
            break;
        case 1:
            b_r.s0 = c_r[4];
            b_r.s1 = c_r[5];
            b_r.s2 = c_r[6];
            b_r.s3 = c_r[7];
            break;
        case 2:
            b_r.s0 = c_r[8];
            b_r.s1 = c_r[9];
            b_r.s2 = c_r[10];
            b_r.s3 = c_r[11];
            break;
        case 3:
            b_r.s0 = c_r[12];
            b_r.s1 = c_r[13];
            b_r.s2 = c_r[14];
            b_r.s3 = c_r[15];
            break;
        }
        ((global float4 *)c)[a_off] = b_r;
        a_off += N / T_LOCAL_WIDTH;
    }
}

__kernel void gemm_fast_direct_kk(int M, int N, int K, int local_width,  __global float *a, __global float *b, __global float *c)
{
    float c_r[T_WIDTH * T_WIDTH] = {0};
    float8 a_r;
    float8 b_r;
    int output_width = (N + T_WIDTH - 1) / T_WIDTH;
    int group_width = output_width / local_width;
    group_width = group_width < 1 ? 1 : group_width;
    local_width = output_width < local_width ? output_width : local_width;

    int a_off = ( (get_group_id(0)/group_width)*local_width + (get_local_id(0)/local_width) );
    int b_off = ( (get_group_id(0)%group_width)*local_width + (get_local_id(0)%local_width) );
    int a_y = M - (a_off + 1) * T_WIDTH;
    int b_x = N - (b_off + 1) * T_WIDTH;
    /*printf("get_group_id(0): %d get_local_id(0): %d, %d %d   %d %d %d %d\n", get_group_id(0), get_local_id(0),
           (get_group_id(0)/group_width)*local_width, get_local_id(0)/local_width,
           a_off, b_off, a_y, b_x);*/

    int i, k;
    for( int k = 0; k < K; k += 1 ) {
        if(a_y >= 0 && b_x >= 0){
            a_r = ((global float8 const *)a)[a_off];
            b_r = ((global float8 const *)b)[b_off];
        } else if(a_y >= 0 && b_x < 0){    // right upper
            a_r = ((global float8 const *)a)[a_off];
            b_r = (float8)0;
            for(i = -b_x; i < T_WIDTH; i++){
                *((float *)&b_r + (i + b_x)) = b[b_off * T_WIDTH + (i + b_x)];
            }
        } else if(a_y < 0 && b_x >= 0){
            a_r = (float8)0;
            for(i = -a_y; i < T_WIDTH; i++){
                *((float *)&a_r + (i + a_y)) = a[a_off * T_WIDTH + (i + a_y)];
            }
            b_r = ((global float8 const *)b)[b_off];
        } else {
            a_r = (float8)0;
            b_r = (float8)0;
            for(i = -a_y; i < T_WIDTH; i++){
                *((float *)&a_r + (i + a_y)) = a[a_off * T_WIDTH + (i + a_y)];
            }
            for(i = -b_x; i < T_WIDTH; i++){
                *((float *)&b_r + (i + b_x)) = b[b_off * T_WIDTH + (i + b_x)];
            }
        }

        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s0*b_r.s4;
        c_r[5] += a_r.s0*b_r.s5;
        c_r[6] += a_r.s0*b_r.s6;
        c_r[7] += a_r.s0*b_r.s7;
        c_r[8] += a_r.s1*b_r.s0;
        c_r[9] += a_r.s1*b_r.s1;
        c_r[10] += a_r.s1*b_r.s2;
        c_r[11] += a_r.s1*b_r.s3;
        c_r[12] += a_r.s1*b_r.s4;
        c_r[13] += a_r.s1*b_r.s5;
        c_r[14] += a_r.s1*b_r.s6;
        c_r[15] += a_r.s1*b_r.s7;
        c_r[16] += a_r.s2*b_r.s0;
        c_r[17] += a_r.s2*b_r.s1;
        c_r[18] += a_r.s2*b_r.s2;
        c_r[19] += a_r.s2*b_r.s3;
        c_r[20] += a_r.s2*b_r.s4;
        c_r[21] += a_r.s2*b_r.s5;
        c_r[22] += a_r.s2*b_r.s6;
        c_r[23] += a_r.s2*b_r.s7;
        c_r[24] += a_r.s3*b_r.s0;
        c_r[25] += a_r.s3*b_r.s1;
        c_r[26] += a_r.s3*b_r.s2;
        c_r[27] += a_r.s3*b_r.s3;
        c_r[28] += a_r.s3*b_r.s4;
        c_r[29] += a_r.s3*b_r.s5;
        c_r[30] += a_r.s3*b_r.s6;
        c_r[31] += a_r.s3*b_r.s7;
        c_r[32] += a_r.s4*b_r.s0;
        c_r[33] += a_r.s4*b_r.s1;
        c_r[34] += a_r.s4*b_r.s2;
        c_r[35] += a_r.s4*b_r.s3;
        c_r[36] += a_r.s4*b_r.s4;
        c_r[37] += a_r.s4*b_r.s5;
        c_r[38] += a_r.s4*b_r.s6;
        c_r[39] += a_r.s4*b_r.s7;
        c_r[40] += a_r.s5*b_r.s0;
        c_r[41] += a_r.s5*b_r.s1;
        c_r[42] += a_r.s5*b_r.s2;
        c_r[43] += a_r.s5*b_r.s3;
        c_r[44] += a_r.s5*b_r.s4;
        c_r[45] += a_r.s5*b_r.s5;
        c_r[46] += a_r.s5*b_r.s6;
        c_r[47] += a_r.s5*b_r.s7;
        c_r[48] += a_r.s6*b_r.s0;
        c_r[49] += a_r.s6*b_r.s1;
        c_r[50] += a_r.s6*b_r.s2;
        c_r[51] += a_r.s6*b_r.s3;
        c_r[52] += a_r.s6*b_r.s4;
        c_r[53] += a_r.s6*b_r.s5;
        c_r[54] += a_r.s6*b_r.s6;
        c_r[55] += a_r.s6*b_r.s7;
        c_r[56] += a_r.s7*b_r.s0;
        c_r[57] += a_r.s7*b_r.s1;
        c_r[58] += a_r.s7*b_r.s2;
        c_r[59] += a_r.s7*b_r.s3;
        c_r[60] += a_r.s7*b_r.s4;
        c_r[61] += a_r.s7*b_r.s5;
        c_r[62] += a_r.s7*b_r.s6;
        c_r[63] += a_r.s7*b_r.s7;

        a_off += K / T_WIDTH;
        b_off += K / T_WIDTH;
    }

    int c_y = (get_group_id(0)/group_width)*local_width + (get_local_id(0)/local_width);
    int c_x = (get_group_id(0)%group_width)*local_width + (get_local_id(0)%local_width);

    int c_off;
    a_y = M - (c_y + 1) * T_WIDTH;
    b_x = N - (c_x + 1) * T_WIDTH;

    /*printf("c get_group_id(0): %d get_local_id(0): %d, %d %d   %d %d %d %d\n", get_group_id(0), get_local_id(0),
           (get_group_id(0)/group_width)*local_width, get_local_id(0)/local_width,
           c_y, c_x, a_y, b_x);*/
    if(a_y >= 0 && b_x >= 0){
        c_off = (c_y * N + c_x) * T_WIDTH;
        for( int Mt = 0; Mt < T_WIDTH; ++Mt ) {
            switch(Mt) {
            case 0:
                b_r.s0 = c_r[0];
                b_r.s1 = c_r[1];
                b_r.s2 = c_r[2];
                b_r.s3 = c_r[3];
                b_r.s4 = c_r[4];
                b_r.s5 = c_r[5];
                b_r.s6 = c_r[6];
                b_r.s7 = c_r[7];
                break;
            case 1:
                b_r.s0 = c_r[8];
                b_r.s1 = c_r[9];
                b_r.s2 = c_r[10];
                b_r.s3 = c_r[11];
                b_r.s4 = c_r[12];
                b_r.s5 = c_r[13];
                b_r.s6 = c_r[14];
                b_r.s7 = c_r[15];
                break;
            case 2:
                b_r.s0 = c_r[16];
                b_r.s1 = c_r[17];
                b_r.s2 = c_r[18];
                b_r.s3 = c_r[19];
                b_r.s4 = c_r[20];
                b_r.s5 = c_r[21];
                b_r.s6 = c_r[22];
                b_r.s7 = c_r[23];
                break;
            case 3:
                b_r.s0 = c_r[24];
                b_r.s1 = c_r[25];
                b_r.s2 = c_r[26];
                b_r.s3 = c_r[27];
                b_r.s4 = c_r[28];
                b_r.s5 = c_r[29];
                b_r.s6 = c_r[30];
                b_r.s7 = c_r[31];
                break;
            case 4:
                b_r.s0 = c_r[32];
                b_r.s1 = c_r[33];
                b_r.s2 = c_r[34];
                b_r.s3 = c_r[35];
                b_r.s4 = c_r[36];
                b_r.s5 = c_r[37];
                b_r.s6 = c_r[38];
                b_r.s7 = c_r[39];
                break;
            case 5:
                b_r.s0 = c_r[40];
                b_r.s1 = c_r[41];
                b_r.s2 = c_r[42];
                b_r.s3 = c_r[43];
                b_r.s4 = c_r[44];
                b_r.s5 = c_r[45];
                b_r.s6 = c_r[46];
                b_r.s7 = c_r[47];
                break;
            case 6:
                b_r.s0 = c_r[48];
                b_r.s1 = c_r[49];
                b_r.s2 = c_r[50];
                b_r.s3 = c_r[51];
                b_r.s4 = c_r[52];
                b_r.s5 = c_r[53];
                b_r.s6 = c_r[54];
                b_r.s7 = c_r[55];
                break;
            case 7:
                b_r.s0 = c_r[56];
                b_r.s1 = c_r[57];
                b_r.s2 = c_r[58];
                b_r.s3 = c_r[59];
                b_r.s4 = c_r[60];
                b_r.s5 = c_r[61];
                b_r.s6 = c_r[62];
                b_r.s7 = c_r[63];
                break;
            }
            //printf("%d %d: %d\n", a_y, b_x, c_off);
            *((global float8 *)(c + c_off)) = b_r;
            c_off += N;
        }
    } else if(a_y >= 0 && b_x < 0){  // right upper
        for(i = 0; i < T_WIDTH; i++){
            for(k = -b_x; k < T_WIDTH; k++){
                /*printf("%d %d %d %d: %d %d %d %f\n", c_y, c_x, a_y, b_x, i, k,
                       (c_y * N + c_x) * T_WIDTH + i * N + (k + b_x),
                       c_r[i * T_WIDTH + (k + b_x)]);*/
                c[(c_y * N + c_x) * T_WIDTH + i * N + (k + b_x)] = c_r[i * T_WIDTH + (k + b_x)];
            }
        }
    } else if(a_y < 0 && b_x >= 0){  // left down
        for(i = -a_y; i < T_WIDTH; i++){
            b_r.s0 = c_r[(i + a_y) * T_WIDTH];
            b_r.s1 = c_r[(i + a_y) * T_WIDTH + 1];
            b_r.s2 = c_r[(i + a_y) * T_WIDTH + 2];
            b_r.s3 = c_r[(i + a_y) * T_WIDTH + 3];
            b_r.s4 = c_r[(i + a_y) * T_WIDTH + 4];
            b_r.s5 = c_r[(i + a_y) * T_WIDTH + 5];
            b_r.s6 = c_r[(i + a_y) * T_WIDTH + 6];
            b_r.s7 = c_r[(i + a_y) * T_WIDTH + 7];
            /*printf("%d %d %d %d: %d %d, %d %d %f\n", c_y, c_x, a_y, b_x, i, k,
                   (c_y * N + c_x) * T_WIDTH + (i + a_y) * N, output_width,
                   c_r[(i + a_y) * T_WIDTH]);*/
            *((global float8 *)(c + (c_y * N + c_x) * T_WIDTH + (i + a_y) * N)) = b_r;
        }
    } else {
        for(i = -a_y; i < T_WIDTH; i++){
            for(k = -b_x; k < T_WIDTH; k++){
                /*printf("%d %d %d %d: %d %d %d %f\n", c_y, c_x, a_y, b_x, i, k,
                       (c_y * N + c_x) * T_WIDTH + (i + a_y) * N + (k + b_x),
                       c_r[i * T_WIDTH + (k + b_x)]);*/
                c[(c_y * N + c_x) * T_WIDTH + (i + a_y) * N + (k + b_x)] = c_r[(i + a_y) * T_WIDTH + (k + b_x)];
            }
        }
    }
}

__kernel void matrix_transpose_cl(__global float *matrix, __global float *matrix_t, int width, int height)
{
    const int wid_x  = get_global_id(0);
    const int wid_y  = get_global_id(1);
    __global const float *offset = matrix + width * 4 * wid_y;
    const float4 rows[] = {
        vload4(wid_x, offset),
        vload4(wid_x, offset + width),
        vload4(wid_x, offset + 2 * width),
        vload4(wid_x, offset + 3 * width),
    };
    __global float *write_offset = matrix_t + height * 4 * wid_x;
    vstore4((float4)(rows[0].x, rows[1].x, rows[2].x, rows[3].x), wid_y, write_offset);
    vstore4((float4)(rows[0].y, rows[1].y, rows[2].y, rows[3].y), wid_y, write_offset + height);
    vstore4((float4)(rows[0].z, rows[1].z, rows[2].z, rows[3].z), wid_y, write_offset + 2 * height);
    vstore4((float4)(rows[0].w, rows[1].w, rows[2].w, rows[3].w), wid_y, write_offset + 3 * height);
}

__kernel void matrix_transpose_direct_cl(__global float *matrix, __global float *matrix_t, int width, int height)
{
    const int wid_x  = get_global_id(0);
    const int wid_y  = get_global_id(1);
    matrix_t[height * wid_x + wid_y] = matrix[width * wid_y + wid_x];
}

__kernel void gemm_fast_image(int M, int N, int K, int local_width,  __global float *a, __read_only image2d_t b, __global float *c)
{
    float c_r[8*8] = {0};
    float8 a_r;
    float8 b_r;
    int group_width = M / 8 / local_width;

    int const a_off_thr = ((get_group_id(0)/group_width)*local_width + (get_local_id(0)/local_width));
    //int const b_off_thr = ((get_group_id(0)%group_width)*local_width + (get_local_id(0)%local_width));
    int b_off = ((get_group_id(0)%group_width)*local_width + (get_local_id(0)%local_width));

    int a_off = a_off_thr;
    for( int k = 0; k < K; k += 1 ) {
        //b_r = ((global float8 const *)b)[b_off];
        *((float4 *)&b_r) = read_imagef(b, (int2)(b_off * 2, k)); //(b_off * 8 / 4, k)
        *((float4 *)&b_r +1) = read_imagef(b, (int2)(b_off * 2 +1, k));
        a_r = ((global float8 const *)a)[a_off];
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s0*b_r.s4;
        c_r[5] += a_r.s0*b_r.s5;
        c_r[6] += a_r.s0*b_r.s6;
        c_r[7] += a_r.s0*b_r.s7;
        c_r[8] += a_r.s1*b_r.s0;
        c_r[9] += a_r.s1*b_r.s1;
        c_r[10] += a_r.s1*b_r.s2;
        c_r[11] += a_r.s1*b_r.s3;
        c_r[12] += a_r.s1*b_r.s4;
        c_r[13] += a_r.s1*b_r.s5;
        c_r[14] += a_r.s1*b_r.s6;
        c_r[15] += a_r.s1*b_r.s7;
        c_r[16] += a_r.s2*b_r.s0;
        c_r[17] += a_r.s2*b_r.s1;
        c_r[18] += a_r.s2*b_r.s2;
        c_r[19] += a_r.s2*b_r.s3;
        c_r[20] += a_r.s2*b_r.s4;
        c_r[21] += a_r.s2*b_r.s5;
        c_r[22] += a_r.s2*b_r.s6;
        c_r[23] += a_r.s2*b_r.s7;
        c_r[24] += a_r.s3*b_r.s0;
        c_r[25] += a_r.s3*b_r.s1;
        c_r[26] += a_r.s3*b_r.s2;
        c_r[27] += a_r.s3*b_r.s3;
        c_r[28] += a_r.s3*b_r.s4;
        c_r[29] += a_r.s3*b_r.s5;
        c_r[30] += a_r.s3*b_r.s6;
        c_r[31] += a_r.s3*b_r.s7;
        c_r[32] += a_r.s4*b_r.s0;
        c_r[33] += a_r.s4*b_r.s1;
        c_r[34] += a_r.s4*b_r.s2;
        c_r[35] += a_r.s4*b_r.s3;
        c_r[36] += a_r.s4*b_r.s4;
        c_r[37] += a_r.s4*b_r.s5;
        c_r[38] += a_r.s4*b_r.s6;
        c_r[39] += a_r.s4*b_r.s7;
        c_r[40] += a_r.s5*b_r.s0;
        c_r[41] += a_r.s5*b_r.s1;
        c_r[42] += a_r.s5*b_r.s2;
        c_r[43] += a_r.s5*b_r.s3;
        c_r[44] += a_r.s5*b_r.s4;
        c_r[45] += a_r.s5*b_r.s5;
        c_r[46] += a_r.s5*b_r.s6;
        c_r[47] += a_r.s5*b_r.s7;
        c_r[48] += a_r.s6*b_r.s0;
        c_r[49] += a_r.s6*b_r.s1;
        c_r[50] += a_r.s6*b_r.s2;
        c_r[51] += a_r.s6*b_r.s3;
        c_r[52] += a_r.s6*b_r.s4;
        c_r[53] += a_r.s6*b_r.s5;
        c_r[54] += a_r.s6*b_r.s6;
        c_r[55] += a_r.s6*b_r.s7;
        c_r[56] += a_r.s7*b_r.s0;
        c_r[57] += a_r.s7*b_r.s1;
        c_r[58] += a_r.s7*b_r.s2;
        c_r[59] += a_r.s7*b_r.s3;
        c_r[60] += a_r.s7*b_r.s4;
        c_r[61] += a_r.s7*b_r.s5;
        c_r[62] += a_r.s7*b_r.s6;
        c_r[63] += a_r.s7*b_r.s7;

        a_off += 1*K/8;
        //b_off += 1*K/8;
    }

    int c_off = ( (get_group_id(0)/group_width)*local_width + (get_local_id(0)/local_width) )*K +
        ( (get_group_id(0)%group_width)*local_width + (get_local_id(0)%local_width) );

    for( int Mt = 0; Mt < 8; ++Mt ) {
        switch(Mt) {
        case 0:
            b_r.s0 = c_r[0];
            b_r.s1 = c_r[1];
            b_r.s2 = c_r[2];
            b_r.s3 = c_r[3];
            b_r.s4 = c_r[4];
            b_r.s5 = c_r[5];
            b_r.s6 = c_r[6];
            b_r.s7 = c_r[7];
            break;
        case 1:
            b_r.s0 = c_r[8];
            b_r.s1 = c_r[9];
            b_r.s2 = c_r[10];
            b_r.s3 = c_r[11];
            b_r.s4 = c_r[12];
            b_r.s5 = c_r[13];
            b_r.s6 = c_r[14];
            b_r.s7 = c_r[15];
            break;
        case 2:
            b_r.s0 = c_r[16];
            b_r.s1 = c_r[17];
            b_r.s2 = c_r[18];
            b_r.s3 = c_r[19];
            b_r.s4 = c_r[20];
            b_r.s5 = c_r[21];
            b_r.s6 = c_r[22];
            b_r.s7 = c_r[23];
            break;
        case 3:
            b_r.s0 = c_r[24];
            b_r.s1 = c_r[25];
            b_r.s2 = c_r[26];
            b_r.s3 = c_r[27];
            b_r.s4 = c_r[28];
            b_r.s5 = c_r[29];
            b_r.s6 = c_r[30];
            b_r.s7 = c_r[31];
            break;
        case 4:
            b_r.s0 = c_r[32];
            b_r.s1 = c_r[33];
            b_r.s2 = c_r[34];
            b_r.s3 = c_r[35];
            b_r.s4 = c_r[36];
            b_r.s5 = c_r[37];
            b_r.s6 = c_r[38];
            b_r.s7 = c_r[39];
            break;
        case 5:
            b_r.s0 = c_r[40];
            b_r.s1 = c_r[41];
            b_r.s2 = c_r[42];
            b_r.s3 = c_r[43];
            b_r.s4 = c_r[44];
            b_r.s5 = c_r[45];
            b_r.s6 = c_r[46];
            b_r.s7 = c_r[47];
            break;
        case 6:
            b_r.s0 = c_r[48];
            b_r.s1 = c_r[49];
            b_r.s2 = c_r[50];
            b_r.s3 = c_r[51];
            b_r.s4 = c_r[52];
            b_r.s5 = c_r[53];
            b_r.s6 = c_r[54];
            b_r.s7 = c_r[55];
            break;
        case 7:
            b_r.s0 = c_r[56];
            b_r.s1 = c_r[57];
            b_r.s2 = c_r[58];
            b_r.s3 = c_r[59];
            b_r.s4 = c_r[60];
            b_r.s5 = c_r[61];
            b_r.s6 = c_r[62];
            b_r.s7 = c_r[63];
            break;
        }

        ((global float8 *)c)[c_off+0] = b_r;
        c_off += K/8;
    }
}

__kernel void gemm_fast(int M, int N, int N_tile, int K,  __global float *a, __global float *b, __global float *c)
{
    float c_r[T_WIDTH*T_WIDTH] = {0};
    float8 a_r;
    float8 b_r;

    int a_off = get_global_id(1);
    int b_off = get_global_id(0);
    int i, k;
    //if((get_group_id(0) > 4)) printf("%d %d %d %d %d %d\n", M, N, N_tile, K, a_off, b_off);
    for(k = 0; k < K; k += 1 ) {
        a_r = ((global float8 const *)a)[a_off];
        b_r = ((global float8 const *)b)[b_off];
        //break;
#ifdef FP_FAST_FMAF
        //printf("have FP_FAST_FMAF\n");
        c_r[0] = fma(a_r.s0, b_r.s0, c_r[0]);
        c_r[1] = fma(a_r.s0, b_r.s1, c_r[1]);
        c_r[2] = fma(a_r.s0, b_r.s2, c_r[2]);
        c_r[3] = fma(a_r.s0, b_r.s3, c_r[3]);
        c_r[4] = fma(a_r.s0, b_r.s4, c_r[4]);
        c_r[5] = fma(a_r.s0, b_r.s5, c_r[5]);
        c_r[6] = fma(a_r.s0, b_r.s6, c_r[6]);
        c_r[7] = fma(a_r.s0, b_r.s7, c_r[7]);
        c_r[8] = fma(a_r.s1, b_r.s0, c_r[8]);
        c_r[9] = fma(a_r.s1, b_r.s1, c_r[9]);
        c_r[10] = fma(a_r.s1, b_r.s2, c_r[10]);
        c_r[11] = fma(a_r.s1, b_r.s3, c_r[11]);
        c_r[12] = fma(a_r.s1, b_r.s4, c_r[12]);
        c_r[13] = fma(a_r.s1, b_r.s5, c_r[13]);
        c_r[14] = fma(a_r.s1, b_r.s6, c_r[14]);
        c_r[15] = fma(a_r.s1, b_r.s7, c_r[15]);
        c_r[16] = fma(a_r.s2, b_r.s0, c_r[16]);
        c_r[17] = fma(a_r.s2, b_r.s1, c_r[17]);
        c_r[18] = fma(a_r.s2, b_r.s2, c_r[18]);
        c_r[19] = fma(a_r.s2, b_r.s3, c_r[19]);
        c_r[20] = fma(a_r.s2, b_r.s4, c_r[20]);
        c_r[21] = fma(a_r.s2, b_r.s5, c_r[21]);
        c_r[22] = fma(a_r.s2, b_r.s6, c_r[22]);
        c_r[23] = fma(a_r.s2, b_r.s7, c_r[23]);
        c_r[24] = fma(a_r.s3, b_r.s0, c_r[24]);
        c_r[25] = fma(a_r.s3, b_r.s1, c_r[25]);
        c_r[26] = fma(a_r.s3, b_r.s2, c_r[26]);
        c_r[27] = fma(a_r.s3, b_r.s3, c_r[27]);
        c_r[28] = fma(a_r.s3, b_r.s4, c_r[28]);
        c_r[29] = fma(a_r.s3, b_r.s5, c_r[29]);
        c_r[30] = fma(a_r.s3, b_r.s6, c_r[30]);
        c_r[31] = fma(a_r.s3, b_r.s7, c_r[31]);
        c_r[32] = fma(a_r.s4, b_r.s0, c_r[32]);
        c_r[33] = fma(a_r.s4, b_r.s1, c_r[33]);
        c_r[34] = fma(a_r.s4, b_r.s2, c_r[34]);
        c_r[35] = fma(a_r.s4, b_r.s3, c_r[35]);
        c_r[36] = fma(a_r.s4, b_r.s4, c_r[36]);
        c_r[37] = fma(a_r.s4, b_r.s5, c_r[37]);
        c_r[38] = fma(a_r.s4, b_r.s6, c_r[38]);
        c_r[39] = fma(a_r.s4, b_r.s7, c_r[39]);
        c_r[40] = fma(a_r.s5, b_r.s0, c_r[40]);
        c_r[41] = fma(a_r.s5, b_r.s1, c_r[41]);
        c_r[42] = fma(a_r.s5, b_r.s2, c_r[42]);
        c_r[43] = fma(a_r.s5, b_r.s3, c_r[43]);
        c_r[44] = fma(a_r.s5, b_r.s4, c_r[44]);
        c_r[45] = fma(a_r.s5, b_r.s5, c_r[45]);
        c_r[46] = fma(a_r.s5, b_r.s6, c_r[46]);
        c_r[47] = fma(a_r.s5, b_r.s7, c_r[47]);
        c_r[48] = fma(a_r.s6, b_r.s0, c_r[48]);
        c_r[49] = fma(a_r.s6, b_r.s1, c_r[49]);
        c_r[50] = fma(a_r.s6, b_r.s2, c_r[50]);
        c_r[51] = fma(a_r.s6, b_r.s3, c_r[51]);
        c_r[52] = fma(a_r.s6, b_r.s4, c_r[52]);
        c_r[53] = fma(a_r.s6, b_r.s5, c_r[53]);
        c_r[54] = fma(a_r.s6, b_r.s6, c_r[54]);
        c_r[55] = fma(a_r.s6, b_r.s7, c_r[55]);
        c_r[56] = fma(a_r.s7, b_r.s0, c_r[56]);
        c_r[57] = fma(a_r.s7, b_r.s1, c_r[57]);
        c_r[58] = fma(a_r.s7, b_r.s2, c_r[58]);
        c_r[59] = fma(a_r.s7, b_r.s3, c_r[59]);
        c_r[60] = fma(a_r.s7, b_r.s4, c_r[60]);
        c_r[61] = fma(a_r.s7, b_r.s5, c_r[61]);
        c_r[62] = fma(a_r.s7, b_r.s6, c_r[62]);
        c_r[63] = fma(a_r.s7, b_r.s7, c_r[63]);
#else
        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s0*b_r.s4;
        c_r[5] += a_r.s0*b_r.s5;
        c_r[6] += a_r.s0*b_r.s6;
        c_r[7] += a_r.s0*b_r.s7;
        c_r[8] += a_r.s1*b_r.s0;
        c_r[9] += a_r.s1*b_r.s1;
        c_r[10] += a_r.s1*b_r.s2;
        c_r[11] += a_r.s1*b_r.s3;
        c_r[12] += a_r.s1*b_r.s4;
        c_r[13] += a_r.s1*b_r.s5;
        c_r[14] += a_r.s1*b_r.s6;
        c_r[15] += a_r.s1*b_r.s7;
        c_r[16] += a_r.s2*b_r.s0;
        c_r[17] += a_r.s2*b_r.s1;
        c_r[18] += a_r.s2*b_r.s2;
        c_r[19] += a_r.s2*b_r.s3;
        c_r[20] += a_r.s2*b_r.s4;
        c_r[21] += a_r.s2*b_r.s5;
        c_r[22] += a_r.s2*b_r.s6;
        c_r[23] += a_r.s2*b_r.s7;
        c_r[24] += a_r.s3*b_r.s0;
        c_r[25] += a_r.s3*b_r.s1;
        c_r[26] += a_r.s3*b_r.s2;
        c_r[27] += a_r.s3*b_r.s3;
        c_r[28] += a_r.s3*b_r.s4;
        c_r[29] += a_r.s3*b_r.s5;
        c_r[30] += a_r.s3*b_r.s6;
        c_r[31] += a_r.s3*b_r.s7;
        c_r[32] += a_r.s4*b_r.s0;
        c_r[33] += a_r.s4*b_r.s1;
        c_r[34] += a_r.s4*b_r.s2;
        c_r[35] += a_r.s4*b_r.s3;
        c_r[36] += a_r.s4*b_r.s4;
        c_r[37] += a_r.s4*b_r.s5;
        c_r[38] += a_r.s4*b_r.s6;
        c_r[39] += a_r.s4*b_r.s7;
        c_r[40] += a_r.s5*b_r.s0;
        c_r[41] += a_r.s5*b_r.s1;
        c_r[42] += a_r.s5*b_r.s2;
        c_r[43] += a_r.s5*b_r.s3;
        c_r[44] += a_r.s5*b_r.s4;
        c_r[45] += a_r.s5*b_r.s5;
        c_r[46] += a_r.s5*b_r.s6;
        c_r[47] += a_r.s5*b_r.s7;
        c_r[48] += a_r.s6*b_r.s0;
        c_r[49] += a_r.s6*b_r.s1;
        c_r[50] += a_r.s6*b_r.s2;
        c_r[51] += a_r.s6*b_r.s3;
        c_r[52] += a_r.s6*b_r.s4;
        c_r[53] += a_r.s6*b_r.s5;
        c_r[54] += a_r.s6*b_r.s6;
        c_r[55] += a_r.s6*b_r.s7;
        c_r[56] += a_r.s7*b_r.s0;
        c_r[57] += a_r.s7*b_r.s1;
        c_r[58] += a_r.s7*b_r.s2;
        c_r[59] += a_r.s7*b_r.s3;
        c_r[60] += a_r.s7*b_r.s4;
        c_r[61] += a_r.s7*b_r.s5;
        c_r[62] += a_r.s7*b_r.s6;
        c_r[63] += a_r.s7*b_r.s7;
#endif

        a_off += M/T_WIDTH;
        b_off += N_tile/T_WIDTH;
    }

    a_off = get_global_id(1);
    b_off = get_global_id(0);

    int a_y = M - (a_off + 1) * T_WIDTH;
    int b_x = N - (b_off + 1) * T_WIDTH;
    int path = b_x >= 0 ? 0 : 1;

    switch(path) {
    case 0:
        a_off = (a_off * N + b_off) * T_WIDTH;
//#pragma unroll
        //for(k = 0; k < T_WIDTH; k++) {
        //switch(k) {
        //case 0:
        b_r.s0 = c_r[0];
        b_r.s1 = c_r[1];
        b_r.s2 = c_r[2];
        b_r.s3 = c_r[3];
        b_r.s4 = c_r[4];
        b_r.s5 = c_r[5];
        b_r.s6 = c_r[6];
        b_r.s7 = c_r[7];
        *((global float8 *)(c + a_off)) = b_r;
        a_off += N;
        //break;
        //case 1:
        b_r.s0 = c_r[8];
        b_r.s1 = c_r[9];
        b_r.s2 = c_r[10];
        b_r.s3 = c_r[11];
        b_r.s4 = c_r[12];
        b_r.s5 = c_r[13];
        b_r.s6 = c_r[14];
        b_r.s7 = c_r[15];
        *((global float8 *)(c + a_off)) = b_r;
        a_off += N;
        //break;
        //case 2:
        b_r.s0 = c_r[16];
        b_r.s1 = c_r[17];
        b_r.s2 = c_r[18];
        b_r.s3 = c_r[19];
        b_r.s4 = c_r[20];
        b_r.s5 = c_r[21];
        b_r.s6 = c_r[22];
        b_r.s7 = c_r[23];
        //break;
        *((global float8 *)(c + a_off)) = b_r;
        a_off += N;
        //case 3:
        b_r.s0 = c_r[24];
        b_r.s1 = c_r[25];
        b_r.s2 = c_r[26];
        b_r.s3 = c_r[27];
        b_r.s4 = c_r[28];
        b_r.s5 = c_r[29];
        b_r.s6 = c_r[30];
        b_r.s7 = c_r[31];
        *((global float8 *)(c + a_off)) = b_r;
        a_off += N;
        //break;
        //case 4:
        b_r.s0 = c_r[32];
        b_r.s1 = c_r[33];
        b_r.s2 = c_r[34];
        b_r.s3 = c_r[35];
        b_r.s4 = c_r[36];
        b_r.s5 = c_r[37];
        b_r.s6 = c_r[38];
        b_r.s7 = c_r[39];
        *((global float8 *)(c + a_off)) = b_r;
        a_off += N;
        //break;
        //case 5:
        b_r.s0 = c_r[40];
        b_r.s1 = c_r[41];
        b_r.s2 = c_r[42];
        b_r.s3 = c_r[43];
        b_r.s4 = c_r[44];
        b_r.s5 = c_r[45];
        b_r.s6 = c_r[46];
        b_r.s7 = c_r[47];
        *((global float8 *)(c + a_off)) = b_r;
        a_off += N;
        //break;
        //case 6:
        b_r.s0 = c_r[48];
        b_r.s1 = c_r[49];
        b_r.s2 = c_r[50];
        b_r.s3 = c_r[51];
        b_r.s4 = c_r[52];
        b_r.s5 = c_r[53];
        b_r.s6 = c_r[54];
        b_r.s7 = c_r[55];
        *((global float8 *)(c + a_off)) = b_r;
        a_off += N;
        //break;
        //case 7:
        b_r.s0 = c_r[56];
        b_r.s1 = c_r[57];
        b_r.s2 = c_r[58];
        b_r.s3 = c_r[59];
        b_r.s4 = c_r[60];
        b_r.s5 = c_r[61];
        b_r.s6 = c_r[62];
        b_r.s7 = c_r[63];
        *((global float8 *)(c + a_off)) = b_r;
        a_off += N;
        //break;
        //*((global float8 *)(c + a_off)) = b_r;
        //a_off += N;
        //}
        break;
    case 1:  // right upper
        for(i = 0; i < T_WIDTH; i++){
            for(k = -b_x; k < T_WIDTH; k++){
                /*printf("%d %d %d %d: %d %d %d %f\n", a_off, b_off, a_y, b_x, i, k,
                       (a_off * N + b_off) * T_WIDTH + i * N + (k + b_x),
                       c_r[i * T_WIDTH + (k + b_x)]);*/
                c[(a_off * N + b_off) * T_WIDTH + i * N + (k + b_x)] = c_r[i * T_WIDTH + (k + b_x)];
            }
        }
    }
}

__kernel void gemm_fast_direct(int M, int M_tile, int N, int K,  __global float *a, __global float *b, __global float *c)
{
    float c_r[T_WIDTH*T_WIDTH] = {0};
    float8 a_r;
    float8 b_r;

    int a_off = get_global_id(1) * T_WIDTH;
    int b_off = get_global_id(0) * T_WIDTH;
    int a_y = M_tile - (a_off + T_WIDTH);
    int b_x = N - (b_off + T_WIDTH);
    int path = (a_y >= 0 && b_x >= 0) ? 0 : ((a_y >= 0 && b_x < 0) ? 1 : ((a_y < 0 && b_x >= 0) ? 2 : 3));

    int i, k;
    //if((get_group_id(0) > 4)) printf("%d %d %d %d %d %d\n", M, N, N_tile, K, a_off, b_off);
    for(k = 0; k < K; k += 1 ) {
        switch(path) {
        case 0:
            a_r = *((global float8 const *)(a + a_off));
            b_r = *((global float8 const *)(b + b_off));
            break;
        case 1:
            a_r = *((global float8 const *)(a + a_off));
            b_r = (float8)0;
            //if(k == 0) printf("%d %d\n", b_off, b_x);
            for(i = -b_x; i < T_WIDTH; i++){
                *((float *)&b_r + (i + b_x)) = b[b_off + (i + b_x)];
            }
             break;
        case 2:
            a_r = (float8)0;
            for(i = -a_y; i < T_WIDTH; i++){
                *((float *)&a_r + (i + a_y)) = a[a_off + (i + a_y)];
            }
            b_r = *((global float8 const *)(b +b_off));
            break;
        case 3:
            a_r = (float8)0;
            b_r = (float8)0;
            for(i = -a_y; i < T_WIDTH; i++){
                *((float *)&a_r + (i + a_y)) = a[a_off + (i + a_y)];
            }
            for(i = -b_x; i < T_WIDTH; i++){
                *((float *)&b_r + (i + b_x)) = b[b_off + (i + b_x)];
            }
            break;
        }

        c_r[0] += a_r.s0*b_r.s0;
        c_r[1] += a_r.s0*b_r.s1;
        c_r[2] += a_r.s0*b_r.s2;
        c_r[3] += a_r.s0*b_r.s3;
        c_r[4] += a_r.s0*b_r.s4;
        c_r[5] += a_r.s0*b_r.s5;
        c_r[6] += a_r.s0*b_r.s6;
        c_r[7] += a_r.s0*b_r.s7;
        c_r[8] += a_r.s1*b_r.s0;
        c_r[9] += a_r.s1*b_r.s1;
        c_r[10] += a_r.s1*b_r.s2;
        c_r[11] += a_r.s1*b_r.s3;
        c_r[12] += a_r.s1*b_r.s4;
        c_r[13] += a_r.s1*b_r.s5;
        c_r[14] += a_r.s1*b_r.s6;
        c_r[15] += a_r.s1*b_r.s7;
        c_r[16] += a_r.s2*b_r.s0;
        c_r[17] += a_r.s2*b_r.s1;
        c_r[18] += a_r.s2*b_r.s2;
        c_r[19] += a_r.s2*b_r.s3;
        c_r[20] += a_r.s2*b_r.s4;
        c_r[21] += a_r.s2*b_r.s5;
        c_r[22] += a_r.s2*b_r.s6;
        c_r[23] += a_r.s2*b_r.s7;
        c_r[24] += a_r.s3*b_r.s0;
        c_r[25] += a_r.s3*b_r.s1;
        c_r[26] += a_r.s3*b_r.s2;
        c_r[27] += a_r.s3*b_r.s3;
        c_r[28] += a_r.s3*b_r.s4;
        c_r[29] += a_r.s3*b_r.s5;
        c_r[30] += a_r.s3*b_r.s6;
        c_r[31] += a_r.s3*b_r.s7;
        c_r[32] += a_r.s4*b_r.s0;
        c_r[33] += a_r.s4*b_r.s1;
        c_r[34] += a_r.s4*b_r.s2;
        c_r[35] += a_r.s4*b_r.s3;
        c_r[36] += a_r.s4*b_r.s4;
        c_r[37] += a_r.s4*b_r.s5;
        c_r[38] += a_r.s4*b_r.s6;
        c_r[39] += a_r.s4*b_r.s7;
        c_r[40] += a_r.s5*b_r.s0;
        c_r[41] += a_r.s5*b_r.s1;
        c_r[42] += a_r.s5*b_r.s2;
        c_r[43] += a_r.s5*b_r.s3;
        c_r[44] += a_r.s5*b_r.s4;
        c_r[45] += a_r.s5*b_r.s5;
        c_r[46] += a_r.s5*b_r.s6;
        c_r[47] += a_r.s5*b_r.s7;
        c_r[48] += a_r.s6*b_r.s0;
        c_r[49] += a_r.s6*b_r.s1;
        c_r[50] += a_r.s6*b_r.s2;
        c_r[51] += a_r.s6*b_r.s3;
        c_r[52] += a_r.s6*b_r.s4;
        c_r[53] += a_r.s6*b_r.s5;
        c_r[54] += a_r.s6*b_r.s6;
        c_r[55] += a_r.s6*b_r.s7;
        c_r[56] += a_r.s7*b_r.s0;
        c_r[57] += a_r.s7*b_r.s1;
        c_r[58] += a_r.s7*b_r.s2;
        c_r[59] += a_r.s7*b_r.s3;
        c_r[60] += a_r.s7*b_r.s4;
        c_r[61] += a_r.s7*b_r.s5;
        c_r[62] += a_r.s7*b_r.s6;
        c_r[63] += a_r.s7*b_r.s7;

        a_off += M_tile;
        b_off += N;
    }

    a_off = get_global_id(1) * T_WIDTH;
    b_off = get_global_id(0) * T_WIDTH;

    a_y = M - (a_off + T_WIDTH);
    b_x = N - (b_off + T_WIDTH);
    path = (a_y >= 0 && b_x >= 0) ? 0 : ((a_y >= 0 && b_x < 0) ? 1 : ((a_y < 0 && b_x >= 0) ? 2 : 3));
    //if(path != 0) printf("path %d\n", path);

    switch(path) {
    case 0:
        a_off = (a_off * N + b_off);
        for(k = 0; k < T_WIDTH; k++) {
            switch(k) {
            case 0:
                b_r.s0 = c_r[0];
                b_r.s1 = c_r[1];
                b_r.s2 = c_r[2];
                b_r.s3 = c_r[3];
                b_r.s4 = c_r[4];
                b_r.s5 = c_r[5];
                b_r.s6 = c_r[6];
                b_r.s7 = c_r[7];
                break;
            case 1:
                b_r.s0 = c_r[8];
                b_r.s1 = c_r[9];
                b_r.s2 = c_r[10];
                b_r.s3 = c_r[11];
                b_r.s4 = c_r[12];
                b_r.s5 = c_r[13];
                b_r.s6 = c_r[14];
                b_r.s7 = c_r[15];
                break;
            case 2:
                b_r.s0 = c_r[16];
                b_r.s1 = c_r[17];
                b_r.s2 = c_r[18];
                b_r.s3 = c_r[19];
                b_r.s4 = c_r[20];
                b_r.s5 = c_r[21];
                b_r.s6 = c_r[22];
                b_r.s7 = c_r[23];
                break;
            case 3:
                b_r.s0 = c_r[24];
                b_r.s1 = c_r[25];
                b_r.s2 = c_r[26];
                b_r.s3 = c_r[27];
                b_r.s4 = c_r[28];
                b_r.s5 = c_r[29];
                b_r.s6 = c_r[30];
                b_r.s7 = c_r[31];
                break;
            case 4:
                b_r.s0 = c_r[32];
                b_r.s1 = c_r[33];
                b_r.s2 = c_r[34];
                b_r.s3 = c_r[35];
                b_r.s4 = c_r[36];
                b_r.s5 = c_r[37];
                b_r.s6 = c_r[38];
                b_r.s7 = c_r[39];
                break;
            case 5:
                b_r.s0 = c_r[40];
                b_r.s1 = c_r[41];
                b_r.s2 = c_r[42];
                b_r.s3 = c_r[43];
                b_r.s4 = c_r[44];
                b_r.s5 = c_r[45];
                b_r.s6 = c_r[46];
                b_r.s7 = c_r[47];
                break;
            case 6:
                b_r.s0 = c_r[48];
                b_r.s1 = c_r[49];
                b_r.s2 = c_r[50];
                b_r.s3 = c_r[51];
                b_r.s4 = c_r[52];
                b_r.s5 = c_r[53];
                b_r.s6 = c_r[54];
                b_r.s7 = c_r[55];
                break;
            case 7:
                b_r.s0 = c_r[56];
                b_r.s1 = c_r[57];
                b_r.s2 = c_r[58];
                b_r.s3 = c_r[59];
                b_r.s4 = c_r[60];
                b_r.s5 = c_r[61];
                b_r.s6 = c_r[62];
                b_r.s7 = c_r[63];
                break;
            }
            *((global float8 *)(c + a_off)) = b_r;
            a_off += N;
        }
        break;
    case 1:  // right upper
        for(i = 0; i < T_WIDTH; i++){
            for(k = -b_x; k < T_WIDTH; k++){
                /*printf("%d %d %d %d: %d %d %d %f\n", a_off, b_off, a_y, b_x, i, k,
                       (a_off * N + b_off) * T_WIDTH + i * N + (k + b_x),
                       c_r[i * T_WIDTH + (k + b_x)]);*/
                c[(a_off * N + b_off) + i * N + (k + b_x)] = c_r[i * T_WIDTH + (k + b_x)];
            }
        }
        break;
    case 2:  // left down
        for(i = -a_y; i < T_WIDTH; i++){
            b_r.s0 = c_r[(i + a_y) * T_WIDTH];
            b_r.s1 = c_r[(i + a_y) * T_WIDTH + 1];
            b_r.s2 = c_r[(i + a_y) * T_WIDTH + 2];
            b_r.s3 = c_r[(i + a_y) * T_WIDTH + 3];
            b_r.s4 = c_r[(i + a_y) * T_WIDTH + 4];
            b_r.s5 = c_r[(i + a_y) * T_WIDTH + 5];
            b_r.s6 = c_r[(i + a_y) * T_WIDTH + 6];
            b_r.s7 = c_r[(i + a_y) * T_WIDTH + 7];
            /*printf("%d %d %d %d: %d %d, %d %d %f\n", a_off, b_off, a_y, b_x, i, k,
                   (a_off * N + b_off) * T_WIDTH + (i + a_y) * N, output_width,
                   c_r[(i + a_y) * T_WIDTH]);*/
            *((global float8 *)(c + (a_off * N + b_off) + (i + a_y) * N)) = b_r;
        }
        break;
    case 3:
        for(i = -a_y; i < T_WIDTH; i++){
            for(k = -b_x; k < T_WIDTH; k++){
                /*printf("%d %d %d %d: %d %d %d %f\n", a_off, b_off, a_y, b_x, i, k,
                       (a_off * N + b_off) * T_WIDTH + (i + a_y) * N + (k + b_x),
                       c_r[i * T_WIDTH + (k + b_x)]);*/
                c[(a_off * N + b_off) + (i + a_y) * N + (k + b_x)] = c_r[(i + a_y) * T_WIDTH + (k + b_x)];
            }
        }
    }
}

__kernel void gemm_image(const int m, const int n, const int k,
                         __global const float *A, const int lda,
                         __read_only image2d_t Bi,
                         __global float *C, const int ldc)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1) << 3;
    //if (((gx << 2) >= n) || (gy >= m)) return;
    float4 a[8];
    float4 b[4];
    float4 c[8] = {0};
    int A_y_off = gy * lda;

    for (int pos = 0; pos < k; pos += 4) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            b[i] = read_imagef(Bi, (int2)(gx, pos + i));
        }

        int A_off = A_y_off + pos;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            a[i] = vload4(0, A + A_off);
            A_off += lda;
        }

#pragma unroll
        for (int i = 0; i < 8; i++) {
            c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int C_offs = (gy + i) * ldc + (gx << 2);
        vstore4(c[i], 0, C + C_offs);
    }
}

/* use buf for gemm_image */
__kernel void gemm_image_buf(const int m, const int n, const int k,
                             __global const float *A, const int lda,
                             __global const float *B, const int ldb,
                             __global float *C, const int ldc)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1) << 3;
    if (((gx << 2) < n) && (gy < m)) {
        float4 a[8];
        float4 b[4];
        float4 c[8];
        for (int i = 0; i < 8; i++) {
            c[i] = 0.0f;
        }
        int A_y_off = gy * lda;

        for (int pos = 0; pos < k; pos += 4) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                //b[i] = read_imagef(Bi, (int2)(gx, pos + i));
                b[i] = vload4(0, B + (pos + i) * ldb + (gx << 2));
            }

            int A_off = A_y_off + pos;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                a[i] = vload4(0, A + A_off);
                A_off += lda;
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];
            }

        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int C_offs = (gy + i) * ldc + (gx << 2);
            vstore4(c[i], 0, C + C_offs);
        }
    }
}

/* Computes the matrix product C = A * B
   There is no size restriction for the matrices, calculates the result using an efficient tiled algorithm.
   For the portion of the result matrix not covered by tiles it uses a less efficient naive implementation.
   Each work item computes a 4-column by 8-row (8x4) section of the output matrix.
   The inner loops read in a 1x4 section of matrix B, a 8x1 section of matrix A,
   and accumulate the partial results for the corresponding 8x4 section of matrix C.
   The outer loop iterates over the width of matrix A and the height of matrix B
   to get the complete result.
*/
__kernel void gemm_tile_8x4(__global const float *matrix_a, __global const float *matrix_b, __global float *matrix_c,
                            int matrix_b_width, int matrix_a_width)
{
    const int wid_x = get_global_id(0);
    const int wid_y = get_global_id(1);
    float  a[8];
    float4 b;
    float4 c[8];

    for (int i = 0; i < 8; ++i) {
        c[i] = (float4)(0.0f);
    }

    for (int j = 0; j < matrix_a_width; ++j) {
        b = vload4(0, matrix_b + j * matrix_b_width + (wid_x * 4));

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            a[i] = matrix_a[((wid_y * 8) + i) * matrix_a_width + j];
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[i] += a[i] * b;
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        vstore4(c[i], 0, matrix_c + ((wid_y * 8) + i) * matrix_b_width + (wid_x * 4));
    }
}

// The remainder version calculates a single element of the output matrix per work item.
__kernel void matmul_remainder(__global const  float *matrix_a,
                               __global const  float *matrix_b,
                               __global        float *matrix_c,
                                               int    x_rem_start,
                                               int    y_rem_start,
                                               int    matrix_b_width,
                                               int    matrix_a_width)
{
    const int wid_x = get_global_id(0) + x_rem_start;
    const int wid_y = get_global_id(1) + y_rem_start;

    float c     = 0.0f;
    int   a_idx = matrix_a_width * wid_y;
    int   b_idx = wid_x;

#pragma unroll 8
    for (int i = 0; i < matrix_a_width; ++i) {
        c += matrix_a[a_idx] * matrix_b[b_idx];
        ++a_idx;
        b_idx += matrix_b_width;
    }

    const int c_idx = wid_x + matrix_b_width * wid_y;
    matrix_c[c_idx] = c;
}

__kernel void gemm_native(const int matrixRowsA, const int matrixColsARowsB, const int matrixColsB,
                          __global float* matrixA, __global float* matrixB, __global float* matrixProduct)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if( i < matrixRowsA && j < matrixColsB ) {
        float result = 0.0;
        for( int k = 0; k < matrixColsARowsB; k++ ) {
            int indexA = i * matrixColsARowsB + k;
            int indexB = k * matrixColsB + j;
            result += matrixA[indexA] * matrixB[indexB];
        }
        matrixProduct[i * matrixColsB + j] = result;
    }
}

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

__kernel void array_add_cl(__global float *A, __global float *B, __global float *C, int N)
{
    int i = get_global_id(0);
    *((global float8 *)(C + i * T_WIDTH)) = *((global float8 *)(A + i * T_WIDTH)) + *((global float8 *)(B + i * T_WIDTH));
}

__kernel void copy_cl(int N, __global float *X, int INCX, __global float *Y, int INCY)
{
    int i = get_global_id(0);
    Y[i*INCY] = X[i*INCX];
}

__kernel void im2col_cl_3x3(__global float *data_im, int offset, int height, int width, int ksize, int pad, int stride,
                        int height_col, int width_col, __global float *data_col, int width_tile)
{
    int w_out = get_global_id(0) % width_col;
    int h_index = get_global_id(0) / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    int data_col_index = channel_out * width_tile + h_out * width_col + w_out;
    int data_im_index = offset + (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
        int h = h_in + i;
        float3 tmp = vload3(0, data_im + data_im_index + i * width);
        int w = w_in;
        data_col[data_col_index] = (h >= 0 && w >= 0 && h < height && w < width) ? tmp.s0 : 0;
        data_col_index += width_tile;

        w = w_in + 1;
        data_col[data_col_index] = (h >= 0 && w >= 0 && h < height && w < width) ? tmp.s1 : 0;
        data_col_index += width_tile;

        w = w_in + 2;
        data_col[data_col_index] = (h >= 0 && w >= 0 && h < height && w < width) ? tmp.s2 : 0;
        data_col_index += width_tile;
    }
}

__kernel void im2col_cl(__global float *data_im, int offset, int height, int width, int ksize, int pad, int stride,
                        int height_col, int width_col, __global float *data_col, int width_tile)
{
    int w_out = get_global_id(0) % width_col;
    int h_index = get_global_id(0) / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    int data_col_index = channel_out * width_tile + h_out * width_col + w_out;
    int data_im_index = offset + (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            int h = h_in + i;
            int w = w_in + j;
            data_col[data_col_index] = (h >= 0 && w >= 0 && h < height && w < width) ? data_im[data_im_index + i * width + j] : 0;
            data_col_index += width_tile;
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
    x[index] = (x[index] - mean[f])/sqrt(variance[f] + .00002f);
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
                                       __global float *input, __global float *output, __global int *indexes, int test)
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
    if(0 == test){    // 0: train, 1: valid
        indexes[out_index] = max_i;
    }
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

__kernel void l2normalize_cl(__global float *x, int batch, int filters, int spatial, __global float *norm_data)
{
    int index_ = get_global_id(0);
    int b = index_ / spatial;
    int i = index_ % spatial;
    int f;
    float sum = 1e-6;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        sum += (x[index] * x[index]);
    }
    sum = sqrt(sum);
    //if(sum == 0) sum = 1;
    //norm_data[b * spatial + i] = sum;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        x[index] /= sum;
    }
}
