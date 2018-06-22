#include <stdio.h>
#include <time.h>

#include "utils.h"
#include "image.h"
#include "gemm.h"

#ifdef GPU
#include "cuda.h"
#endif

float *make_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = rand_uniform(0, 1);
    }
    return m;
}


void time_gemm(int w, int h)
{
    float *a = make_matrix(h, w);
    float *b = make_matrix(h, w);
    float *c = make_matrix(h, w);
    double start = what_time_is_it_now(), end;
    gemm(0,0,h,w,w,1,a,w,b,w,0,c,w);
    end = what_time_is_it_now();
    float sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("Matrix Multiplication cpu %dx%d * %dx%d, sum: %f, %lf s\n", h, w, h, w, sum, end-start);
    free(a);
    free(b);
    free(c);
}

#ifdef GPU

int test_gemm_gpu(int w, int h)
{
    float *a = make_matrix(h, w);
    float *b = make_matrix(h, w);
    float *c = make_matrix(h, w);
    double start = what_time_is_it_now(), end;
    gemm(0,0,h,w,w,1,a,w,b,w,0,c,w);
    end = what_time_is_it_now();
    float sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    double cpu_time = end-start;
    printf("Matrix Multiplication cpu %dx%d * %dx%d, sum: %f, %lf s\n", h, w, h, w, sum, end-start);

    float *a_gpu = cuda_make_array(a, w * h);
    float *b_gpu = cuda_make_array(b, w * h);
    float *c_gpu = cuda_make_array(0, w * h);
    start = what_time_is_it_now();
    gemm_gpu(0,0,h,w,w,1,a_gpu,w,b_gpu,w,0,c_gpu,w);
    end = what_time_is_it_now();
    double gpu_time = end-start;
    cuda_pull_array(c_gpu, c, w * h);
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("Matrix Multiplication gpu %dx%d * %dx%d, sum: %f, %lf s speedup: %f\n",
    		h, w, h, w, sum, end-start, cpu_time/ gpu_time);
    cuda_free(a_gpu);
    cuda_free(b_gpu);
    cuda_free(c_gpu);
    free(a);
    free(b);
    free(c);
    return 0;
}

#endif

void load_csv_image(char *filename, char *save_dir)
{
    FILE *fp = fopen(filename, "r");
    if(!fp) file_error(filename);

    char *line;
    int n = 0;
    while((line = fgetl(fp))){
        char class = line[0];
        int fields = count_fields(line);
        float *value = parse_fields(line, fields);
        image im;
        im.h = sqrt(fields);
        im.w = sqrt(fields);
        im.c = 1;
        im.data = value + 1;
        char name[128] = {0};
        sprintf(name, "%s/%05d_%c", save_dir, n, class);
        printf("%s %d %d\n", name, im.h, im.w);
        //save_image_png(im, name);
        free(line);
        n++;
        //break;
    }
    fclose(fp);
}

int main(int argc, char **argv)
{
    // https://pjreddie.com/projects/mnist-in-csv/
    //load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_train.csv", "/home/luyao/git/cnn/.data/mnist/train");
    //load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_test.csv", "/home/luyao/git/cnn/.data/mnist/test");
    //test_convolutional_layer();
    //time_gemm(2000, 2000);
    #ifdef GPU
    test_gemm_gpu(1000, 1000);
    #endif
    return 0;
}
