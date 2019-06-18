#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "utils.h"
#include "image.h"
#include "gemm.h"

#ifdef GPU
#include "cuda.h"
#elif defined(OPENCL)
#include "opencl.h"
#elif defined(INTEL_MKL)
#include "mkl.h"
#elif defined(OPENBLAS_ARM)
#include "cblas.h"
#elif defined(QML)
#include <qml_cblas3.h>
#endif

float *make_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = rand_uniform(0, 0.001);
        //m[i] = i * 0.001;
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

#if defined(INTEL_MKL)
    start = what_time_is_it_now();
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,h,w,w,1,a,w,b,w,0,c,w);
    end = what_time_is_it_now();
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("Matrix Multiplication df mkl cpu %dx%d * %dx%d, sum: %f, %lf s\n", h, w, h, w, sum, end-start);
#endif

#if defined(OPENBLAS_ARM)
    start = what_time_is_it_now();
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,h,w,w,1,a,w,b,w,0,c,w);
    end = what_time_is_it_now();
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    int num_threads = openblas_get_num_threads();
    printf("Matrix Multiplication OPENBLAS_ARM cpu %d num, %dx%d * %dx%d, sum: %f, %lf s\n", num_threads, h, w, h, w, sum, end-start);
#endif

#if defined(QML)
    start = what_time_is_it_now();
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,h,w,w,1,a,w,b,w,0,c,w);
    end = what_time_is_it_now();
    sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    int num_threads = 1;//openblas_get_num_threads();
    printf("Matrix Multiplication QML cpu %d num, %dx%d * %dx%d, sum: %f, %lf s\n", num_threads, h, w, h, w, sum, end-start);
#endif

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
    float *a_gpu = cuda_make_array(a, w * h);
    float *b_gpu = cuda_make_array(b, w * h);
    float *c_gpu = cuda_make_array(0, w * h);
    double start, end;
    for(int i = 0; i < 10; i++){
        gemm_gpu(0,0,h,w,w,1,a_gpu,w,b_gpu,w,0,c_gpu,w);
    }
    cudaThreadSynchronize();
    int try_times = 15;
    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_gpu(0,0,h,w,w,1,a_gpu,w,b_gpu,w,0,c_gpu,w);
    }
    cudaThreadSynchronize();
    end = what_time_is_it_now();
    cuda_pull_array(c_gpu, c, w * h);
    float sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("Matrix Multiplication gpu %dx%d * %dx%d, sum: %f, %lf s\n", h, w, h, w, sum, (end-start) / try_times);
    cuda_free(a_gpu);
    cuda_free(b_gpu);
    cuda_free(c_gpu);
    free(a);
    free(b);
    free(c);
    return 0;
}
#endif

#ifdef OPENCL
void test_gemm_cl(int w, int h)
{
    cl_setup();
    float *a = make_matrix(h, w);
    float *b = make_matrix(h, w);
    float *c = make_matrix(h, w);
    cl_mem a_cl = cl_make_array(a, w * h);
    cl_mem b_cl = cl_make_array(b, w * h);
    cl_mem c_cl = cl_make_array(0, w * h);

    double start, end;
    for(int i = 0; i < 2; i++){
        gemm_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
    }
    int try_times = 3;
    start = what_time_is_it_now();
    for(int i = 0; i < try_times; i++){
        gemm_cl(0,0,h,w,w,1,a_cl,0,w,b_cl,0,w,0,c_cl,0,w);
    }
    end = what_time_is_it_now();
    cl_read_array(c_cl, c, w * h);
    float sum = 0;
    for(int i = 0; i < w * h; i++) sum += c[i];
    printf("Matrix Multiplication cl %dx%d * %dx%d, sum: %f, %lf s\n",
           h, w, h, w, sum, (end-start) / try_times);
    clReleaseMemObject(a_cl);
    clReleaseMemObject(b_cl);
    clReleaseMemObject(c_cl);
    free(a);
    free(b);
    free(c);
}
#endif
/*
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
        im.h = sqrtf(fields);
        im.w = sqrtf(fields);
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

void test_image()
{
    int w = 96, h = 112, c = 3;
    float hue = 0.001, saturation = 0.999, exposure = 0.8;
    image img = load_image("./test.jpg", w, h, c);
    for(int i = 0; i < 100; i++) {
        random_distort_image(img, hue, saturation, exposure);
        char save_name[128];
        sprintf(save_name, "distort_image_%04d", i);
        save_image_png(img, save_name);
        //sleep(1);
    }
}

void test_load_csv_image()
{
    FILE *fp = fopen("/var/darknet/insightface/src/face_data_train.csv", "r");
    if(!fp) file_error("file not exist!\n");
    int fields = 0;  // the number of pixel per image
    int w, h;
    int n = 0;
    char *line;
    int train_set_size = 1;
    while((line = fgetl(fp)) && (n < train_set_size)){
        char class = atoi(line);
        if(0 == fields){
            fields = count_fields(line);
            w = sqrtf(fields / 3);
            h = sqrtf(fields / 3);
        }
        printf("class: %d, w: %d, h: %d\n", class, w, h);
        float *value = parse_fields(line, fields);
        image tmp;
        tmp.w = w;
        tmp.h = h;
        tmp.c = 3;
        tmp.data = value + 1;
        save_image_png(tmp, "input.jpg");
        sleep(1);
        free(line);
    }
}
*/

int main(int argc, char **argv)
{
    // https://pjreddie.com/projects/mnist-in-csv/
    //load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_train.csv", "/home/luyao/git/cnn/.data/mnist/train");
    //load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_test.csv", "/home/luyao/git/cnn/.data/mnist/test");
    //test_convolutional_layer();
#ifdef GPU
    int w = 4096 * 2;
    int h = 4096 * 2;
    test_gemm_gpu(w, h);
#elif defined(OPENCL)
    int w = 4096 * 2;
    int h = 4096 * 2;
    test_gemm_cl(w, h);
#else
    time_gemm(2000, 2000);
    //test_image();
    //test_load_csv_image();
#endif
    return 0;
}
