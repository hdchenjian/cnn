#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "utils.h"
#include "image.h"

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
    load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_train.csv", "/home/luyao/git/cnn/.data/mnist/train");
    load_csv_image("/home/luyao/git/cnn/.data/mnist/mnist_test.csv", "/home/luyao/git/cnn/.data/mnist/test");
    return 0;
}
