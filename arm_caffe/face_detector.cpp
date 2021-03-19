#include <unistd.h>
#include <sys/time.h>

#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils/GraphUtils.h"
#include "arm_compute/graph.h"
#include "utils/Utils.h"

#include "yolo_layer.h"

bool have_init = false;
std::string model_path_prefix = "/sdcard/A/";
std::mutex gpu_lock;

int yolo_num_detections(const yolo_layer *l, float thresh);
void forward_yolo_layer(const yolo_layer *l, void *net, float *input, int test);
yolo_layer *make_yolo_snpe(int c, int h, int w, const char *mask_str, const char *anchors);
void do_nms_sort(detection *dets, int total, int classes, float thresh);
void free_ptr(void **ptr);
int get_yolo_detections(yolo_layer *l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);

#define YOLO_CHANNEL 18
#define COARSE_SIZE 13
#define FINE_SIZE 26
#define MAX_BBOX_NUM 5
#define FEATURE_LENGTH 512

arm_compute::graph::Target graph_target = arm_compute::graph::Target::NEON;
//arm_compute::graph::Target graph_target = arm_compute::graph::Target::CL;
arm_compute::graph::FastMathHint fast_math_hint = arm_compute::graph::FastMathHint::Enabled; //Disabled;
int num_threads = 0;
bool use_tuner = false;
//arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::EXHAUSTIVE;
arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::NORMAL;
//arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::RAPID;

arm_compute::graph::frontend::Stream graph_yolo_tiny(0, "yolov3-tiny");

float *input_image_yolo = NULL;
float yolo1[COARSE_SIZE * COARSE_SIZE * YOLO_CHANNEL] = {0};
float yolo2[FINE_SIZE * FINE_SIZE * YOLO_CHANNEL] = {0};
bool yolo_input_load = false;
bool yolo_output_load = false;
bool yolo_output1_load = false;

int num_detections_local(float thresh);
detection *make_network_boxes_local(float thresh, int *num, int classes);
void fill_network_boxes_local(int net_w, int net_h, int w, int h, float thresh, int *map, int relative, detection *dets);
detection *get_network_boxes_local(int net_w, int net_h, int w, int h, float thresh, int *map, int relative, int *num, int classes);
void free_yolo_layer(void *input);
yolo_layer *yolo_coarse_layer = 0;
yolo_layer *yolo_fine_layer = 0;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

inline std::unique_ptr<arm_compute::graph::ITensorAccessor> get_weights_accessor(const std::string &path, const std::string &data_file)
{
    arm_compute::DataLayout file_layout = arm_compute::DataLayout::NCHW;
    if(path.empty()) {
        std::cout << "get_weights_accessor path not found: " << path << std::endl;
        exit(-1);
    } else {
        return arm_compute::support::cpp14::make_unique<arm_compute::graph_utils::NumPyBinLoader>(path + data_file, file_layout);
    }
}

class LoadInputData final : public arm_compute::graph::ITensorAccessor
{
public:
    LoadInputData(float *data, bool *already_loaded) : _already_loaded(already_loaded), _data(data){}

    bool access_tensor(arm_compute::ITensor &tensor) override {
        if(!*_already_loaded){
            if(tensor.info()->data_type() != arm_compute::DataType::F32){
                printf("Unsupported format\n");
                exit(-1);
            }
            arm_compute::Window window;
            window.use_tensor_dimensions(tensor.info()->tensor_shape());
            arm_compute::utils::map(tensor, true);
            arm_compute::Iterator it(&tensor, window);

            int i = 0;
            //printf("LoadInputData %p\n", reinterpret_cast<float *>(it.ptr()));
            execute_window_loop(window, [&](const arm_compute::Coordinates & id) {
                    //if(i < 10) printf("LoadInputData %d %d %d %d %f\n", i, id.x(), id.y(), id.z(), _data[i]);
                    //*reinterpret_cast<float*>( tensor.buffer() + tensor.info()->offset_element_in_bytes(id)) = _data[i++];
                    *reinterpret_cast<float *>(it.ptr()) = _data[i++];
                },
                it);
            arm_compute::utils::unmap(tensor);
        }
        *_already_loaded = !*_already_loaded;
        return *_already_loaded;
    }

private:
    bool *_already_loaded;
    float *_data;
};

class ReadOutputData final : public arm_compute::graph::ITensorAccessor
{
public:
    ReadOutputData(float *output_data, bool *already_read) : _already_read(already_read), data(output_data){}

    bool access_tensor(arm_compute::ITensor &tensor) override {
        //std::cout << "ReadOutputData access_tensor" << std::endl;
        if(!*_already_read){
            if(tensor.info()->data_type() != arm_compute::DataType::F32){
                printf("Unsupported format\n");
                exit(-1);
            }
            /*
            const int idx_width = arm_compute::get_data_layout_dimension_index(
                arm_compute::DataLayout::NCHW, arm_compute::DataLayoutDimension::WIDTH);
            const int idx_height = arm_compute::get_data_layout_dimension_index(
                arm_compute::DataLayout::NCHW, arm_compute::DataLayoutDimension::HEIGHT);
            const int idx_channel = arm_compute::get_data_layout_dimension_index(
                arm_compute::DataLayout::NCHW, arm_compute::DataLayoutDimension::CHANNEL);
            printf("output tensor w %zd h %zd c %zd\n",
                   tensor.info()->dimension(idx_width), tensor.info()->dimension(idx_height), tensor.info()->dimension(idx_channel));
            */
            arm_compute::Window window;
            window.use_tensor_dimensions(tensor.info()->tensor_shape());
            arm_compute::utils::map(tensor, true);
            arm_compute::Iterator it(&tensor, window);

            int i = 0;
            execute_window_loop(window, [&](const arm_compute::Coordinates & id) {
                    data[i++] = *reinterpret_cast<float *>(it.ptr());
                    //if(i < 10) printf("%d %f\n", i - 1, data[i - 1]);
                },
                it);
            arm_compute::utils::unmap(tensor);
        }
        *_already_read = !*_already_read;
        return *_already_read;
    }

private:
    bool *_already_read;
    float *data;
};

void set_yolo_input(cv::Mat &image_input){
    int channels = image_input.channels();
    cv::Mat resized;
    int network_w = 416;
    int network_h = 416;
    int new_w = image_input.cols;
    int new_h = image_input.rows;
    if (((float)network_w / image_input.cols) < ((float)network_h / image_input.rows)) {
        new_w = network_w;
        new_h = (image_input.rows * network_w) / image_input.cols;
    } else {
        new_h = network_h;
        new_w = (image_input.cols * network_h) / image_input.rows;
    }
    cv::resize(image_input, resized, cv::Size(new_w, new_h));
    //cv::imwrite("1.jpg", resized);
    int network_input_size = network_w * network_h * channels;
    for(int i = 0; i < network_input_size; i++) input_image_yolo[i] = 0.5f;
    int dx = (network_w - resized.cols) / 2;
    int dy = (network_h - resized.rows) / 2;
    for(int k = 0; k < channels; ++k){
        int k_index = k* network_w * network_h;
        for(int i = 0; i < resized.rows; ++i){
            int i_index = (i + dy) * network_w;
            for(int j = 0; j < resized.cols; ++j){
                float val = resized.at<cv::Vec3b>(i, j)[channels - k - 1] / 255.0f;
                if(j + dx >= 0 && i + dy >= 0 && j + dx < network_w && i + dy < network_h){
                    input_image_yolo[k_index + i_index + (j + dx)] = val;
                }
            }
        }
    }
}

void get_yolo_result(cv::Mat image_input, int *total_bbox_num, int *detection_bbox){
    forward_yolo_layer(yolo_coarse_layer, 0, yolo1, 1);
    forward_yolo_layer(yolo_fine_layer, 0, yolo2, 1);

    int network_w = 416;
    int network_h = 416;
    float thresh = .6;
    float nms = .45;
    int *map = 0;
    int classes = 1;
    int nboxes = 0;
    int image_original_w = image_input.cols;
    int image_original_h = image_input.rows;
    detection *dets = get_network_boxes_local(network_w, network_h, image_original_w, image_original_h, thresh, map, 0, &nboxes, classes);
    if(nms) do_nms_sort(dets, nboxes, classes, nms);
    int bbox_num = 0;
    for(int i = 0; i < nboxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;
        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > image_original_w) xmax = image_original_w;
        if (ymax > image_original_h) ymax = image_original_h;
        //printf("%d / %d prob %f, %f %f %f %f\n", i, nboxes, dets[i].prob[0], xmin, ymin, xmax, ymax);
        for(int j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh && bbox_num < MAX_BBOX_NUM){
                detection_bbox[bbox_num * 4] = xmin;
                detection_bbox[bbox_num * 4 + 1] = ymin;
                detection_bbox[bbox_num * 4 + 2] = xmax;
                detection_bbox[bbox_num * 4 + 3] = ymax;
                bbox_num += 1;
            }
        }
    }
    *total_bbox_num = bbox_num;
    for(int i = 0; i < nboxes; ++i){
        free_ptr((void **)&(dets[i].prob));
    }
    free_ptr((void **)&dets);
    //printf("run_detect spend: %f ms, %d face\n", what_time_is_it_now() - start, bbox_num);
    //for(int i = 0; i < 2 * 4; i++) printf("%d %d\n", i, detection_bbox[i]);
}

void run_yolo_tiny(cv::Mat &image_input, int *total_bbox_num, int *detection_bbox){
    yolo_input_load = false;
    yolo_output_load = false;
    yolo_output1_load = false;
    double start = what_time_is_it_now();
    set_yolo_input(image_input);
    graph_yolo_tiny.run();
    //for(int i = 0; i < 3; i++) printf("yolo1 %d %f\n", i, yolo1[i]);
    //for(int i = 0; i < 3; i++) printf("yolo2 %d %f\n", i, yolo2[i]);
    //return;
    get_yolo_result(image_input, total_bbox_num, detection_bbox);
}

void add_yolo_tiny_conv(const std::string &data_path, unsigned int channel, int weight_index, int stride, arm_compute::graph::frontend::Stream &graph_net) {
    arm_compute::ActivationLayerInfo active_info = arm_compute::ActivationLayerInfo(
        arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f);
    std::string unit_path = std::to_string(weight_index);

    graph_net << arm_compute::graph::frontend::ConvolutionLayer(
                   3U, 3U, channel,
                   get_weights_accessor(data_path, "conv" + unit_path + "_w.npy"),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name("conv" + unit_path)

               << arm_compute::graph::frontend::BatchNormalizationLayer(
                   get_weights_accessor(data_path, "bn" + unit_path + "_w.npy"),
                   get_weights_accessor(data_path, "bn" + unit_path + "_b.npy"),
                   get_weights_accessor(data_path, "bn" + unit_path + "_scale_w.npy"),
                   get_weights_accessor(data_path, "bn" + unit_path + "_scale_b.npy"),
                   0.00002f).set_name("bn" + unit_path)
              << arm_compute::graph::frontend::ActivationLayer(active_info).set_name("relu" + unit_path);
    if(weight_index != 6){
        graph_net << arm_compute::graph::frontend::PoolingLayer(
                  arm_compute::PoolingLayerInfo(
                      arm_compute::PoolingType::MAX, 2, arm_compute::PadStrideInfo(stride, stride, 0, 0))).set_name("pool" + unit_path);
    } else {
        graph_net << arm_compute::graph::frontend::PoolingLayer(
                  arm_compute::PoolingLayerInfo(
                      arm_compute::PoolingType::MAX, 2,
                      arm_compute::PadStrideInfo(stride, stride, 0, 1, 0, 1, arm_compute::DimensionRoundingType::FLOOR))).set_name("pool" + unit_path);
    }
}

void add_convolutional_yolo_tiny_sub(const std::string &data_path, unsigned int channel, unsigned int kernel, int pad, int weight_index,
                                     arm_compute::graph::frontend::Stream &graph_net){
    arm_compute::ActivationLayerInfo active_info = arm_compute::ActivationLayerInfo(
        arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f);
    std::string unit_path = std::to_string(weight_index);
    graph_net << arm_compute::graph::frontend::ConvolutionLayer(kernel, kernel, channel,
                                                                get_weights_accessor(data_path, "conv" + unit_path + "_w.npy"),
                                                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                                                arm_compute::PadStrideInfo(1, 1, pad, pad)).set_name("conv" + unit_path)
              << arm_compute::graph::frontend::BatchNormalizationLayer(
                  get_weights_accessor(data_path, "bn" + unit_path + "_w.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_b.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_scale_w.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_scale_b.npy"), 0.00002f).set_name("bn" + unit_path)
              << arm_compute::graph::frontend::ActivationLayer(active_info).set_name("relu" + unit_path);
}

void add_convolutional_yolo_tiny_substream(const std::string &data_path, unsigned int channel, unsigned int kernel, int pad, int weight_index,
                                     arm_compute::graph::frontend::SubStream &graph_net){
    arm_compute::ActivationLayerInfo active_info = arm_compute::ActivationLayerInfo(
        arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f);
    std::string unit_path = std::to_string(weight_index);
    graph_net << arm_compute::graph::frontend::ConvolutionLayer(kernel, kernel, channel,
                                                                get_weights_accessor(data_path, "conv" + unit_path + "_w.npy"),
                                                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                                                arm_compute::PadStrideInfo(1, 1, pad, pad)).set_name("conv" + unit_path)
              << arm_compute::graph::frontend::BatchNormalizationLayer(
                  get_weights_accessor(data_path, "bn" + unit_path + "_w.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_b.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_scale_w.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_scale_b.npy"), 0.00002f).set_name("bn" + unit_path)
              << arm_compute::graph::frontend::ActivationLayer(active_info).set_name("relu" + unit_path);
}

void init_yolo_tiny(float *input_image_yolo, float *yolo1, float *yolo2){
    const char *mask_str_coarse = "3,4,5";
    const char *anchors = "10,14,  23,27,  37,58,  81,82,  135,169,  344,319";
    const char *mask_str_fine = "0,1,2";
    if(yolo_coarse_layer == 0) yolo_coarse_layer = make_yolo_snpe(YOLO_CHANNEL, COARSE_SIZE, COARSE_SIZE, mask_str_coarse, anchors);
    if(yolo_fine_layer == 0) yolo_fine_layer = make_yolo_snpe(YOLO_CHANNEL, FINE_SIZE, FINE_SIZE, mask_str_fine, anchors);

    std::string data_path = model_path_prefix + "yolov3_tiny/";
    const arm_compute::DataLayout weights_layout = arm_compute::DataLayout::NCHW;
    const arm_compute::TensorShape tensor_shape = arm_compute::TensorShape(416U, 416U, 3U, 1U); // DataLayout::NCHW, DataLayout::NCHW);
    arm_compute::graph::TensorDescriptor input_descriptor = arm_compute::graph::TensorDescriptor(
        tensor_shape, arm_compute::DataType::F32).set_layout(weights_layout);

    graph_yolo_tiny << graph_target
               << fast_math_hint
               << arm_compute::graph::frontend::InputLayer(
                   input_descriptor, arm_compute::support::cpp14::make_unique<LoadInputData>(input_image_yolo, &yolo_input_load));
    add_yolo_tiny_conv(data_path, 16, 1, 2,  graph_yolo_tiny);
    add_yolo_tiny_conv(data_path, 32, 2, 2, graph_yolo_tiny);
    add_yolo_tiny_conv(data_path, 64, 3, 2, graph_yolo_tiny);
    add_yolo_tiny_conv(data_path, 128, 4, 2, graph_yolo_tiny);

    graph_yolo_tiny << arm_compute::graph::frontend::ConvolutionLayer(
                   3U, 3U, 256,
                   get_weights_accessor(data_path, "conv5_w.npy"),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name("conv4")

               << arm_compute::graph::frontend::BatchNormalizationLayer(
                   get_weights_accessor(data_path, "bn5_w.npy"),
                   get_weights_accessor(data_path, "bn5_b.npy"),
                   get_weights_accessor(data_path, "bn5_scale_w.npy"),
                   get_weights_accessor(data_path, "bn5_scale_b.npy"),
                   0.00002f).set_name("bn5")
              << arm_compute::graph::frontend::ActivationLayer(
                  arm_compute::ActivationLayerInfo(
                      arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("relu5");

    arm_compute::graph::frontend::SubStream upsample_route(graph_yolo_tiny);
    
    graph_yolo_tiny << arm_compute::graph::frontend::PoolingLayer(
        arm_compute::PoolingLayerInfo(
            arm_compute::PoolingType::MAX, 2, arm_compute::PadStrideInfo(2, 2, 0, 0))).set_name("pool5");

    //add_yolo_tiny_conv(data_path, 256, 5, 2, graph_yolo_tiny);
    add_yolo_tiny_conv(data_path, 512, 6, 1, graph_yolo_tiny);

    add_convolutional_yolo_tiny_sub(data_path, 1024, 3, 1, 7, graph_yolo_tiny);
    add_convolutional_yolo_tiny_sub(data_path, 256, 1, 0, 8, graph_yolo_tiny);
    arm_compute::graph::frontend::SubStream yolo_route(graph_yolo_tiny);
    add_convolutional_yolo_tiny_sub(data_path, 512, 3, 1, 9, graph_yolo_tiny);
    graph_yolo_tiny << arm_compute::graph::frontend::ConvolutionLayer(1U, 1U, 18,
                                                                get_weights_accessor(data_path, std::string("conv10") + "_w.npy"),
                                                                get_weights_accessor(data_path, std::string("conv10") + "_b.npy"),
                                                                arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv10")
                    << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(yolo1, &yolo_output_load));

    add_convolutional_yolo_tiny_substream(data_path, 128, 1, 0, 11, yolo_route);
    yolo_route << arm_compute::graph::frontend::UpsampleLayer(arm_compute::Size2D(2, 2),
                                                              arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR).set_name("Upsample_19");
    arm_compute::graph::frontend::SubStream concat_1(yolo_route);
    concat_1 << arm_compute::graph::frontend::ConcatLayer(std::move(yolo_route), std::move(upsample_route)).set_name("Route1");
    add_convolutional_yolo_tiny_substream(data_path, 256, 3, 1, 12, concat_1);
    concat_1 << arm_compute::graph::frontend::ConvolutionLayer(1U, 1U, 18,
                                                               get_weights_accessor(data_path, std::string("conv13") + "_w.npy"),
                                                               get_weights_accessor(data_path, std::string("conv13") + "_b.npy"),
                                                               arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv13")
             << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(yolo2, &yolo_output1_load));

    arm_compute::graph::GraphConfig config;
    config.num_threads = num_threads;
    config.use_tuner = use_tuner;
    config.tuner_mode = cl_tuner_mode;
    config.tuner_file = model_path_prefix + "acl_tuner_yolov3_tiny.csv";

    graph_yolo_tiny.finalize(graph_target, config);
    return;
}

int recognition_face(cv::Mat &img, std::vector<float> &feature){
    std::lock_guard<std::mutex> gpu_lock_guard(gpu_lock, std::adopt_lock);
    double start = what_time_is_it_now();
    
    if(img.empty()){
        return 0;
    }
    int face_count = 0;
    int detection_bbox[MAX_BBOX_NUM * 4];
    run_yolo_tiny(img, &face_count, detection_bbox);
    /*
    static int index = 0;
    cv::Mat img_clone = img.clone();
    cv::rectangle(img_clone, cvPoint(detection_bbox[0], detection_bbox[1]),
                  cvPoint(detection_bbox[2], detection_bbox[3]), cvScalar(255,0,0), 4);
    cv::imwrite(model_path_prefix + std::to_string(index) + ".jpg", img_clone);
    index += 1;
    */
    printf("face detected thread id %lu, spend: %f, face_count: %d, input image %dx%d, face region %d %d %d %d\n",
         std::hash<std::thread::id>{}(std::this_thread::get_id()), what_time_is_it_now() - start, face_count,
         img.cols, img.cols, detection_bbox[0], detection_bbox[1], detection_bbox[2], detection_bbox[3]);
    return face_count;
}

void init_network_cnn(){
    input_image_yolo = (float *)calloc(416 * 416 * 3, sizeof(float));
    init_yolo_tiny(input_image_yolo, yolo1, yolo2);
    printf("init_yolo_tiny over");
    have_init = true;
}

bool recognition_start(char* model_path){
    printf("model path is : %s", model_path);
    std::lock_guard<std::mutex> gpu_lock_guard(gpu_lock, std::adopt_lock);
    if(have_init) return false;
    if(model_path == NULL) {
        printf("model path is empty! exit");
        exit(-1);
    }
    printf("model path is : %s", model_path);
    std::string model_path_string(model_path);
    std::string data_path = model_path_string + "face34_glint_refine/fc1_w.npy";
    if(access(data_path.c_str(), 0)){
        printf("model path is not exist %s", model_path);
        exit(-1);
    }
    model_path_prefix = model_path_string;
    init_network_cnn();
    return true;
}

void uninit_network_cnn(){
    free(input_image_yolo);
    printf("uninit_network_cnn \n");
}

bool recognition_stop(){
    if(!have_init) return false;
    uninit_network_cnn();
    have_init = false;
    return true;
}

int main(int argc, char **argv)
{
    printf("model path is : ");
    recognition_start("A/");

    cv::Mat img1 = cv::imread("1.jpg");
    std::vector<float> feature;
    int face_count = recognition_face(img1, feature);
    recognition_stop();
    return 0;
}

void normalize_cpu_local(float *x, float *mean, float *variance, int batch, int filters, int spatial)
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

void scale_bias_local(float *output, float *scales, int batch, int n, int size)
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

void add_bias_local(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

int num_detections_local(float thresh)
{
    int s = 0;
    s += yolo_num_detections(yolo_coarse_layer, thresh);
    s += yolo_num_detections(yolo_fine_layer, thresh);
    return s;
}

detection *make_network_boxes_local(float thresh, int *num, int classes)
{
    int nboxes = num_detections_local(thresh);
    if(num) *num = nboxes;
    detection *dets = (detection *)calloc(nboxes, sizeof(detection));
    for(int i = 0; i < nboxes; ++i){
        dets[i].prob = (float *)calloc(classes, sizeof(float));
    }
    return dets;
}

void fill_network_boxes_local(int net_w, int net_h, int w, int h, float thresh, int *map, int relative, detection *dets)
{
    int count = get_yolo_detections(yolo_coarse_layer, w, h, net_w, net_h, thresh, map, relative, dets);
    dets += count;
    count = get_yolo_detections(yolo_fine_layer, w, h, net_w, net_h, thresh, map, relative, dets);
    dets += count;
}

detection *get_network_boxes_local(int net_w, int net_h, int w, int h, float thresh, int *map, int relative, int *num, int classes)
{
    detection *dets = make_network_boxes_local(thresh, num, classes);
    fill_network_boxes_local(net_w, net_h, w, h, thresh, map, relative, dets);
    return dets;
}
