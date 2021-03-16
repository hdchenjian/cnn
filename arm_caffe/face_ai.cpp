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

//arm_compute::graph::Target graph_target = arm_compute::graph::Target::NEON;
arm_compute::graph::Target graph_target = arm_compute::graph::Target::CL;
arm_compute::graph::FastMathHint fast_math_hint = arm_compute::graph::FastMathHint::Enabled; //Disabled;
int num_threads = 0;
bool use_tuner = false;
//arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::EXHAUSTIVE;
arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::NORMAL;
//arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::RAPID;

arm_compute::graph::frontend::Stream graph_face(0, "arcface_34");
arm_compute::graph::frontend::Stream graph_yolo_tiny(0, "yolov3-tiny");
arm_compute::graph::frontend::Stream graph_landmark(0, "mtcnn_48net");

float face_feature[FEATURE_LENGTH] = {0};
float *face_image_input;
float *input_image_yolo = NULL;
float yolo1[COARSE_SIZE * COARSE_SIZE * YOLO_CHANNEL] = {0};
float yolo2[FINE_SIZE * FINE_SIZE * YOLO_CHANNEL] = {0};
float input_landmark[48 * 48 *3] = {0};
float output_landmark[10] = {0};
bool yolo_input_load = false;
bool landmark_input_load = false;
bool arcface_input_load = false;
bool yolo_output_load = false;
bool yolo_output1_load = false;
bool landmark_output_load = false;
bool arcface_output_load = false;

int num_detections_local(float thresh);
detection *make_network_boxes_local(float thresh, int *num, int classes);
void fill_network_boxes_local(int net_w, int net_h, int w, int h, float thresh, int *map, int relative, detection *dets);
detection *get_network_boxes_local(int net_w, int net_h, int w, int h, float thresh, int *map, int relative, int *num, int classes);
void free_yolo_layer(void *input);
yolo_layer *yolo_coarse_layer = 0;
yolo_layer *yolo_fine_layer = 0;

float *fc1_w = NULL;
float *fc1_b = NULL;
float *fc1_scale_w = NULL;
float *fc1_scale_b = NULL;

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

float *load_npy(const std::string &npy_filename){
    std::ifstream _fs;
    arm_compute::DataLayout file_layout = arm_compute::DataLayout::NCHW;
    std::vector<unsigned long> _shape;
    bool _fortran_order = false;
    std::string _typestring = "";
    try {
        _fs.open(npy_filename, std::ios::in | std::ios::binary);
        if(!_fs.good()){
            printf("Failed to load binary data from %s", npy_filename.c_str());
            exit(-1);
        }
        _fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        std::tie(_shape, _fortran_order, _typestring) = arm_compute::utils::parse_npy_header(_fs);
    }
    catch(const std::ifstream::failure &e) {
        printf("Accessing %s: %s", npy_filename.c_str(), e.what());
        exit(-1);
    }
    unsigned int total_size = 1;
    for(int i = 0; i < _shape.size(); i++){
        //printf("_shape %d %ld, _fortran_order %d _typestring %s\n", i, _shape[i], _fortran_order, _typestring.c_str());
        total_size *= _shape[i];
    }
    total_size *= sizeof(float);
    /*const size_t current_position = _fs.tellg();
      _fs.seekg(0, std::ios_base::end);
      const size_t end_position = _fs.tellg();
      _fs.seekg(current_position, std::ios_base::beg);
      unsigned int total_size = (end_position - current_position) * sizeof(float); */
    float *data = (float *)malloc(total_size);
    _fs.read((char *)data, total_size);
    //printf("load_npy: load data from %s, %lu element\n", npy_filename.c_str(), total_size / sizeof(float));
    return data;
}

void add_residual_block(const std::string &data_path, const std::string &name, unsigned int channel, unsigned int num_units,
                        unsigned int stride, const char **prelu_weight_file, int prelu_index, arm_compute::graph::frontend::Stream &graph_net) {
    for(unsigned int i = 0; i < num_units; ++i) {
        std::stringstream unit_path_ss;
        unit_path_ss << name << "_unit" << (i + 1) << "_";
        std::string unit_path = unit_path_ss.str();
        //const arm_compute::TensorShape last_shape = graph.graph().node(graph.tail_node())->output(0)->desc().shape;

        arm_compute::graph::frontend::SubStream route(graph_net);
        if(i == 0){
            route << arm_compute::graph::frontend::ConvolutionLayer(1U, 1U, channel,
                                                                    get_weights_accessor(data_path, unit_path + "conv1sc_w.npy"),
                                                                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                                                    arm_compute::PadStrideInfo(2, 2, 0, 0)).set_name(unit_path + "conv1sc")
                  << arm_compute::graph::frontend::BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_path + "sc_w.npy"),
                      get_weights_accessor(data_path, unit_path + "sc_b.npy"),
                      get_weights_accessor(data_path, unit_path + "sc_scale_w.npy"),
                      get_weights_accessor(data_path, unit_path + "sc_scale_b.npy"), 0.00002f).set_name(unit_path + "sc");
        }

        unsigned int middle_stride = (i == 0) ? stride : 1;
        arm_compute::graph::frontend::SubStream residual(graph_net);
        residual << arm_compute::graph::frontend::BatchNormalizationLayer(get_weights_accessor(data_path, unit_path + "bn1_w.npy"),
                                                                          get_weights_accessor(data_path, unit_path + "bn1_b.npy"),
                                                                          get_weights_accessor(data_path, unit_path + "bn1_scale_w.npy"),
                                                                          get_weights_accessor(data_path, unit_path + "bn1_scale_b.npy"), 0.00002f).set_name(unit_path + "bn1")
                 << arm_compute::graph::frontend::ConvolutionLayer(
                     3U, 3U, channel,
                     get_weights_accessor(data_path, unit_path + "conv1_w.npy"),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name(unit_path + "conv1")
                 << arm_compute::graph::frontend::BatchNormalizationLayer(
                     get_weights_accessor(data_path, unit_path + "bn2_w.npy"),
                     get_weights_accessor(data_path, unit_path + "bn2_b.npy"),
                     get_weights_accessor(data_path, unit_path + "bn2_scale_w.npy"),
                     get_weights_accessor(data_path, unit_path + "bn2_scale_b.npy"), 0.00002f).set_name(unit_path + "bn2")
                 << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, prelu_weight_file[prelu_index + i])).set_name(unit_path + "relu1")
                 << arm_compute::graph::frontend::ConvolutionLayer(
                     3U, 3U, channel,
                     get_weights_accessor(data_path, unit_path + "conv2_w.npy"),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     arm_compute::PadStrideInfo(middle_stride, middle_stride, 1, 1)).set_name(unit_path + "conv2")
                 << arm_compute::graph::frontend::BatchNormalizationLayer(
                     get_weights_accessor(data_path, unit_path + "bn3_w.npy"),
                     get_weights_accessor(data_path, unit_path + "bn3_b.npy"),
                     get_weights_accessor(data_path, unit_path + "bn3_scale_w.npy"),
                     get_weights_accessor(data_path, unit_path + "bn3_scale_b.npy"), 0.00002f).set_name(unit_path + "bn3");
        graph_net << arm_compute::graph::frontend::EltwiseLayer(
            std::move(route), std::move(residual), arm_compute::graph::frontend::EltwiseOperation::Add).set_name(unit_path + "add");
    }
}

void normalize_cpu_local(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void scale_bias_local(float *output, float *scales, int batch, int n, int size);
void add_bias_local(float *output, float *biases, int batch, int n, int size);

void l2normalize_cpu_local(float *x, int batch, int filters, int spatial)
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
        }
    }
}

void run_arcface(cv::Mat &img){
    arcface_input_load = false;
    arcface_output_load = false;
    int channels = img.channels();
    int face_width = 112;
    int face_height = 112;
    for(int k= 0; k < channels; ++k){
        int k_index = k* face_width * face_height;
        for(int m = 0; m < face_height; ++m){
            int j_index = m * face_width;
            for(int n = 0; n < face_width; ++n){
                face_image_input[k_index + j_index + n] = (img.at<cv::Vec3b>(m, n)[channels - k - 1] - 127.5) * 0.0078125;
            }
        }
    }
    graph_face.run();

    int net_batch = 1;
    int output_size = FEATURE_LENGTH;
    if(fc1_w == NULL){
        printf("arcface fc1_w not init!\n");
        exit(-1);
    }
    normalize_cpu_local(face_feature, fc1_w, fc1_b, net_batch, output_size, 1);
    scale_bias_local(face_feature, fc1_scale_w, net_batch, output_size, 1);
    add_bias_local(face_feature, fc1_scale_b, net_batch, output_size, 1);
    l2normalize_cpu_local(face_feature, net_batch, output_size, 1);
}

void init_arcface(float *face_feature, float *face_image_input){
    std::string data_path = model_path_prefix + "face34_glint_refine/";
    printf("init_arcface %s start", data_path.c_str());
    fc1_w = load_npy(data_path + "fc1_w.npy");
    printf("init_arcface %s load_npy over", data_path.c_str());
    fc1_b = load_npy(data_path + "fc1_b.npy");
    fc1_scale_w = load_npy(data_path + "fc1_scale_w.npy");
    fc1_scale_b = load_npy(data_path + "fc1_scale_b.npy");

    const arm_compute::DataLayout weights_layout = arm_compute::DataLayout::NCHW;
    const arm_compute::TensorShape tensor_shape = arm_compute::TensorShape(112U, 112U, 3U, 1U); // DataLayout::NCHW, DataLayout::NCHW);
    arm_compute::graph::TensorDescriptor input_descriptor = arm_compute::graph::TensorDescriptor(
        tensor_shape, arm_compute::DataType::F32).set_layout(weights_layout);

    const char *prelu_weight_file[] = {"relu0_w.npy",
                                       "stage1_unit1_relu1_w.npy", "stage1_unit2_relu1_w.npy", "stage1_unit3_relu1_w.npy",
                                       "stage2_unit1_relu1_w.npy", "stage2_unit2_relu1_w.npy",
                                       "stage2_unit3_relu1_w.npy", "stage2_unit4_relu1_w.npy",
                                       "stage3_unit1_relu1_w.npy", "stage3_unit2_relu1_w.npy", "stage3_unit3_relu1_w.npy",
                                       "stage3_unit4_relu1_w.npy", "stage3_unit5_relu1_w.npy", "stage3_unit6_relu1_w.npy",
                                       "stage4_unit1_relu1_w.npy", "stage4_unit2_relu1_w.npy", "stage4_unit3_relu1_w.npy"};
    graph_face << graph_target
               << fast_math_hint
               << arm_compute::graph::frontend::InputLayer(
                   input_descriptor, arm_compute::support::cpp14::make_unique<LoadInputData>(face_image_input, &arcface_input_load))
               << arm_compute::graph::frontend::ConvolutionLayer(
                   3U, 3U, 64U,
                   get_weights_accessor(data_path, "conv0_w.npy"),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name("conv0")

               << arm_compute::graph::frontend::BatchNormalizationLayer(
                   get_weights_accessor(data_path, "bn0_w.npy"),
                   get_weights_accessor(data_path, "bn0_b.npy"),
                   get_weights_accessor(data_path, "bn0_scale_w.npy"),
                   get_weights_accessor(data_path, "bn0_scale_b.npy"),
                   0.00002f).set_name("bn0")

               << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, prelu_weight_file[0])).set_name("relu0");

    add_residual_block(data_path, "stage1", 64, 3, 2, prelu_weight_file, 1, graph_face);
    add_residual_block(data_path, "stage2", 128, 4, 2, prelu_weight_file, 4, graph_face);
    add_residual_block(data_path, "stage3", 256, 6, 2, prelu_weight_file, 8, graph_face);
    add_residual_block(data_path, "stage4", 512, 3, 2, prelu_weight_file, 14, graph_face);
    graph_face << arm_compute::graph::frontend::BatchNormalizationLayer(get_weights_accessor(data_path, "bn1_w.npy"),
                                                                        get_weights_accessor(data_path, "bn1_b.npy"),
                                                                        get_weights_accessor(data_path, "bn1_scale_w.npy"),
                                                                        get_weights_accessor(data_path, "bn1_scale_b.npy"), 0.00002f).set_name("bn1")
               << arm_compute::graph::frontend::FullyConnectedLayer(512U,
                                                                    get_weights_accessor(data_path, "pre_fc1_w.npy"),
                                                                    get_weights_accessor(data_path, "pre_fc1_b.npy")).set_name("pre_fc1");

    /* BatchNormalizationLayer error for 1 x 1 input
              << arm_compute::graph::frontend::BatchNormalizationLayer(get_weights_accessor(data_path, "fc1_w.npy"),
                                                                       get_weights_accessor(data_path, "fc1_b.npy"),
                                                                       get_weights_accessor(data_path, "fc1_scale_w.npy"),
                                                                       get_weights_accessor(data_path, "fc1_scale_b.npy"), 0.00002f).set_name("fc1")
    */
    graph_face << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(face_feature, &arcface_output_load));

    arm_compute::graph::GraphConfig config;
    config.num_threads = num_threads;
    config.use_tuner = use_tuner;
    config.tuner_mode = cl_tuner_mode;
    config.tuner_file = model_path_prefix + "acl_tuner_arcface.csv";

    graph_face.finalize(graph_target, config);
    /*
    std::cout << "graph.run()" << std::endl;
    graph_face.run();

    int net_batch = 1;
    int output_size = FEATURE_LENGTH;
    if(fc1_w == NULL){
        printf("arcface fc1_w not init!\n");
        exit(-1);
    }
    normalize_cpu(face_feature, fc1_w, fc1_b, net_batch, output_size, 1);
    scale_bias_local(face_feature, fc1_scale_w, net_batch, output_size, 1);
    add_bias_local(face_feature, fc1_scale_b, net_batch, output_size, 1);
    l2normalize_cpu(face_feature, net_batch, output_size, 1);
    //for(int i = 0; i < 10; i++) printf("%d %f\n", i, face_feature[i]);

    return;
    */
}


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

void run_landmark(cv::Mat& img, cv::Rect &faces, std::vector<cv::Point2f>& extracted_landmarks){
    landmark_input_load = false;
    landmark_output_load = false;

    int channels = img.channels();
    int face_width = 48;
    int face_height = 48;
    cv::Mat mtcnn_img;
    cv::resize(img(faces), mtcnn_img, cv::Size(face_width, face_height));
    for(int k= 0; k < channels; ++k){
        for(int m = 0; m < face_height; ++m){
            for(int n = 0; n < face_width; ++n){
                input_landmark[k * face_width * face_height + n *face_width + m] =(mtcnn_img.at<cv::Vec3b>(m, n)[k] - 127.5) * 0.0078125;
            }
        }
    }
    graph_landmark.run();
        
    int roi_w = faces.width;
    int roi_h = faces.height;
    for(int j = 0; j < 5; j++){
        extracted_landmarks[j] = cv::Point(output_landmark[j] * roi_w + faces.x, output_landmark[j + 5] * roi_h +  + faces.y);
        //std::cout << "landmarks " << j << " " << extracted_landmarks[j] << std::endl;
        //cv::circle(img(faces[i]), facial_points[j], 4, cv::Scalar(0,0,255), -1);
    }
}

cv::Mat similarity_matric(std::vector<cv::Point2f> const& srcTri_s, std::vector<cv::Point2f> &dstTri_s) {
    cv::Mat warp_mat(2, 3, CV_32FC1);
    int num_point = srcTri_s.size();
    //int M = srcTri_s.size();
    cv::Mat X(num_point * 2, 4, CV_32FC1);
    cv::Mat U(num_point * 2, 1, CV_32FC1);

    for (int i = 0; i < num_point; i++) {
        U.at<float>(i, 0) = srcTri_s[i].x;
        U.at<float>(i + num_point, 0) = srcTri_s[i].y;
    }

    for (int i = 0; i < num_point; i++) {
        X.at<float>(i, 0) = dstTri_s[i].x;
        X.at<float>(i, 1) = dstTri_s[i].y;
        X.at<float>(i, 2) = 1;
        X.at<float>(i, 3) = 0;

        X.at<float>(i + num_point, 0) = dstTri_s[i].y;
        X.at<float>(i + num_point, 1) = -dstTri_s[i].x;
        X.at<float>(i + num_point, 2) = 0;
        X.at<float>(i + num_point, 3) = 1;
    }

    cv::Mat X_t;
    cv::transpose(X, X_t);
    cv::Mat XX;
    XX = X_t * X;
    cv::Mat r;
    cv::Mat invXX;
    cv::invert(XX, invXX);

    r = invXX * X_t * U;
    cv::Mat Tinv(3, 3, CV_32FC1);
    float sc = r.at<float>(0, 0);
    float ss = r.at<float>(1, 0);
    float tx = r.at<float>(2, 0);
    float ty = r.at<float>(3, 0);
    Tinv.at<float>(0, 0) = sc;
    Tinv.at<float>(0, 1) = -ss;
    Tinv.at<float>(0, 2) = 0;

    Tinv.at<float>(1, 0) = ss;
    Tinv.at<float>(1, 1) = sc;
    Tinv.at<float>(1, 2) = 0;

    Tinv.at<float>(2, 0) = tx;
    Tinv.at<float>(2, 1) = ty;
    Tinv.at<float>(2, 2) = 1;

    cv::Mat T;
    cv::invert(Tinv, T);
    cv::transpose(T, T);

    cv::Mat tmp = T(cv::Rect(0, 0, 3, 2));
    return tmp;
}

int resizeInterestPoints(std::vector<cv::Point2f> const& in, std::vector<cv::Point2f> &out)
{
    double dist = cv::norm(in[0] - in[1]);
    float mTwoEyesPixel = 36;
    int zoom_in = dist / mTwoEyesPixel;
    zoom_in = zoom_in > 0 ? zoom_in : 1;
    for(size_t i = 0; i < in.size(); ++i){
        out[i].x = in[i].x / zoom_in;
        out[i].y = in[i].y / zoom_in;
    }
    return zoom_in;
}

void alignFace(std::vector<cv::Point2f> const& srcTri, cv::Mat const& frame, cv::Mat &aligned_face, cv::Rect &face_region)
{
    cv::Mat warp_mat(2, 3, CV_32FC1);
    std::vector<cv::Point2f> dstTri(5);
    dstTri[0] = cv::Point2f(30.2946f + 8, 51.6963f);
    dstTri[1] = cv::Point2f(65.5318f + 8, 51.5014f);
    dstTri[2] = cv::Point2f(48.0252f + 8, 71.7366f);
    dstTri[3] = cv::Point2f(33.5493f + 8, 92.3655f);
    dstTri[4] = cv::Point2f(62.7299f + 8, 92.2041f);
    /*
    for(int j = 0; j < 5; j++){
        cv::circle(frame(face_region), cv::Point2f(dstTri[j].x * (face_region.width / 112.0f), dstTri[j].y * (face_region.height / 112.0f)), 4, cv::Scalar(0,255,0), -1);
    }
    */
    std::vector<cv::Point2f> srcTri2(srcTri.size());
    int zoom_sz = resizeInterestPoints(srcTri, srcTri2);
    warp_mat = similarity_matric(srcTri2, dstTri);
    cv::Mat frame2;
    cv::resize(frame, frame2, cv::Size(frame.size().width / zoom_sz, frame.size().height / zoom_sz));
    //warp_mat = cv::getAffineTransform(srcTri, dstTri);
    cv::Mat warp_dst; // = cv::Mat::zeros(112, 112, frame.type());
    cv::warpAffine(frame2, warp_dst, warp_mat, cv::Size(112, 112));
    //cv::imwrite("/home/iim/face_align.jpg", warp_dst);
    aligned_face = warp_dst;
}

void init_landmark(float *input_landmark, float *output_landmark){
    std::string data_path = model_path_prefix + "mtcnn/";
    const arm_compute::DataLayout weights_layout = arm_compute::DataLayout::NCHW;
    const arm_compute::TensorShape tensor_shape = arm_compute::TensorShape(48U, 48U, 3U, 1U);
    arm_compute::graph::TensorDescriptor input_descriptor = arm_compute::graph::TensorDescriptor(
        tensor_shape, arm_compute::DataType::F32).set_layout(weights_layout);

    graph_landmark << graph_target
                   << fast_math_hint
                   << arm_compute::graph::frontend::InputLayer(
                       input_descriptor, arm_compute::support::cpp14::make_unique<LoadInputData>(input_landmark, &landmark_input_load))
                   << arm_compute::graph::frontend::ConvolutionLayer(
                       3U, 3U, 32,
                       get_weights_accessor(data_path, "conv1_w.npy"),
                       get_weights_accessor(data_path, "conv1_b.npy"),
                       arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv1")
                   << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "prelu1_w.npy")).set_name("relu1")
                   << arm_compute::graph::frontend::PoolingLayer(
                       arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, 3, arm_compute::PadStrideInfo(2, 2, 1, 1))).set_name("pool1")
              
                   << arm_compute::graph::frontend::ConvolutionLayer(
                       3U, 3U, 64,
                       get_weights_accessor(data_path, "conv2_w.npy"),
                       get_weights_accessor(data_path, "conv2_b.npy"),
                       arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv2")
                   << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "prelu2_w.npy")).set_name("relu2")
                   << arm_compute::graph::frontend::PoolingLayer(
                       arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, 3, arm_compute::PadStrideInfo(2, 2, 0, 0))).set_name("pool2")

                   << arm_compute::graph::frontend::ConvolutionLayer(
                       3U, 3U, 64,
                       get_weights_accessor(data_path, "conv3_w.npy"),
                       get_weights_accessor(data_path, "conv3_b.npy"),
                       arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv3")
                   << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "prelu3_w.npy")).set_name("relu3")
                   << arm_compute::graph::frontend::PoolingLayer(
                       arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, 2, arm_compute::PadStrideInfo(2, 2, 0, 0))).set_name("pool3")

                   << arm_compute::graph::frontend::ConvolutionLayer(
                       2U, 2U, 128,
                       get_weights_accessor(data_path, "conv4_w.npy"),
                       get_weights_accessor(data_path, "conv4_b.npy"),
                       arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv4")
                   << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "prelu4_w.npy")).set_name("relu4")


                   << arm_compute::graph::frontend::FullyConnectedLayer(256U,
                                                                        get_weights_accessor(data_path, "conv5_w.npy"),
                                                                        get_weights_accessor(data_path, "conv5_b.npy")).set_name("conv5")
                   << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "prelu5_w.npy")).set_name("relu5")
              
                   << arm_compute::graph::frontend::FullyConnectedLayer(10U,
                                                                        get_weights_accessor(data_path, "conv6-3_w.npy"),
                                                                        get_weights_accessor(data_path, "conv6-3_b.npy")).set_name("conv6-3")
                   << arm_compute::graph::frontend::OutputLayer(
                       arm_compute::support::cpp14::make_unique<ReadOutputData>(output_landmark, &landmark_output_load));

    arm_compute::graph::GraphConfig config;
    config.num_threads = num_threads;
    config.use_tuner = use_tuner;
    config.tuner_mode = cl_tuner_mode;
    config.tuner_file = model_path_prefix + "acl_tuner_mtcnn_48net.csv";

    graph_landmark.finalize(graph_target, config);
    return;
}

void get_image_feature(cv::Mat &image_input, int *face_count, int *detection_bbox, int *side_face){
    double start = what_time_is_it_now();
    run_yolo_tiny(image_input, face_count, detection_bbox);
    printf("run_yolo_tiny spend: %f s, face_count %d\n", what_time_is_it_now() - start, *face_count);
    //for(int i = 0; i < *face_count * 4; i++) printf("%d %d\n", i, detection_bbox[i]);
/*
    for(int i = 0; i < *face_count; ++i){
        cv::Rect roi_local(cv::Point(detection_bbox[i * 4], detection_bbox[i * 4 + 1]),
                           cv::Size(detection_bbox[i * 4 + 2] - detection_bbox[i * 4],
                                    detection_bbox[i * 4 + 3] - detection_bbox[i * 4 + 1]));
        //printf("face size %d %d\n", roi_local.width, roi_local.height);
        cv::rectangle(image_input, cvPoint(roi_local.x, roi_local.y),
                      cvPoint(roi_local.x + roi_local.width, roi_local.y + roi_local.height), cvScalar(255,0,0), 4);
    }
    static int save_index = 0;
    cv::imwrite("testss" + std::to_string(save_index) + ".jpg", image_input);
    save_index += 1;
*/

    if(*face_count == 0) return;
    cv::Rect face_region;
    if(*face_count == 1) {
        face_region = cv::Rect(cv::Point(detection_bbox[0], detection_bbox[1]),
                               cv::Size(detection_bbox[2] - detection_bbox[0], detection_bbox[3] - detection_bbox[1]));
    } else {
        int first_face_size = (detection_bbox[2] - detection_bbox[0]) * (detection_bbox[3] - detection_bbox[1]);
        int max_face_size_index = 0;
        for(int i = 1; i < *face_count; ++i){
            int face_size = (detection_bbox[i * 4 + 2] - detection_bbox[i * 4]) * (detection_bbox[i * 4 + 3] - detection_bbox[i * 4 + 1]);
            if(first_face_size < face_size){
                first_face_size = face_size;
                max_face_size_index = i;
            }
        }
        *face_count = 1;
        face_region = cv::Rect(cv::Point(detection_bbox[max_face_size_index * 4], detection_bbox[max_face_size_index * 4 + 1]),
                               cv::Size(detection_bbox[max_face_size_index * 4 + 2] - detection_bbox[max_face_size_index * 4],
                                        detection_bbox[max_face_size_index * 4 + 3] - detection_bbox[max_face_size_index * 4 + 1]));
        if(max_face_size_index > 0){
            for(int i = 1; i < 4; ++i) detection_bbox[i] = detection_bbox[max_face_size_index * 4 + i];
        }
    }

    std::vector<cv::Point2f> extracted_landmarks(5, cv::Point2f(0.f, 0.f));
    run_landmark(image_input, face_region, extracted_landmarks);
    float two_eye_height_diff = fabs((extracted_landmarks[1].y - extracted_landmarks[0].y) / face_region.height);
    float two_lips_height_diff = fabs((extracted_landmarks[4].y - extracted_landmarks[3].y) / face_region.height);
    float two_eye_width_diff = fabs((extracted_landmarks[1].x - extracted_landmarks[0].x) / face_region.width);
    if(two_eye_height_diff > 0.05 ||  two_lips_height_diff > 0.05 || two_eye_width_diff < 0.35){
        printf("side face: %f %f %f\n", two_eye_height_diff, two_lips_height_diff, two_eye_width_diff);
        *side_face = 1;
    } else {
        printf("not side face: %f %f %f\n", two_eye_height_diff, two_lips_height_diff, two_eye_width_diff);
        *side_face = 0;
    }
    printf("run_landmark spend: %f s\n", what_time_is_it_now() - start);
    /*
    extracted_landmarks[0] = cv::Point2f(1734.0f, 1103.0f);
    extracted_landmarks[1] = cv::Point2f(2215.0f, 1071.0f);
    extracted_landmarks[2] = cv::Point2f(1997.0f, 1400.0f);
    extracted_landmarks[3] = cv::Point2f(1774.0f, 1649.0f);
    extracted_landmarks[4] = cv::Point2f(2198.0f, 1618.0f);
    */
    cv::Mat face_resized;
    alignFace(extracted_landmarks, image_input, face_resized, face_region);

    //std::cout << "face_region " << face_region << std::endl;
    run_arcface(face_resized);
    printf("run_arcface spend: %f s\n", what_time_is_it_now() - start);
    //for(int i = 0; i < 3; i++) printf("%d %f\n", i, face_feature[i]);
}

int recognition_face(cv::Mat &img, std::vector<float> &feature){
    std::lock_guard<std::mutex> gpu_lock_guard(gpu_lock, std::adopt_lock);
    double start = what_time_is_it_now();
    
    if(img.empty()){
        return 0;
    }
    int face_count = 0;
    int detection_bbox[MAX_BBOX_NUM * 4];

    int side_face = 0;
    get_image_feature(img, &face_count, detection_bbox, &side_face);
    if(face_count == 0){
        return 0;
    }

    /*
    static int index = 0;
    cv::Mat img_clone = img.clone();
    cv::rectangle(img_clone, cvPoint(detection_bbox[0], detection_bbox[1]),
                  cvPoint(detection_bbox[2], detection_bbox[3]), cvScalar(255,0,0), 4);
    cv::imwrite(model_path_prefix + std::to_string(index) + ".jpg", img_clone);
    index += 1;
    */
    printf("face detected thread id %lu, spend: %f, face_count: %d, input image %dx%d, face region %d %d %d %d, side_face %d\n",
         std::hash<std::thread::id>{}(std::this_thread::get_id()), what_time_is_it_now() - start, face_count,
         img.cols, img.cols, detection_bbox[0], detection_bbox[1], detection_bbox[2], detection_bbox[3], side_face);
    std::vector<float> feature_tmp(face_feature, face_feature + FEATURE_LENGTH);
    feature = feature_tmp; 
    return face_count;
}

void init_network_cnn(){
    face_image_input = (float *)calloc(112 * 112 * 3, sizeof(float));
    init_arcface(face_feature, face_image_input);
    printf("init_arcface over");
    init_landmark(input_landmark, output_landmark);
    printf("init_landmark over");
    //init_yolo(input_image_yolo, yolo1, yolo2);
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
    free(fc1_w);
    free(fc1_b);
    free(fc1_scale_w);
    free(fc1_scale_b);
    free(face_image_input);
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
    printf("face_count %d\n", face_count);

    cv::Mat img2 = cv::imread("2.jpg");
    std::vector<float> feature2;
    for(int i = 0; i < 10; i++){
        face_count = recognition_face(img2, feature2);
    }
    printf("face_count %d\n", face_count);
    float score = 0;
    for(int i = 0; i < FEATURE_LENGTH; i++){
        score += feature[i] * feature2[i];
    }
    printf("score %f\n", score);
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
