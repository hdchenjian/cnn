#include <jni.h>
#include <android/log.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils/GraphUtils.h"
#include "arm_compute/graph.h"
#include "utils/Utils.h"

#define YOLO_CHANNEL 18
#define COARSE_SIZE 13
#define FINE_SIZE 26
#define MAX_BBOX_NUM 5
#define FEATURE_LENGTH 512

//arm_compute::graph::Target graph_target = arm_compute::graph::Target::NEON;
arm_compute::graph::Target graph_target = arm_compute::graph::Target::CL;
arm_compute::graph::FastMathHint fast_math_hint = arm_compute::graph::FastMathHint::Enabled; //Disabled;
int num_threads = 0;
bool use_tuner = true;
//arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::EXHAUSTIVE;
arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::NORMAL;

float face_feature[FEATURE_LENGTH] = {0};
float face_image_input[112 * 112 *3] = {0};
float input_image_yolo[416 * 416 * 3] = {0};
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
            
            const int idx_width = arm_compute::get_data_layout_dimension_index(
                arm_compute::DataLayout::NCHW, arm_compute::DataLayoutDimension::WIDTH);
            const int idx_height = arm_compute::get_data_layout_dimension_index(
                arm_compute::DataLayout::NCHW, arm_compute::DataLayoutDimension::HEIGHT);
            const int idx_channel = arm_compute::get_data_layout_dimension_index(
                arm_compute::DataLayout::NCHW, arm_compute::DataLayoutDimension::CHANNEL);
            printf("output tensor w %zd h %zd c %zd\n",
                   tensor.info()->dimension(idx_width), tensor.info()->dimension(idx_height), tensor.info()->dimension(idx_channel));
            
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

void run_yolo_tiny(cv::Mat &image_input, int *total_bbox_num, int *detection_bbox){
    arm_compute::graph::frontend::Stream graph_yolo_tiny(0, "yolov3-tiny");
    const char *mask_str_coarse = "3,4,5";
    const char *anchors = "10,14,  23,27,  37,58,  81,82,  135,169,  344,319";
    const char *mask_str_fine = "0,1,2";

    std::string data_path = "yolov3_tiny/";
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
    config.tuner_file  = "acl_tuner_yolov3_tiny.csv";

    graph_yolo_tiny.finalize(graph_target, config);

    double start = what_time_is_it_now();
    set_yolo_input(image_input);
    graph_yolo_tiny.run();
    for(int i = 0; i < 3; i++) printf("yolo1 %d %f\n", i, yolo1[i]);
    for(int i = 0; i < 3; i++) printf("yolo2 %d %f\n", i, yolo2[i]);
    //return;
    //get_yolo_result(image_input, total_bbox_num, detection_bbox);
}

int main(int argc, char **argv)
{
    cv::Mat image_input = cv::imread("1.jpg");
    int detection_bbox[MAX_BBOX_NUM * 4];
    int face_count = 0;
    for(int i = 0; i < 1; i++) {
        run_yolo_tiny(image_input, &face_count, detection_bbox);
        printf("%d times\n\n", i);
        return 0;
    }
    return 0;
}
