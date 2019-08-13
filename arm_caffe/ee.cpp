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
arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::EXHAUSTIVE;
//arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::NORMAL;

arm_compute::graph::frontend::Stream graph_spoofing(0, "spoofing_binary");
float spoofing_result[2] = {0};
float spoofing_image_input[96 * 112 *3] = {0};
bool spoofing_input_load = false;
bool spoofing_output_load = false;


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


void add_spoofing_residual_block(const std::string &data_path, const std::string &unit_path, unsigned int channel,
                              arm_compute::graph::frontend::Stream &graph_net){
    arm_compute::ActivationLayerInfo active_info = arm_compute::ActivationLayerInfo(
        arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f);

    arm_compute::graph::frontend::SubStream route(graph_net);
    arm_compute::graph::frontend::SubStream residual(graph_net);
    residual << arm_compute::graph::frontend::ConvolutionLayer(
                 3U, 3U, channel,
                 get_weights_accessor(data_path, unit_path + "conv1_w.npy"),
                 get_weights_accessor(data_path, unit_path + "conv1_b.npy"),
                 arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name(unit_path + "conv1")
             << arm_compute::graph::frontend::ActivationLayer(active_info).set_name(unit_path + "relu1")
    
             << arm_compute::graph::frontend::ConvolutionLayer(
                 3U, 3U, channel,
                 get_weights_accessor(data_path, unit_path + "conv2_w.npy"),
                 get_weights_accessor(data_path, unit_path + "conv2_b.npy"),
                 arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name(unit_path + "conv2")
             << arm_compute::graph::frontend::ActivationLayer(active_info).set_name(unit_path + "relu2");

    graph_net << arm_compute::graph::frontend::EltwiseLayer(
        std::move(route), std::move(residual), arm_compute::graph::frontend::EltwiseOperation::Add).set_name(unit_path + "add");
    
}

void add_spoofing_route_block(const std::string &data_path, const std::string &unit_path, unsigned int channel,
                              arm_compute::graph::frontend::Stream &graph_net){
    arm_compute::ActivationLayerInfo active_info = arm_compute::ActivationLayerInfo(
        arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f);

    arm_compute::graph::frontend::SubStream route(graph_net);
    route << arm_compute::graph::frontend::ConvolutionLayer(1U, 1U, channel,
                                                            get_weights_accessor(data_path, unit_path + "conv1sc_w.npy"),
                                                            get_weights_accessor(data_path, unit_path + "conv1sc_b.npy"),
                                                            arm_compute::PadStrideInfo(2, 2, 0, 0)).set_name(unit_path + "conv1sc")
          << arm_compute::graph::frontend::ActivationLayer(active_info).set_name(unit_path + "relu3");

    arm_compute::graph::frontend::SubStream residual(graph_net);
    residual << arm_compute::graph::frontend::ConvolutionLayer(
                 3U, 3U, channel,
                 get_weights_accessor(data_path, unit_path + "conv1_w.npy"),
                 get_weights_accessor(data_path, unit_path + "conv1_b.npy"),
                 arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name(unit_path + "conv1")
             << arm_compute::graph::frontend::ActivationLayer(active_info).set_name(unit_path + "relu1")
    
             << arm_compute::graph::frontend::ConvolutionLayer(
                 3U, 3U, channel,
                 get_weights_accessor(data_path, unit_path + "conv2_w.npy"),
                 get_weights_accessor(data_path, unit_path + "conv2_b.npy"),
                 arm_compute::PadStrideInfo(2, 2, 1, 1)).set_name(unit_path + "conv2")
             << arm_compute::graph::frontend::ActivationLayer(active_info).set_name(unit_path + "relu2");

    graph_net << arm_compute::graph::frontend::EltwiseLayer(
        std::move(route), std::move(residual), arm_compute::graph::frontend::EltwiseOperation::Add).set_name(unit_path + "add");
    
}

void init_spoofing(){
    std::string data_path = "/sdcard/A/spoofing_binary/";
    const arm_compute::DataLayout weights_layout = arm_compute::DataLayout::NCHW;
    const arm_compute::TensorShape tensor_shape = arm_compute::TensorShape(96U, 112U, 3U, 1U);
    arm_compute::graph::TensorDescriptor input_descriptor = arm_compute::graph::TensorDescriptor(
        tensor_shape, arm_compute::DataType::F32).set_layout(weights_layout);

    arm_compute::ActivationLayerInfo active_info = arm_compute::ActivationLayerInfo(
        arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f);

    graph_spoofing << graph_target
                   << fast_math_hint
                   << arm_compute::graph::frontend::InputLayer(
                       input_descriptor, arm_compute::support::cpp14::make_unique<LoadInputData>(spoofing_image_input, &spoofing_input_load))
                   << arm_compute::graph::frontend::ConvolutionLayer(
                       3U, 3U, 64U,
                       get_weights_accessor(data_path, "conv0_w.npy"),
                       get_weights_accessor(data_path, "conv0_b.npy"),
                       arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name("conv0")

                   << arm_compute::graph::frontend::ActivationLayer(active_info).set_name("relu0");

    add_spoofing_route_block(data_path, "stage1_unit1_", 64, graph_spoofing);
    
    add_spoofing_residual_block(data_path, "stage2_unit1_", 64, graph_spoofing);
    add_spoofing_residual_block(data_path, "stage2_unit2_", 64, graph_spoofing);
    add_spoofing_route_block(data_path, "stage2_unit3_", 128, graph_spoofing);

    add_spoofing_residual_block(data_path, "stage3_unit1_", 128, graph_spoofing);
    add_spoofing_residual_block(data_path, "stage3_unit2_", 128, graph_spoofing);
    add_spoofing_route_block(data_path, "stage3_unit3_", 256, graph_spoofing);

    add_spoofing_residual_block(data_path, "stage4_unit1_", 256, graph_spoofing);
    add_spoofing_residual_block(data_path, "stage4_unit2_", 256, graph_spoofing);
    add_spoofing_route_block(data_path, "stage4_unit3_", 512, graph_spoofing);

    add_spoofing_residual_block(data_path, "stage5_unit1_", 512, graph_spoofing);
    add_spoofing_residual_block(data_path, "stage5_unit2_", 512, graph_spoofing);
    
    graph_spoofing << arm_compute::graph::frontend::PoolingLayer(
        arm_compute::PoolingLayerInfo(
            arm_compute::PoolingType::AVG, arm_compute::Size2D(6, 7), arm_compute::PadStrideInfo(1, 1, 0, 0))).set_name("pool1")
                   << arm_compute::graph::frontend::ConvolutionLayer(
                       1U, 1U, 2U,
                       get_weights_accessor(data_path, "conv00_w.npy"),
                       get_weights_accessor(data_path, "conv00_b.npy"),
                       arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv00");

    graph_spoofing << arm_compute::graph::frontend::OutputLayer(
        arm_compute::support::cpp14::make_unique<ReadOutputData>(spoofing_result, &spoofing_output_load));

    arm_compute::graph::GraphConfig config;
    config.num_threads = num_threads;
    config.use_tuner = use_tuner;
    config.tuner_mode = cl_tuner_mode;
    config.tuner_file  = "acl_tuner_spoofing.csv";

    graph_spoofing.finalize(graph_target, config);
}

void run_spoofing(cv::Mat &img){
    spoofing_input_load = false;
    spoofing_output_load = false;
    int channels = img.channels();
    int spoofing_width = 96;
    int spoofing_height = 112;
    for(int k= 0; k < channels; ++k){
        int k_index = k* spoofing_width * spoofing_height;
        for(int m = 0; m < spoofing_height; ++m){
            int j_index = m * spoofing_width;
            for(int n = 0; n < spoofing_width; ++n){
                spoofing_image_input[k_index + j_index + n] = (img.at<cv::Vec3b>(m, n)[channels - k - 1] - 127.5) * 0.0078125;
            }
        }
    }
    graph_spoofing.run();
}

int main(int argc, char **argv)
{
    cv::Mat image_input = cv::imread("0_rgb_1565349352_148.jpg");
    init_spoofing();
    for(int i = 0; i < 30; i++) {        
        double start = what_time_is_it_now();
        run_spoofing(image_input);
        printf("%d times %f %f, spend %f\n\n", i, spoofing_result[0], spoofing_result[1], what_time_is_it_now() - start);
    }
    return 0;
}
