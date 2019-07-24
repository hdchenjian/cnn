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


void run_arcface(cv::Mat &img){
    arm_compute::graph::frontend::Stream graph_face(0, "arcface_34");
    std::string data_path = "face34/";
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
    config.tuner_file  = "acl_tuner_arcface.csv";

    graph_face.finalize(graph_target, config);

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
}


int main(int argc, char **argv)
{
    cv::Mat image_input = cv::imread("1.jpg");
    int detection_bbox[MAX_BBOX_NUM * 4];
    int face_count = 0;
    for(int i = 0; i < 1; i++) {
        std::vector<cv::Point2f> extracted_landmarks(5, cv::Point2f(0.f, 0.f));
        cv::Rect face_region = cv::Rect(cv::Point(0, 0), cv::Size(112, 112));
        cv::Mat face_resized = image_input(face_region);
        run_arcface(face_resized);
        printf("%d times\n\n", i);
        return 0;
    }
    return 0;
}
