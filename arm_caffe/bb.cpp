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


void run_landmark(cv::Mat& img, cv::Rect &faces, std::vector<cv::Point2f>& extracted_landmarks){
    arm_compute::graph::frontend::Stream graph_landmark(0, "mtcnn_48net");
    std::string data_path = "mtcnn/";
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
    config.tuner_file  = "acl_tuner_mtcnn_48net.csv";

    graph_landmark.finalize(graph_target, config);

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


int main(int argc, char **argv)
{
    cv::Mat image_input = cv::imread("1.jpg");
    int detection_bbox[MAX_BBOX_NUM * 4];
    int face_count = 0;
    for(int i = 0; i < 1; i++) {
        std::vector<cv::Point2f> extracted_landmarks(5, cv::Point2f(0.f, 0.f));
        cv::Rect face_region = cv::Rect(cv::Point(10, 10), cv::Size(64, 64));
        run_landmark(image_input, face_region, extracted_landmarks);
        printf("%d times\n\n", i);
        return 0;
    }
    return 0;
}
