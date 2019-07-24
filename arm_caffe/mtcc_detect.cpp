#include "utils/GraphUtils.h"
#include "arm_compute/graph.h"
#include "utils/Utils.h"

#include "arm_compute/runtime/CL/CLTensor.h"

arm_compute::graph::Target graph_target = arm_compute::graph::Target::NEON;
//arm_compute::graph::Target graph_target = arm_compute::graph::Target::CL;
arm_compute::graph::FastMathHint fast_math_hint = arm_compute::graph::FastMathHint::Enabled; //Disabled;
int num_threads = 0;
bool use_tuner = true;

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
    LoadInputData() : _already_loaded(false){}

    bool access_tensor(arm_compute::ITensor &tensor) override {
        //std::cout << "LoadInputData access_tensor" << std::endl;
        if(!_already_loaded){
            if(tensor.info()->data_type() != arm_compute::DataType::F32){
                printf("Unsupported format\n");
                exit(-1);
            }
            arm_compute::Window window;
            window.use_tensor_dimensions(tensor.info()->tensor_shape());
            arm_compute::utils::map(tensor, true);
            arm_compute::Iterator it(&tensor, window);

            execute_window_loop(window, [&](const arm_compute::Coordinates & id) {
                    *reinterpret_cast<float *>(it.ptr()) = 0.5f;
                },
                it);
            arm_compute::utils::unmap(tensor);
        }
        _already_loaded = !_already_loaded;
        return _already_loaded;
    }

private:
    bool _already_loaded;
};

class ReadOutputData final : public arm_compute::graph::ITensorAccessor
{
public:
    ReadOutputData(float *output_data) : _already_read(false), data(output_data){}

    bool access_tensor(arm_compute::ITensor &tensor) override {
        //std::cout << "ReadOutputData access_tensor" << std::endl;
        if(!_already_read){
            if(tensor.info()->data_type() != arm_compute::DataType::F32){
                printf("Unsupported format\n");
                exit(-1);
            }
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
        _already_read = !_already_read;
        return _already_read;
    }

private:
    bool _already_read;
    float *data;
};

void mtcnn_12net(){
    arm_compute::graph::frontend::Stream graph_net(0, "12net_scale_1");
    std::string data_path = "mtcnn/12net/";
    const arm_compute::DataLayout weights_layout = arm_compute::DataLayout::NCHW;
    const arm_compute::TensorShape tensor_shape = arm_compute::TensorShape(416U, 416U, 3U, 1U); // DataLayout::NCHW, DataLayout::NCHW);
    arm_compute::graph::TensorDescriptor input_descriptor = arm_compute::graph::TensorDescriptor(
        tensor_shape, arm_compute::DataType::F32).set_layout(weights_layout);

    int prob1_size = 2 * 203 * 203;
    float *conv4_2 = (float *)calloc(100 * prob1_size * 2, sizeof(float));
    float *prob1 = (float *)malloc(prob1_size * sizeof(float));
    float *conv4_1 = (float *)malloc(prob1_size * sizeof(float));

    graph_net << graph_target
              << fast_math_hint
              << arm_compute::graph::frontend::InputLayer(input_descriptor, arm_compute::support::cpp14::make_unique<LoadInputData>())
              << arm_compute::graph::frontend::ConvolutionLayer(
                  3U, 3U, 10,
                  get_weights_accessor(data_path, "conv1_w.npy"),
                  get_weights_accessor(data_path, "conv1_b.npy"),
                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv1")
              << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "PReLU1_w.npy")).set_name("relu1")
              << arm_compute::graph::frontend::PoolingLayer(
                  arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, 2, arm_compute::PadStrideInfo(2, 2, 0, 0))).set_name("pool1")

              
              << arm_compute::graph::frontend::ConvolutionLayer(
                  3U, 3U, 16,
                  get_weights_accessor(data_path, "conv2_w.npy"),
                  get_weights_accessor(data_path, "conv2_b.npy"),
                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv2")
              << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "PReLU2_w.npy")).set_name("relu2")

              << arm_compute::graph::frontend::ConvolutionLayer(
                  3U, 3U, 32,
                  get_weights_accessor(data_path, "conv3_w.npy"),
                  get_weights_accessor(data_path, "conv3_b.npy"),
                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv3")
              << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "PReLU3_w.npy")).set_name("relu3");

    arm_compute::graph::frontend::SubStream net12_route(graph_net);

    net12_route << arm_compute::graph::frontend::ConvolutionLayer(1U, 1U, 2,
                                                                  get_weights_accessor(data_path, "conv4-1_w.npy"),
                                                                  get_weights_accessor(data_path, "conv4-1_b.npy"),
                                                                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv4-1")

        //<< arm_compute::graph::frontend::SoftmaxLayer().set_name("prob1")
                << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(conv4_1));

    graph_net << arm_compute::graph::frontend::ConvolutionLayer(
                  1U, 1U, 4,
                  get_weights_accessor(data_path, "conv4-2_w.npy"),
                  get_weights_accessor(data_path, "conv4-2_b.npy"),
                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv4-2")
              << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(conv4_2));

    arm_compute::graph::GraphConfig config;
    config.num_threads = num_threads;
    config.use_tuner = use_tuner;
    config.tuner_mode = arm_compute::CLTunerMode::EXHAUSTIVE;
    config.tuner_file  = "acl_tuner_mtcnn_12net.csv";

    graph_net.finalize(graph_target, config);
    std::cout << "graph.run()" << std::endl;
    graph_net.run();


    int prob1_size_channel = 203 * 203;
    float *softmax_max = (float *)malloc(prob1_size_channel * sizeof(float));
    float *softmax_sum = (float *)calloc(prob1_size_channel,  sizeof(float));
    float *conv4_1_index = conv4_1 + prob1_size_channel;
    for(int i = 0; i < prob1_size_channel; ++i){
        softmax_max[i] = conv4_1[i] > conv4_1_index[i] ? conv4_1[i] : conv4_1_index[i];
    }
    for (int k = 0; k < 2; k++){
        float *data_prt = conv4_1 + k * prob1_size_channel;
        float *data_output = prob1 + k * prob1_size_channel;
        for(int i = 0; i < prob1_size_channel; ++i){
            float e = expf(data_prt[i] - softmax_max[i]);
            softmax_sum[i] += e;
            data_output[i] = e;
        }
    }
    
    for (int k = 0; k < 2; k++){
        float *data_output = prob1 + k * prob1_size_channel;
        for(int i = 0; i < prob1_size_channel; ++i){
            data_output[i] /= softmax_sum[i];
        }
    }
    for(int i = 0; i < 10; i++) printf("%d %f\n", i, conv4_2[i]);
    for(int i = 0; i < 10; i++) printf("%d %f\n", i, prob1[i]);

    double start_time = what_time_is_it_now();
    int run_times = 10;
    for(int i = 0; i < run_times; i++) graph_net.run();
    double end_time = what_time_is_it_now();
    printf("mtcnn_12net spend %f\n", (end_time - start_time) / run_times);
    return;
}

void mtcnn_24net(){
    arm_compute::graph::frontend::Stream graph_net(0, "24net_scale_1");
    std::string data_path = "mtcnn/24net/";
    const arm_compute::DataLayout weights_layout = arm_compute::DataLayout::NCHW;
    const arm_compute::TensorShape tensor_shape = arm_compute::TensorShape(416U, 416U, 3U, 1U); // DataLayout::NCHW, DataLayout::NCHW);
    arm_compute::graph::TensorDescriptor input_descriptor = arm_compute::graph::TensorDescriptor(
        tensor_shape, arm_compute::DataType::F32).set_layout(weights_layout);

    int prob1_size = 2 * 203 * 203;
    float *conv4_2 = (float *)calloc(100 * prob1_size * 2, sizeof(float));
    float *prob1 = (float *)malloc(prob1_size * sizeof(float));
    float *conv4_1 = (float *)malloc(prob1_size * sizeof(float));

    graph_net << graph_target
              << fast_math_hint
              << arm_compute::graph::frontend::InputLayer(input_descriptor, arm_compute::support::cpp14::make_unique<LoadInputData>())
              << arm_compute::graph::frontend::ConvolutionLayer(
                  3U, 3U, 10,
                  get_weights_accessor(data_path, "conv1_w.npy"),
                  get_weights_accessor(data_path, "conv1_b.npy"),
                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv1")
              << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "PReLU1_w.npy")).set_name("relu1")
              << arm_compute::graph::frontend::PoolingLayer(
                  arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, 2, arm_compute::PadStrideInfo(2, 2, 0, 0))).set_name("pool1")

              
              << arm_compute::graph::frontend::ConvolutionLayer(
                  3U, 3U, 16,
                  get_weights_accessor(data_path, "conv2_w.npy"),
                  get_weights_accessor(data_path, "conv2_b.npy"),
                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv2")
              << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "PReLU2_w.npy")).set_name("relu2")

              << arm_compute::graph::frontend::ConvolutionLayer(
                  3U, 3U, 32,
                  get_weights_accessor(data_path, "conv3_w.npy"),
                  get_weights_accessor(data_path, "conv3_b.npy"),
                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv3")
              << arm_compute::graph::frontend::PreluLayer(get_weights_accessor(data_path, "PReLU3_w.npy")).set_name("relu3");

    arm_compute::graph::frontend::SubStream net12_route(graph_net);

    net12_route << arm_compute::graph::frontend::ConvolutionLayer(1U, 1U, 2,
                                                                  get_weights_accessor(data_path, "conv4-1_w.npy"),
                                                                  get_weights_accessor(data_path, "conv4-1_b.npy"),
                                                                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv4-1")

        //<< arm_compute::graph::frontend::SoftmaxLayer().set_name("prob1")
                << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(conv4_1));

    graph_net << arm_compute::graph::frontend::ConvolutionLayer(
                  1U, 1U, 4,
                  get_weights_accessor(data_path, "conv4-2_w.npy"),
                  get_weights_accessor(data_path, "conv4-2_b.npy"),
                  arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv4-2")
              << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(conv4_2));

    arm_compute::graph::GraphConfig config;
    config.num_threads = num_threads;
    config.use_tuner = use_tuner;
    config.tuner_mode = arm_compute::CLTunerMode::EXHAUSTIVE;
    config.tuner_file  = "acl_tuner_mtcnn_24net.csv";

    graph_net.finalize(graph_target, config);
    std::cout << "graph.run()" << std::endl;
    graph_net.run();


    int prob1_size_channel = 203 * 203;
    float *softmax_max = (float *)malloc(prob1_size_channel * sizeof(float));
    float *softmax_sum = (float *)calloc(prob1_size_channel,  sizeof(float));
    float *conv4_1_index = conv4_1 + prob1_size_channel;
    for(int i = 0; i < prob1_size_channel; ++i){
        softmax_max[i] = conv4_1[i] > conv4_1_index[i] ? conv4_1[i] : conv4_1_index[i];
    }
    for (int k = 0; k < 2; k++){
        float *data_prt = conv4_1 + k * prob1_size_channel;
        float *data_output = prob1 + k * prob1_size_channel;
        for(int i = 0; i < prob1_size_channel; ++i){
            float e = expf(data_prt[i] - softmax_max[i]);
            softmax_sum[i] += e;
            data_output[i] = e;
        }
    }
    
    for (int k = 0; k < 2; k++){
        float *data_output = prob1 + k * prob1_size_channel;
        for(int i = 0; i < prob1_size_channel; ++i){
            data_output[i] /= softmax_sum[i];
        }
    }
    for(int i = 0; i < 10; i++) printf("%d %f\n", i, conv4_2[i]);
    for(int i = 0; i < 10; i++) printf("%d %f\n", i, prob1[i]);

    double start_time = what_time_is_it_now();
    int run_times = 10;
    for(int i = 0; i < run_times; i++) graph_net.run();
    double end_time = what_time_is_it_now();
    printf("mtcnn_24net spend %f\n", (end_time - start_time) / run_times);
    return;
}

int main(int argc, char **argv)
{
    //mtcnn_12net();
    mtcnn_24net();
    return 0;
}

