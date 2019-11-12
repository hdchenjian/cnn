#include <unistd.h>
#include <jni.h>
#include <android/log.h>

#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "utils/GraphUtils.h"
#include "arm_compute/graph.h"
#include "utils/Utils.h"

#include "yolo_layer.h"

#define TAG "my-jni"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#ifdef __cplusplus
extern "C"{
#endif
    JNIEXPORT jboolean JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_recognition_1start(
        JNIEnv *env, jobject obj, jstring model_path_java, jint use_spoofing);
    JNIEXPORT jboolean JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_recognition_1stop(JNIEnv *env, jobject obj);
    JNIEXPORT jint JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_recognition_1face(
        JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jfloatArray feature_save, jlongArray code_ret,
        jint width, jint height, jintArray is_side_face);
    JNIEXPORT jbyteArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_face_1region(
        JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jint width, jint height);
    JNIEXPORT jint JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_detect_1face(
        JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jint width, jint height);
    JNIEXPORT jintArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_yuv2bitmap_1native(
        JNIEnv *env, jobject obj, jbyteArray image_data, jint width, jint height, jint height_out);
    JNIEXPORT jbyteArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_bitmap2rgb_1native(
        JNIEnv *env, jobject obj, jintArray image_data);
    JNIEXPORT jintArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_rgb2bitmap_1native(
        JNIEnv *env, jobject obj, jbyteArray image_data);
    JNIEXPORT jbyteArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_yv122rgb_1native(
        JNIEnv *env, jobject obj, jbyteArray image_data, jint width, jint height);
    JNIEXPORT jbyteArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_yuv2rgb_1native(
        JNIEnv *env, jobject obj, jbyteArray image_data, jint width, jint height);
    JNIEXPORT void JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_save_1spoofingimage(
        JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jint width, jint height, jint is_rgb, jint is_real);
    JNIEXPORT int JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_run_1spoofing(
        JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jint width, jint height);
#ifdef __cplusplus
}
#endif

cv::CascadeClassifier faces_cascade;

bool have_init = false;
bool have_init_opencv_detect = false;
std::string model_path_prefix = "/sdcard/A/";
//std::string model_path_prefix = "/storage/emulated/0/Android/data/com.iim.vbook/cache/A/";
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
arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::EXHAUSTIVE;
//arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::NORMAL;
//arm_compute::CLTunerMode cl_tuner_mode = arm_compute::CLTunerMode::RAPID;

arm_compute::graph::frontend::Stream graph_face(0, "arcface_34");
arm_compute::graph::frontend::Stream graph_yolo(0, "yolov3");
arm_compute::graph::frontend::Stream graph_yolo_tiny(0, "yolov3-tiny");
arm_compute::graph::frontend::Stream graph_landmark(0, "mtcnn_48net");

arm_compute::graph::frontend::Stream graph_spoofing(0, "spoofing_binary");
float spoofing_result[2] = {0};
float spoofing_image_input[96 * 112 *3] = {0};
bool spoofing_input_load = false;
bool spoofing_output_load = false;

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
    LOGE("init_arcface %s start", data_path.c_str());
    fc1_w = load_npy(data_path + "fc1_w.npy");
    LOGE("init_arcface %s load_npy over", data_path.c_str());
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

void add_convolutional_yolo(const std::string &data_path, unsigned int channel, unsigned int kernel, int pad, int weight_index,
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

void add_convolutional_yolo_sub(const std::string &data_path, unsigned int channel, unsigned int kernel, int pad, int weight_index,
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

void add_residual_yolo(const std::string &data_path, unsigned int channel, int weight_index, arm_compute::graph::frontend::Stream &graph_net) {
    arm_compute::ActivationLayerInfo active_info = arm_compute::ActivationLayerInfo(
        arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f);
    std::string unit_path = std::to_string(weight_index);
    std::string unit_path1 = std::to_string(weight_index + 1);
    std::string unit_path2 = std::to_string(weight_index + 2);

    graph_net << arm_compute::graph::frontend::ConvolutionLayer(3U, 3U, channel * 2,
                                                                get_weights_accessor(data_path, "conv" + unit_path + "_w.npy"),
                                                                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                                                arm_compute::PadStrideInfo(2, 2, 1, 1)).set_name("conv" + unit_path)
              << arm_compute::graph::frontend::BatchNormalizationLayer(
                  get_weights_accessor(data_path, "bn" + unit_path + "_w.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_b.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_scale_w.npy"),
                  get_weights_accessor(data_path, "bn" + unit_path + "_scale_b.npy"), 0.00002f).set_name("bn" + unit_path)
              << arm_compute::graph::frontend::ActivationLayer(active_info).set_name("relu" + unit_path);

    arm_compute::graph::frontend::SubStream route(graph_net);
    arm_compute::graph::frontend::SubStream residual(graph_net);
    residual << arm_compute::graph::frontend::ConvolutionLayer(
                 1U, 1U, channel,
                 get_weights_accessor(data_path, "conv" + unit_path1 + "_w.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv" + unit_path1)
             << arm_compute::graph::frontend::BatchNormalizationLayer(
                 get_weights_accessor(data_path, "bn" + unit_path1 + "_w.npy"),
                 get_weights_accessor(data_path, "bn" + unit_path1 + "_b.npy"),
                 get_weights_accessor(data_path, "bn" + unit_path1 + "_scale_w.npy"),
                 get_weights_accessor(data_path, "bn" + unit_path1 + "_scale_b.npy"), 0.00002f).set_name("bn" + unit_path1)
             << arm_compute::graph::frontend::ActivationLayer(active_info).set_name("relu" + unit_path1)

             << arm_compute::graph::frontend::ConvolutionLayer(
                 1U, 1U, channel * 2,
                 get_weights_accessor(data_path, "conv" + unit_path2 + "_w.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv" + unit_path2)
             << arm_compute::graph::frontend::BatchNormalizationLayer(
                 get_weights_accessor(data_path, "bn" + unit_path2 + "_w.npy"),
                 get_weights_accessor(data_path, "bn" + unit_path2 + "_b.npy"),
                 get_weights_accessor(data_path, "bn" + unit_path2 + "_scale_w.npy"),
                 get_weights_accessor(data_path, "bn" + unit_path2 + "_scale_b.npy"), 0.00002f).set_name("bn" + unit_path2)
             << arm_compute::graph::frontend::ActivationLayer(active_info).set_name("relu" + unit_path2);

    graph_net << arm_compute::graph::frontend::EltwiseLayer(
        std::move(route), std::move(residual), arm_compute::graph::frontend::EltwiseOperation::Add).set_name("add" + unit_path);
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

void run_yolo(cv::Mat &image_input, int *total_bbox_num, int *detection_bbox){
    double start = what_time_is_it_now();
    set_yolo_input(image_input);
    graph_yolo.run();
    //for(int i = 0; i < 3; i++) printf("yolo1 %d %f\n", i, yolo1[i]);
    //for(int i = 0; i < 3; i++) printf("yolo2 %d %f\n", i, yolo2[i]);
    //return;
    get_yolo_result(image_input, total_bbox_num, detection_bbox);
}

void init_yolo(float *input_image_yolo, float *yolo1, float *yolo2){
    const char *mask_str_coarse = "3,4,5";
    const char *anchors = "10,14,  23,27,  37,58,  81,82,  135,169,  344,319";
    const char *mask_str_fine = "0,1,2";
    if(yolo_coarse_layer == 0) yolo_coarse_layer = make_yolo_snpe(YOLO_CHANNEL, COARSE_SIZE, COARSE_SIZE, mask_str_coarse, anchors);
    if(yolo_fine_layer == 0) yolo_fine_layer = make_yolo_snpe(YOLO_CHANNEL, FINE_SIZE, FINE_SIZE, mask_str_fine, anchors);

    std::string data_path = model_path_prefix + "yolov3_tiny_more/";
    const arm_compute::DataLayout weights_layout = arm_compute::DataLayout::NCHW;
    const arm_compute::TensorShape tensor_shape = arm_compute::TensorShape(416U, 416U, 3U, 1U); // DataLayout::NCHW, DataLayout::NCHW);
    arm_compute::graph::TensorDescriptor input_descriptor = arm_compute::graph::TensorDescriptor(
        tensor_shape, arm_compute::DataType::F32).set_layout(weights_layout);

    graph_yolo << graph_target
               << fast_math_hint
               << arm_compute::graph::frontend::InputLayer(
                   input_descriptor, arm_compute::support::cpp14::make_unique<LoadInputData>(input_image_yolo, &yolo_input_load))
               << arm_compute::graph::frontend::ConvolutionLayer(
                   3U, 3U, 32,
                   get_weights_accessor(data_path, "conv1_w.npy"),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   arm_compute::PadStrideInfo(1, 1, 1, 1)).set_name("conv1")

               << arm_compute::graph::frontend::BatchNormalizationLayer(
                   get_weights_accessor(data_path, "bn1_w.npy"),
                   get_weights_accessor(data_path, "bn1_b.npy"),
                   get_weights_accessor(data_path, "bn1_scale_w.npy"),
                   get_weights_accessor(data_path, "bn1_scale_b.npy"),
                   0.00002f).set_name("bn1")
               << arm_compute::graph::frontend::ActivationLayer(
                   arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("relu1");
    add_residual_yolo(data_path, 32, 2, graph_yolo);
    add_residual_yolo(data_path, 64, 5, graph_yolo);
    add_residual_yolo(data_path, 128, 8, graph_yolo);
    add_residual_yolo(data_path, 256, 11, graph_yolo);
    arm_compute::graph::frontend::SubStream upsample_route(graph_yolo);
    add_residual_yolo(data_path, 512, 14, graph_yolo);

    add_convolutional_yolo(data_path, 512, 1, 0, 17, graph_yolo);
    add_convolutional_yolo(data_path, 1024, 3, 1, 18, graph_yolo);
    add_convolutional_yolo(data_path, 512, 1, 0, 19, graph_yolo);
    arm_compute::graph::frontend::SubStream yolo_route(graph_yolo);
    add_convolutional_yolo(data_path, 1024, 3, 1, 20, graph_yolo);

    graph_yolo << arm_compute::graph::frontend::ConvolutionLayer(1U, 1U, 18,
                                                                get_weights_accessor(data_path, std::string("conv21") + "_w.npy"),
                                                                get_weights_accessor(data_path, std::string("conv21") + "_b.npy"),
                                                                arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv21")
               << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(yolo1, &yolo_output_load));

    add_convolutional_yolo_sub(data_path, 256, 1, 0, 22, yolo_route);
    yolo_route << arm_compute::graph::frontend::UpsampleLayer(arm_compute::Size2D(2, 2),
                                                              arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR).set_name("Upsample_29");
    arm_compute::graph::frontend::SubStream concat_1(yolo_route);
    concat_1 << arm_compute::graph::frontend::ConcatLayer(std::move(yolo_route), std::move(upsample_route)).set_name("Route1");
    add_convolutional_yolo_sub(data_path, 256, 1, 0, 23, concat_1);
    add_convolutional_yolo_sub(data_path, 512, 3, 1, 24, concat_1);
    add_convolutional_yolo_sub(data_path, 256, 1, 0, 25, concat_1);
    add_convolutional_yolo_sub(data_path, 512, 3, 1, 26, concat_1);
    concat_1 << arm_compute::graph::frontend::ConvolutionLayer(1U, 1U, 18,
                                                               get_weights_accessor(data_path, std::string("conv27") + "_w.npy"),
                                                               get_weights_accessor(data_path, std::string("conv27") + "_b.npy"),
                                                               arm_compute::PadStrideInfo(1, 1, 0, 0)).set_name("conv27")
             << arm_compute::graph::frontend::OutputLayer(arm_compute::support::cpp14::make_unique<ReadOutputData>(yolo2, &yolo_output1_load));

    arm_compute::graph::GraphConfig config;
    config.num_threads = num_threads;
    config.use_tuner = use_tuner;
    config.tuner_mode = cl_tuner_mode;
    config.tuner_file = model_path_prefix + "acl_tuner_yolov3.csv";

    graph_yolo.finalize(graph_target, config);
    return;
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
        LOGE("side face: %f %f %f\n", two_eye_height_diff, two_lips_height_diff, two_eye_width_diff);
        *side_face = 1;
    } else {
        LOGE("not side face: %f %f %f\n", two_eye_height_diff, two_lips_height_diff, two_eye_width_diff);
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
    std::string data_path = model_path_prefix + "spoofing_binary/";
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

#define clamp_g(x, minValue, maxValue) ((x) < (minValue) ? (minValue) : ((x) > (maxValue) ? (maxValue) : (x)))

JNIEXPORT jintArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_yuv2bitmap_1native(
    JNIEnv *env, jobject obj, jbyteArray image_data, jint width, jint height, jint height_out)
{
    jboolean isCopy;
    uchar *srcYVU = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    unsigned char *srcVU = srcYVU + width * height;
    unsigned char Y, U, V;
    int B, G, R;
    int index = 0;
    jintArray dst_java = env->NewIntArray(height_out * height);
    int *dst = (int *)env->GetIntArrayElements(dst_java, &isCopy);

    int j_offset = width - height_out;
    int j_index = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < height_out; j++) {
            j_index = j_offset + j;
            Y = srcYVU[i * width + j_index];
            V = srcVU[(i / 2 * width / 2 + j_index / 2) * 2 + 0];
            U = srcVU[(i / 2 * width / 2 + j_index / 2) * 2 + 1];
            R = 1.164f*(Y - 16) + 1.596f*(V - 128);
            G = 1.164f*(Y - 16) - 0.813f*(V - 128) - 0.392f*(U - 128);
            B = 1.164f*(Y - 16) + 2.017f*(U - 128);

            B = clamp_g(B, 0, 255);
            G = clamp_g(G, 0, 255);
            R = clamp_g(R, 0, 255);
            dst[i + (height_out - 1 - j) * height] = (R << 16) + (G << 8) + B | 0xFF000000;
        }
    }
    env->ReleaseByteArrayElements(image_data, (jbyte *)srcYVU, 0);
    env->ReleaseIntArrayElements(dst_java, (jint *)dst, 0);

    //cv::Mat img_temp_local(height, width,  CV_8UC3, image_data_point);
    return dst_java;
}

JNIEXPORT jbyteArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_yv122rgb_1native(
    JNIEnv *env, jobject obj, jbyteArray image_data, jint width, jint height)
{
    jboolean isCopy;
    uchar *srcYVU = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    unsigned char *srcV = srcYVU + width * height;
    unsigned char *srcU = srcYVU + width * height + width * height / 2 / 2;
    unsigned char Y, U, V;
    int B, G, R;
    int index = 0;
    jbyteArray dst_java = env->NewByteArray(width * height * 3);
    uchar *dst = (uchar *)env->GetByteArrayElements(dst_java, &isCopy);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Y = srcYVU[i * width + j];
            index = i / 2 * width / 2 + j / 2;
            V = srcV[index];
            U = srcU[index];
            R = Y + 1.370705f*(V - 128);
            G = Y - 0.698001f*(U - 128) - 0.703125f*(V - 128);
            B = Y + 1.732446f*(U - 128);

            index = (i + (width - 1 - j) * height) * 3;
            dst[index] = clamp_g(B, 0, 255);
            dst[index + 1] = clamp_g(G, 0, 255);
            dst[index + 2] = clamp_g(R, 0, 255);
        }
    }
    env->ReleaseByteArrayElements(image_data, (jbyte *)srcYVU, 0);
    env->ReleaseByteArrayElements(dst_java, (jbyte *)dst, 0);
    return dst_java;
}

JNIEXPORT jbyteArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_yuv2rgb_1native(
    JNIEnv *env, jobject obj, jbyteArray image_data, jint width, jint height)
{
    jboolean isCopy;
    uchar *srcYVU = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    unsigned char *srcVU = srcYVU + width * height;
    unsigned char Y, U, V;
    int B, G, R;
    int index = 0;
    jbyteArray dst_java = env->NewByteArray(width * height * 3);
    uchar *dst = (uchar *)env->GetByteArrayElements(dst_java, &isCopy);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Y = srcYVU[i * width + j];
            V = srcVU[(i / 2 * width / 2 + j / 2) * 2 + 0];
            U = srcVU[(i / 2 * width / 2 + j / 2) * 2 + 1];
            R = 1.164f*(Y - 16) + 1.596f*(V - 128);
            G = 1.164f*(Y - 16) - 0.813f*(V - 128) - 0.392f*(U - 128);
            B = 1.164f*(Y - 16) + 2.017f*(U - 128);

            //index = (i + (width - 1 - j) * height) * 3;
            index = (j + i * width) * 3;
            dst[index] = clamp_g(B, 0, 255);
            dst[index + 1] = clamp_g(G, 0, 255);
            dst[index + 2] = clamp_g(R, 0, 255);
        }
    }
    env->ReleaseByteArrayElements(image_data, (jbyte *)srcYVU, 0);
    env->ReleaseByteArrayElements(dst_java, (jbyte *)dst, 0);
    return dst_java;
}

JNIEXPORT jbyteArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_bitmap2rgb_1native(
    JNIEnv *env, jobject obj, jintArray image_data) {
    jboolean isCopy;
    int *src = (int *)env->GetIntArrayElements(image_data, &isCopy);
    int length = env->GetArrayLength(image_data);
    jbyteArray dst_java = env->NewByteArray(length * 3);
    uchar *dst = (uchar *)env->GetByteArrayElements(dst_java, &isCopy);
    for(int i = 0; i < length; i++){
        dst[3 * i] = src[i] & 0xFF;
        dst[3 * i + 1] = src[i] >> 8 & 0xFF;
        dst[3 * i + 2] = src[i] >> 16 & 0xFF;
    }
    env->ReleaseIntArrayElements(image_data, (int *)src, 0);
    env->ReleaseByteArrayElements(dst_java, (jbyte *)dst, 0);
    return dst_java;
}

JNIEXPORT jintArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_rgb2bitmap_1native(
    JNIEnv *env, jobject obj, jbyteArray image_data)
{
    jboolean isCopy;
    uchar *src = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    int length = env->GetArrayLength(image_data);
    length = length / 3;
    jintArray dst_java = env->NewIntArray(length);
    jint *dst = (jint *)env->GetIntArrayElements(dst_java, &isCopy);
    for(int i = 0; i < length; i++){
        dst[i] = ((int)src[3 * i + 2] << 16) + ((int)src[3 * i + 1] << 8) + src[3 * i] | 0xFF000000;
    }
    env->ReleaseByteArrayElements(image_data, (jbyte *)src, 0);
    env->ReleaseIntArrayElements(dst_java, dst, 0);
    return dst_java;
}

JNIEXPORT jint JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_detect_1face(
    JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jint width, jint height){
    if(!have_init_opencv_detect){
        LOGE("init_opencv_detect!");
        std::string model_path = model_path_prefix + "haarcascade_frontalface_alt.xml";
        faces_cascade.load(model_path.c_str());
        have_init_opencv_detect = true;
    }

    double start = what_time_is_it_now();
    jboolean isCopy;    // JNI_TRUE表示原字符串的拷贝，返回JNI_FALSE表示返回原字符串的指针
    if(NULL == image_data) {
        return 0;
    }
    uchar *image_data_point = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    jsize image_data_size = env->GetArrayLength(image_data);
    if (image_data_point == NULL || image_data_size < 1000) {
        LOGE("features extraction: image_data is empty!");
        env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
        return 0;
    }
    cv::Mat img_temp(height, width,  CV_8UC3, image_data_point);
    cv::Mat grayImage;
    //cv::imwrite("/sdcard/A/color.jpg", img_temp);
    cvtColor(img_temp, grayImage, CV_BGR2GRAY);
    equalizeHist(grayImage, grayImage);
    std::vector<cv::Rect> faces;
    faces_cascade.detectMultiScale(grayImage, faces, 1.1, 3, 0, cv::Size(256, 256));
    env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
    if(faces.size() < 1) {
        return 0;
    } else {
        int index = -1;
        int max_area = -1;
        for(int i = 0; i < faces.size(); i++) {
            if(faces[i].width > 256 && faces[i].height > 256 && faces[i].area() > max_area){
                max_area = faces[i].area();
                index = i;
            }
        }
        if(index < 0) return 0;
        int detection_bbox[4];
        detection_bbox[0] = faces[index].x;
        detection_bbox[1] = faces[index].y;
        detection_bbox[2] = faces[index].x + faces[index].width;
        detection_bbox[3] = faces[index].y + faces[index].height;
        env->SetIntArrayRegion(face_region, 0, 4, detection_bbox);
        return 1;
    }
}


cv::Rect get_face_rect(cv::Mat &img_temp, jint* face_region_int){
    int face_width = face_region_int[2] - face_region_int[0];
    int face_height = face_region_int[3] - face_region_int[1];

    int x_start = face_region_int[0];
    if(face_region_int[0] - face_width / 2.0 >= 0) x_start = face_region_int[0] - face_width / 2.0;
    else if(face_region_int[0] - face_width / 4.0 >= 0) x_start = face_region_int[0] - face_width / 4.0;
    else if(face_region_int[0] - face_width / 8.0 >= 0) x_start = face_region_int[0] - face_width / 8.0;
    int y_start = face_region_int[1];
    if(face_region_int[1] - face_height / 2.0 >= 0) y_start = face_region_int[1] - face_height / 2.0;
    else if(face_region_int[1] - face_height / 4.0 >= 0) y_start = face_region_int[1] - face_height / 4.0;
    else if(face_region_int[1] - face_height / 8.0 >= 0) y_start = face_region_int[1] - face_height / 8.0;

    if(x_start + 2 * face_width < img_temp.cols - 1) face_width = 2 * face_width;
    else if(x_start + 1.5 * face_width < img_temp.cols - 1) face_width = 1.5 * face_width;
    else if(x_start + 1.3 * face_width < img_temp.cols - 1) face_width = 1.3 * face_width;
    else face_width = img_temp.cols - 1 - x_start;
    if(y_start + 2 * face_height < img_temp.rows - 1) face_height = 2 * face_height;
    else if(y_start + 1.5 * face_height < img_temp.rows - 1) face_height = 1.5 * face_height;
    else if(y_start + 1.3 * face_height < img_temp.rows - 1) face_height = 1.3 * face_height;
    else face_height = img_temp.rows - 1 - y_start;

    cv::Rect r(x_start, y_start, face_width, face_height);
    return r;
}

JNIEXPORT void JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_save_1spoofingimage(
    JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jint width, jint height, jint is_rgb, jint is_real){
    if(NULL == image_data) {
        return;
    }
    jboolean isCopy;
    uchar *image_data_point = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    jsize image_data_size = env->GetArrayLength(image_data);
    if (image_data_point == NULL || image_data_size < 1000) {
        LOGE("image_data is empty %d !", image_data_size);
        env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
        return;
    }
    cv::Mat img_temp(height, width,  CV_8UC3, image_data_point);

    jint* face_region_int = env->GetIntArrayElements(face_region, &isCopy);
    cv::Rect r = get_face_rect(img_temp, face_region_int);
    if(0 > r.x || 0 > r.width || r.x + r.width >= img_temp.cols || 0 > r.y || 0 > r.height || r.y + r.height >= img_temp.rows){
        for(int i = 1; i < 4; ++i) LOGE("face_region %d\n", face_region_int[i]);
        LOGE("rect 1 size error %d %d %d %d, image size width %d height %d", r.x, r.y, r.width, r.height, img_temp.cols, img_temp.rows);
        env->ReleaseIntArrayElements(face_region, face_region_int, 0);
        env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
        return;
    }
    env->ReleaseIntArrayElements(face_region, face_region_int, 0);
    
    static int index = 0;
    double start = what_time_is_it_now();
    long long timestamp = (long long)start;
    std::string image_path;
    if(is_rgb == 1){
        image_path = model_path_prefix + "spoofing/" +
            std::to_string(is_real) + "_rgb_" + std::to_string(timestamp) + "_" + std::to_string(index) + ".jpg";
    } else {
        image_path = model_path_prefix + "spoofing/" +
            std::to_string(is_real) + "_infrared_" + std::to_string(timestamp) + "_" + std::to_string(index) + ".jpg";
    }
    cv::Mat patch;
    cv::resize(img_temp(r), patch, cv::Size(96, 112));
    env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
    imwrite(image_path, patch);
    LOGE("image_path %s", image_path.c_str());
    index += 1;
    return;
}

JNIEXPORT int JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_run_1spoofing(
    JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jint width, jint height){
    if(NULL == image_data) {
        return 0;
    }
    jboolean isCopy;
    uchar *image_data_point = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    jsize image_data_size = env->GetArrayLength(image_data);
    if (image_data_point == NULL || image_data_size < 1000) {
        LOGE("image_data is empty %d !", image_data_size);
        env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
        return 0;
    }
    cv::Mat img_temp(height, width,  CV_8UC3, image_data_point);

    jint* face_region_int = env->GetIntArrayElements(face_region, &isCopy);
    cv::Rect r = get_face_rect(img_temp, face_region_int);
    if(0 > r.x || 0 > r.width || r.x + r.width >= img_temp.cols || 0 > r.y || 0 > r.height || r.y + r.height >= img_temp.rows){
        for(int i = 1; i < 4; ++i) LOGE("face_region %d\n", face_region_int[i]);
        LOGE("rect 1 size error %d %d %d %d, image size width %d height %d", r.x, r.y, r.width, r.height, img_temp.cols, img_temp.rows);
        env->ReleaseIntArrayElements(face_region, face_region_int, 0);
        env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
        return 0;
    }
    env->ReleaseIntArrayElements(face_region, face_region_int, 0);

    double start = what_time_is_it_now();
    cv::Mat patch;
    cv::resize(img_temp(r), patch, cv::Size(96, 112));
    env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
    run_spoofing(patch);
    if(spoofing_result[0] >= spoofing_result[1]) return 0;
    else return 1;
}

JNIEXPORT jbyteArray JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_face_1region(
    JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jint width, jint height){
    jboolean isCopy;
    jbyteArray return_null_image = env->NewByteArray(0);
    if(NULL == image_data) {
        return return_null_image;
    }
    uchar *image_data_point = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    jsize image_data_size = env->GetArrayLength(image_data);
    if (image_data_point == NULL || image_data_size < 1000) {
        LOGE("features extraction: image_data is empty!");
        env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
        return return_null_image;
    }
    cv::Mat img_temp;
    if(width == 0 && height == 0){
        std::vector<uchar> image_raw_data(image_data_size);
        std::copy(image_data_point, image_data_point + image_data_size, image_raw_data.data());
        img_temp = cv::imdecode(image_raw_data, CV_LOAD_IMAGE_COLOR);
    } else {
        cv::Mat img_local(height, width,  CV_8UC3, image_data_point);
        img_temp = img_local;
    }
    env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
    jint* face_region_point = env->GetIntArrayElements(face_region, &isCopy);
    cv::Rect r = get_face_rect(img_temp, face_region_point);
    if(0 > r.x || 0 > r.width || r.x + r.width >= img_temp.cols || 0 > r.y || 0 > r.height || r.y + r.height >= img_temp.rows){
        for(int i = 1; i < 4; ++i) LOGE("face_region %d\n", face_region_point[i]);
        LOGE("rect 1 size error %d %d %d %d, image size width %d height %d", r.x, r.y, r.width, r.height, img_temp.cols, img_temp.rows);
        env->ReleaseIntArrayElements(face_region, face_region_point, 0);
        return return_null_image;
    }
    env->DeleteLocalRef(return_null_image);
    env->ReleaseIntArrayElements(face_region, face_region_point, 0);

    std::vector<uchar> data_encode;
    cv::imencode(".jpg", (img_temp)(r), data_encode);
    jbyte *data_encode_jbyte = (jbyte *)malloc(data_encode.size() * sizeof(jbyte));
    for(unsigned int i = 0; i < data_encode.size(); i++) {
        data_encode_jbyte[i] = data_encode[i];
    }
    jbyteArray return_byte_array = env->NewByteArray(data_encode.size());
    env->SetByteArrayRegion(return_byte_array, 0, data_encode.size(), data_encode_jbyte);
    free(data_encode_jbyte);
    return return_byte_array;
}

JNIEXPORT jint JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_recognition_1face(
    JNIEnv *env, jobject obj, jbyteArray image_data, jintArray face_region, jfloatArray feature_save,
    jlongArray code_ret, jint width, jint height, jintArray is_side_face){
    std::lock_guard<std::mutex> gpu_lock_guard(gpu_lock, std::adopt_lock);
    int code = 1000;
    double start = what_time_is_it_now();
    jboolean isCopy;    // JNI_TRUE表示原字符串的拷贝，返回JNI_FALSE表示返回原字符串的指针
    jlong* code_point = env->GetLongArrayElements(code_ret, &isCopy);
    if(NULL == image_data || !have_init) {
        code = 1005;
        if(!have_init) LOGE("features extraction: has not initial network!");
        if(NULL == image_data) LOGE("features extraction: image_data is empty!");
        code_point[0] = code;
        env->ReleaseLongArrayElements(code_ret, code_point, 0);
        return 0;
    }
    uchar *image_data_point = (uchar *)env->GetByteArrayElements(image_data, &isCopy);
    jsize image_data_size = env->GetArrayLength(image_data);
    if (image_data_point == NULL || image_data_size < 1000) {
        LOGE("features extraction: image_data is empty!");
        code_point[0] = 1005;
        env->ReleaseLongArrayElements(code_ret, code_point, 0);
        env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
        return 0;
    }
    cv::Mat img_temp;
    if(width == 0 && height == 0){
        std::vector<uchar> image_raw_data(image_data_size);
        std::copy(image_data_point, image_data_point + image_data_size, image_raw_data.data());
        img_temp = cv::imdecode(image_raw_data, CV_LOAD_IMAGE_COLOR);
    } else {
        cv::Mat img_local(height, width,  CV_8UC3, image_data_point);
        img_temp = img_local;
    }

    /*
    static int index = 0;
    cv::imwrite(model_path_prefix + std::to_string(index) + ".jpg", img_temp);
    index += 1;
    */

    /* 1001: blur, 1002: multiple face, 1003: image too small, 1004: image too large, 1005: image empty
       1006: none face, 1007: save aligned-image failed, 1008: save feature failed, 1009: Face is dark
       1010: Face illuminaiton is unbalance, 1011: Image is side face, 1012: Face is noisy */
    int face_count = 0;
    int detection_bbox[MAX_BBOX_NUM * 4];

    int side_face = 0;
    get_image_feature(img_temp, &face_count, detection_bbox, &side_face);
    env->ReleaseByteArrayElements(image_data, (jbyte *)image_data_point, 0);
    if(face_count == 0){
        code_point[0] = 1006;
        env->ReleaseLongArrayElements(code_ret, code_point, 0);
        LOGE("face detected thread id %lu, spend: %f, face_count: %d input image %dx%d",
             std::hash<std::thread::id>{}(std::this_thread::get_id()), what_time_is_it_now() - start, face_count, width, height);
        return 0;
    }

    /*
    static int index = 0;
    cv::Mat img_temp_clone = img_temp.clone();
    cv::rectangle(img_temp_clone, cvPoint(detection_bbox[0], detection_bbox[1]),
                  cvPoint(detection_bbox[2], detection_bbox[3]), cvScalar(255,0,0), 4);
    cv::imwrite(model_path_prefix + std::to_string(index) + ".jpg", img_temp_clone);
    index += 1;
    */

    jint* side_face_point = env->GetIntArrayElements(is_side_face, &isCopy);
    side_face_point[0] = side_face;
    env->ReleaseIntArrayElements(is_side_face, side_face_point, 0);
    env->SetFloatArrayRegion(feature_save, 0, FEATURE_LENGTH, face_feature);
    env->SetIntArrayRegion(face_region, 0, 4, detection_bbox);
    code_point[0] = code;
    env->ReleaseLongArrayElements(code_ret, code_point, 0);
    LOGE("face detected thread id %lu, spend: %f, face_count: %d, input image %dx%d, face region %d %d %d %d, side_face %d",
         std::hash<std::thread::id>{}(std::this_thread::get_id()), what_time_is_it_now() - start, face_count,
         width, height, detection_bbox[0], detection_bbox[1], detection_bbox[2], detection_bbox[3], side_face);
    return face_count;
}

void init_network_cnn(jint use_spoofing){
    init_arcface(face_feature, face_image_input);
    LOGE("init_arcface over");
    init_landmark(input_landmark, output_landmark);
    LOGE("init_landmark over");
    //init_yolo(input_image_yolo, yolo1, yolo2);
    init_yolo_tiny(input_image_yolo, yolo1, yolo2);
    LOGE("init_yolo_tiny over");
    if(use_spoofing == 1){
        init_spoofing();
        LOGE("init_spoofing over");
    }
    have_init = true;
}

JNIEXPORT jboolean JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_recognition_1start(
    JNIEnv *env, jobject obj, jstring model_path_java, jint use_spoofing){
    std::lock_guard<std::mutex> gpu_lock_guard(gpu_lock, std::adopt_lock);
    if(have_init) return false;
    jboolean isCopy;
    const char* model_path = env->GetStringUTFChars(model_path_java, &isCopy);
    if(model_path == NULL) {
        LOGE("model path is empty! exit");
        exit(-1);
    }
    LOGE("model path is : %s", model_path);
    std::string model_path_string(model_path);
    env->ReleaseStringUTFChars(model_path_java, model_path);
    std::string data_path = model_path_string + "face34_glint_refine/fc1_w.npy";
    if(access(data_path.c_str(), 0)){
        LOGE("model path is not exist %s", model_path);
        exit(-1);
    }
    model_path_prefix = model_path_string;
    init_network_cnn(use_spoofing);
    return true;
}

void uninit_network_cnn(){
    free(fc1_w);
    free(fc1_b);
    free(fc1_scale_w);
    free(fc1_scale_b);
}

JNIEXPORT jboolean JNICALL Java_com_iim_recognition_caffe_LoadLibraryModule_recognition_1stop(JNIEnv *env, jobject obj){
    if(!have_init) return false;
    uninit_network_cnn();
    have_init = false;
    return true;
}

int main_sdf(int argc, char **argv)
{
    init_network_cnn(0);
    cv::Mat image_input = cv::imread("1.jpg");
    int detection_bbox[MAX_BBOX_NUM * 4];
    int face_count = 0;
    for(int i = 0; i < 1; i++) {
        int side_face = 0;
        get_image_feature(image_input, &face_count, detection_bbox, &side_face);
        printf("%d times\n\n", i);
        
        for(int j = 0; j < FEATURE_LENGTH; j++) face_feature[j] = 0;
        for(int j = 0; j < 112 * 112 *3; j++) face_image_input[j] = 0;
        for(int j = 0; j < 416 * 416 * 3; j++) input_image_yolo[j] = 0;
        for(int j = 0; j < COARSE_SIZE * COARSE_SIZE * YOLO_CHANNEL; j++) yolo1[j] = 0;
        for(int j = 0; j < FINE_SIZE * FINE_SIZE * YOLO_CHANNEL; j++) yolo2[j] = 0;
        for(int j = 0; j < 48 * 48 *3; j++) input_landmark[j] = 0;
        for(int j = 0; j < 10; j++) output_landmark[j] = 0;
        
    }

    free(fc1_w);
    free(fc1_b);
    free(fc1_scale_w);
    free(fc1_scale_b);
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
