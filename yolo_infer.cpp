// 20260119
// 使用TensorRT推理YOLO模型，目前只做了静态尺寸1 * 3 * 640 * 640的推理(Batch = 1)
// programmer: xlxlqqq

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>

#include <chrono>

using namespace nvinfer1;

// 检测数据类（label， 置信度， 位置）
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

/// <summary>
/// 计算张量元素的总数，以便后续用来申请显存
/// </summary>
/// <param name="d"></param>
/// <returns></returns>
inline int volume(const nvinfer1::Dims& d) {
    int v = 1;
    for (int i = 0; i < d.nbDims; i++) v *= d.d[i];
    return v;
}

// 读取 engine 文件
std::vector<char> readEngineFile(const std::string& engine_file) {
    std::ifstream file(engine_file, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engine_file << std::endl;
        exit(-1);
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    return engine_data;
}

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // 只打印 warning 或更严重的信息
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class TensorRTInfer {
private:
    int batch_;
    int channel_;
    int H_;
    int W_;

    int inputIndex_ = -1;
    int outputIndex_ = -1;
    
    std::string engine_file_;
    std::string image_file_;
    Logger gLogger_;  // createInferRuntime 必须要求传入一个 nvinfer1::ILogger 对象
    
    std::vector<char> engine_data_;
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;

    std::vector<void*> buffers_;
    std::vector<int> sizes_;
    cudaStream_t stream_;

    std::vector<float> input_data_;

    // 统计推理时间，由于GPU属于异步推理，需要用 cudaEvent 记录时间
    cudaEvent_t h2d_start_, h2d_end_;
    cudaEvent_t infer_start_, infer_end_;
    cudaEvent_t d2h_start_, d2h_end_;

    /// <summary>
    /// 准备数据，包括图像预处理、数据拷贝等
    /// </summary>
    void prepareData() {
        cv::Mat image = cv::imread(image_file_);
        if (image.empty()) {
            std::cout << "Error read image file failed." << std::endl;
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        cv::resize(image, image, cv::Size(W_, H_));
        image.convertTo(image, CV_32FC3, 1.0 / 255.0);

        // HWC -> CHW
        input_data_.resize(channel_ * H_ * W_);
        for (int c = 0; c < channel_; c++) {
            for (int h = 0; h < H_; h++) {
                for (int w = 0; w < W_; w++) {
                    input_data_[c * H_ * W_ + h * W_ + w] = image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        // 拷贝CPU里面的输入张量到GPU
        cudaMemcpyAsync(buffers_[inputIndex_], input_data_.data(), input_data_.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream_);
    }


public:
    TensorRTInfer(const std::string& engine_file, const std::string& image_file, int batch_size = 1) :
        engine_file_(engine_file), image_file_(image_file), batch_(batch_size) { };

    ~TensorRTInfer() {
        cudaEventDestroy(h2d_start_);
        cudaEventDestroy(h2d_end_);
        cudaEventDestroy(infer_start_);
        cudaEventDestroy(infer_end_);
        cudaEventDestroy(d2h_start_);
        cudaEventDestroy(d2h_end_);

        if (!buffers_.empty()) {
            for (auto b : buffers_) if (b) cudaFree(b);
        }
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
    }

    void SetImageFile(std::string image_file) {
        image_file_ = image_file;
    }

    std::string GetImageFile() {
        return image_file_;
    }

    void SetEngineFile(std::string engine_file) {
        engine_file_ = engine_file;
    }

    std::string GetEngineFile() {
        return engine_file_;
    }

    // 获取输出维度
    std::vector<int> getOutDims() {
        Dims outDims = engine_->getBindingDimensions(outputIndex_);
        return std::vector<int> {outDims.d[0], outDims.d[1], outDims.d[2]};
    }

    // 获取输入维度
    std::vector<int> getInputDims() {
        Dims inputDims = engine_->getBindingDimensions(inputIndex_);
        return std::vector<int> {inputDims.d[0], inputDims.d[1], inputDims.d[2], inputDims.d[3]};
    }

    /// <summary>
    /// infer 初始化
    /// </summary>
    void initInfer() {
        engine_data_ = readEngineFile(engine_file_);
        runtime_ = createInferRuntime(gLogger_);
        engine_ = runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size());
        context_ = engine_->createExecutionContext();

        if (!runtime_ || !engine_ || !context_) {
            throw std::runtime_error("TensorRT init failed");
        }

        // 查找 input / output binding index哪个绑定的是输入，哪个绑定的是输出
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            if (engine_->bindingIsInput(i)) {
                inputIndex_ = i;
            }
            else {
                outputIndex_ = i;
            }
        }

        if (inputIndex_ == -1 || outputIndex_ == -1) {
            throw std::runtime_error("Failed to find input or output binding index");
        }

        auto inputDims = engine_->getBindingDimensions(0);
        std::cout << "input Dims" << std::endl;
        std::cout << inputDims.d[0] << " " << inputDims.d[1] << " " << inputDims.d[2] << " " << inputDims.d[3] << std::endl;
        batch_ = inputDims.d[0];
        channel_ = inputDims.d[1];
        H_ = inputDims.d[2];
        W_ = inputDims.d[3];

        // 申请显存
        int nbBindings = engine_->getNbBindings();
        buffers_ = std::vector<void*>(nbBindings);
        sizes_ = std::vector<int>(nbBindings);

        for (int i = 0; i < nbBindings; i++) {
            Dims dims = engine_->getBindingDimensions(i);
            sizes_[i] = volume(dims);
            cudaMalloc(&buffers_[i], sizes_[i] * sizeof(float));
        }
        cudaStreamCreate(&stream_);

        cudaEventCreate(&h2d_start_);
        cudaEventCreate(&h2d_end_);
        cudaEventCreate(&infer_start_);
        cudaEventCreate(&infer_end_);
        cudaEventCreate(&d2h_start_);
        cudaEventCreate(&d2h_end_);
    }

    /// <summary>
    /// 外部推理接口
    /// </summary>
    /// <returns></returns>
    std::vector<float> infer() {
        cudaEventRecord(h2d_start_, stream_);
        prepareData();   // 内部有 cudaMemcpyAsync
        cudaEventRecord(h2d_end_, stream_);

        cudaEventRecord(infer_start_, stream_);
        bool status = context_->enqueueV2(buffers_.data(), stream_, nullptr);
        cudaEventRecord(infer_end_, stream_);
        if (!status) {
            std::cout << "infer failed!!!" << std::endl;
            return std::vector<float>();
        }
        Dims out_dims = engine_->getBindingDimensions(1);
        std::vector<float> output(volume(out_dims));

        cudaEventRecord(d2h_start_, stream_);
        // 拷贝GPU里的推理结果到CPU
        cudaMemcpyAsync(output.data(), buffers_[outputIndex_],
            output.size() * sizeof(float),
            cudaMemcpyDeviceToHost, stream_);
        cudaEventRecord(d2h_end_, stream_);

        cudaStreamSynchronize(stream_);
        
        float h2d_ms = 0.f, infer_ms = 0.f, d2h_ms = 0.f;
        cudaEventElapsedTime(&h2d_ms, h2d_start_, h2d_end_);
        cudaEventElapsedTime(&infer_ms, infer_start_, infer_end_);
        cudaEventElapsedTime(&d2h_ms, d2h_start_, d2h_end_);

        std::cout << "H2D time   : " << h2d_ms << " ms" << std::endl;
        std::cout << "Infer time : " << infer_ms << " ms" << std::endl;
        std::cout << "D2H time   : " << d2h_ms << " ms" << std::endl;
        std::cout << "Total GPU  : " << (h2d_ms + infer_ms + d2h_ms) << " ms" << std::endl;

        return output;
    }

};

/// <summary>
/// 解析推理结果，带NMS非极大抑制
/// </summary>
/// <param name="inferResult"></param>
/// <param name="out_dims"></param>
/// <param name="conf_thres"></param>
/// <param name="nms_thres"></param>
/// <returns></returns>
std::vector<Detection> decode(const std::vector<float>& inferResult, 
    const std::vector<int>& out_dims, float conf_thres = 0.25f, float nms_thres = 0.4f) {
    int batch = out_dims[0];
    int num_boxes = out_dims[1];  // 每张预测框的数量
    int num_classes = out_dims[2] - 5;  // 4(mmbox)+1(obj_conf)+num_classes

    std::vector<Detection> detections;
    //detections.resize(batch);

    for (int i = 0; i < num_boxes; i++) {
        const float* detection = inferResult.data() + i * out_dims[2];
        float confidence = detection[4];
        if (confidence < conf_thres) {
            continue;
        }

        float max_conf = 0.0f;
        int class_id = -1;
        for (int c = 0; c < num_classes; ++c)
        {
            if (detection[5 + c] > max_conf)
            {
                max_conf = detection[5 + c];
                class_id = c;
            }
        }

        float conf = confidence * max_conf;
        if (conf < conf_thres) continue;

        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];

        int x = static_cast<int>(cx - w / 2.0f);
        int y = static_cast<int>(cy - h / 2.0f);
        int width = static_cast<int>(w);
        int height = static_cast<int>(h);

        Detection d;
        d.class_id = class_id;
        d.confidence = conf;
        d.box = cv::Rect(x, y, width, height);

        detections.push_back(d);
    }

    // NMS 非极大抑制
    std::vector<Detection> nms_result;

    for (int cls = 0; cls < num_classes; ++cls) {

        // 取出当前类别的框
        std::vector<Detection> cls_dets;
        for (const auto& d : detections) {
            if (d.class_id == cls)
                cls_dets.push_back(d);
        }

        if (cls_dets.empty()) continue;

        // 按置信度排序
        std::sort(cls_dets.begin(), cls_dets.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
            });

        std::vector<bool> removed(cls_dets.size(), false);

        for (size_t i = 0; i < cls_dets.size(); ++i) {
            if (removed[i]) continue;

            nms_result.push_back(cls_dets[i]);
            const cv::Rect& a = cls_dets[i].box;

            for (size_t j = i + 1; j < cls_dets.size(); ++j) {
                if (removed[j]) continue;

                const cv::Rect& b = cls_dets[j].box;

                float inter_area = (a & b).area();
                float union_area = a.area() + b.area() - inter_area;
                float iou = inter_area / union_area;

                if (iou > nms_thres) {
                    removed[j] = true;
                }
            }
        }
    }

    return nms_result;
}

int main() {
    std::string engine_file = "./yolo5/model/best_gpu.engine";
    std::string image_file = "./yolo5/image/0.jpg";
    
    TensorRTInfer infer(engine_file, image_file);
    infer.initInfer();
    std::vector<float> output = infer.infer();

    auto detections = decode(output, infer.getOutDims(), 0.25f);

    for (const auto& d : detections) {
        std::cout << d.class_id << " " << d.confidence << " " << d.box.x << " " << d.box.y << " " << d.box.width << " " << d.box.height << std::endl;
    }

    // TODO: 将检测框位置从640 * 640转化到原始图像尺寸，并输出给用户
    
    system("pause");
    return 0;
}


