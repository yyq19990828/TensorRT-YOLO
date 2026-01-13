# TensorRT-YOLO C++ API Reference

> **Version:** 6.4.0 | **Header:** `#include "trtyolo.hpp"` | **Namespace:** `trtyolo`

本文档详细介绍 TensorRT-YOLO 的 C++ API，包括所有公共类、结构体、方法及其使用方式。

---

## 目录

- [快速入门](#快速入门)
- [命名空间](#命名空间)
- [数据结构](#数据结构)
  - [Image](#image)
  - [Mask](#mask)
  - [KeyPoint](#keypoint)
  - [Box](#box)
  - [RotatedBox](#rotatedbox)
- [结果结构体](#结果结构体)
  - [BaseRes](#baseres)
  - [ClassifyRes](#classifyres)
  - [DetectRes](#detectres)
  - [OBBRes](#obbres)
  - [SegmentRes](#segmentres)
  - [PoseRes](#poseres)
- [配置类](#配置类)
  - [InferOption](#inferoption)
- [模型类](#模型类)
  - [BaseModel](#basemodel)
  - [ClassifyModel](#classifymodel)
  - [DetectModel](#detectmodel)
  - [OBBModel](#obbmodel)
  - [SegmentModel](#segmentmodel)
  - [PoseModel](#posemodel)
- [使用示例](#使用示例)
- [线程安全](#线程安全)
- [错误处理](#错误处理)

---

## 快速入门

```cpp
#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

int main() {
    // 1. 配置推理选项
    trtyolo::InferOption option;
    option.enableSwapRB();           // BGR→RGB 转换
    option.enablePerformanceReport(); // 启用性能统计

    // 2. 创建检测模型
    auto detector = std::make_unique<trtyolo::DetectModel>(
        "yolo11n.engine", option
    );

    // 3. 加载图像
    cv::Mat cv_image = cv::imread("test.jpg");
    trtyolo::Image image(cv_image.data, cv_image.cols, cv_image.rows);

    // 4. 执行推理
    trtyolo::DetectRes result = detector->predict(image);

    // 5. 处理结果
    for (int i = 0; i < result.num; ++i) {
        std::cout << "Class: " << result.classes[i]
                  << ", Score: " << result.scores[i]
                  << ", Box: " << result.boxes[i] << std::endl;
    }

    return 0;
}
```

---

## 命名空间

所有 API 都在 `trtyolo` 命名空间下：

```cpp
namespace trtyolo {
    // 所有类和结构体
}
```

---

## 数据结构

### Image

图像数据封装结构，用于将外部图像数据传递给推理引擎。

```cpp
struct TRTYOLOAPI Image {
    void*  ptr;           // 图像数据指针
    int    width    = 0;  // 图像宽度（像素）
    int    height   = 0;  // 图像高度（像素）
    int    channels = 0;  // 通道数（默认3）
    size_t pitch    = 0;  // 行字节数（包括 padding）

    // 构造函数
    Image(void* data, int width, int height);
    Image(void* data, int width, int height, size_t pitch);
    Image(void* data, int width, int height, int channels, size_t pitch);
};
```

#### 构造函数

| 构造函数 | 描述 |
|---------|------|
| `Image(void* data, int width, int height)` | 紧密排列图像（无 padding），默认 3 通道 |
| `Image(void* data, int width, int height, size_t pitch)` | 带行对齐的图像 |
| `Image(void* data, int width, int height, int channels, size_t pitch)` | 完整参数构造 |

#### 使用示例

```cpp
// 从 OpenCV Mat 创建
cv::Mat mat = cv::imread("image.jpg");
trtyolo::Image img1(mat.data, mat.cols, mat.rows);

// 带 pitch 的图像（例如来自视频解码器）
trtyolo::Image img2(buffer, 1920, 1080, stride);

// 指定通道数和 pitch
trtyolo::Image img3(rgba_buffer, 1920, 1080, 4, stride);
```

> **注意**: `Image` 不复制数据，仅保存指针引用。确保在推理期间数据有效。

---

### Mask

分割掩码数据结构。

```cpp
struct TRTYOLOAPI Mask {
    std::vector<float> data;   // 掩码数据（0.0-1.0）
    int width  = 0;            // 掩码宽度
    int height = 0;            // 掩码高度

    Mask(int width, int height);
};
```

#### 成员说明

| 成员 | 类型 | 描述 |
|-----|------|------|
| `data` | `std::vector<float>` | 掩码像素值，范围 [0.0, 1.0] |
| `width` | `int` | 掩码宽度（与检测框宽度对应） |
| `height` | `int` | 掩码高度（与检测框高度对应） |

---

### KeyPoint

关键点数据结构，用于姿态估计。

```cpp
struct TRTYOLOAPI KeyPoint {
    float x;                      // X 坐标
    float y;                      // Y 坐标
    std::optional<float> conf;    // 置信度（可选）

    KeyPoint(float x, float y, std::optional<float> conf = std::nullopt);
};
```

#### 成员说明

| 成员 | 类型 | 描述 |
|-----|------|------|
| `x` | `float` | 关键点 X 坐标（像素） |
| `y` | `float` | 关键点 Y 坐标（像素） |
| `conf` | `std::optional<float>` | 关键点置信度，若模型不输出则为空 |

---

### Box

轴对齐矩形框（AABB）。

```cpp
struct TRTYOLOAPI Box {
    float left;    // 左边界 X 坐标
    float top;     // 上边界 Y 坐标
    float right;   // 右边界 X 坐标
    float bottom;  // 下边界 Y 坐标

    Box(float left, float top, float right, float bottom);

    // 返回整数坐标 [x1, y1, x2, y2]
    std::array<int, 4> xyxy() const;
};
```

#### 方法

| 方法 | 返回类型 | 描述 |
|-----|---------|------|
| `xyxy()` | `std::array<int, 4>` | 返回四舍五入后的整数坐标 `{left, top, right, bottom}` |

---

### RotatedBox

旋转矩形框（OBB），继承自 `Box`。

```cpp
struct TRTYOLOAPI RotatedBox : public Box {
    float theta;  // 旋转角度（弧度），顺时针为正

    RotatedBox(float left, float top, float right, float bottom, float theta);

    // 返回四个顶点坐标 [x1,y1, x2,y2, x3,y3, x4,y4]
    std::array<int, 8> xyxyxyxy() const;
};
```

#### 方法

| 方法 | 返回类型 | 描述 |
|-----|---------|------|
| `xyxyxyxy()` | `std::array<int, 8>` | 返回四个顶点的整数坐标 |

---

## 结果结构体

### BaseRes

所有结果类型的基类。

```cpp
struct TRTYOLOAPI BaseRes {
    int                num = 0;   // 检测数量
    std::vector<int>   classes;   // 类别 ID 列表
    std::vector<float> scores;    // 置信度列表

    BaseRes() = default;
    BaseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores);
};
```

---

### ClassifyRes

图像分类结果。

```cpp
struct TRTYOLOAPI ClassifyRes : public BaseRes {
    // 继承 num, classes, scores
    // classes[0] 为 Top-1 类别
    // scores[0] 为 Top-1 置信度
};
```

#### 使用示例

```cpp
trtyolo::ClassifyRes result = classifier->predict(image);
std::cout << "Top-1 Class: " << result.classes[0]
          << ", Score: " << result.scores[0] << std::endl;
```

---

### DetectRes

目标检测结果。

```cpp
struct TRTYOLOAPI DetectRes : public BaseRes {
    std::vector<Box> boxes;  // 检测框列表

    DetectRes() = default;
    DetectRes(int num, const std::vector<int>& classes,
              const std::vector<float>& scores,
              const std::vector<Box>& boxes);
};
```

#### 成员说明

| 成员 | 类型 | 描述 |
|-----|------|------|
| `num` | `int` | 检测到的目标数量 |
| `classes` | `std::vector<int>` | 每个目标的类别 ID |
| `scores` | `std::vector<float>` | 每个目标的置信度 |
| `boxes` | `std::vector<Box>` | 每个目标的边界框 |

---

### OBBRes

旋转目标检测结果。

```cpp
struct TRTYOLOAPI OBBRes : public BaseRes {
    std::vector<RotatedBox> boxes;  // 旋转框列表

    OBBRes() = default;
    OBBRes(int num, const std::vector<int>& classes,
           const std::vector<float>& scores,
           const std::vector<RotatedBox>& boxes);
};
```

---

### SegmentRes

实例分割结果。

```cpp
struct TRTYOLOAPI SegmentRes : public BaseRes {
    std::vector<Box>  boxes;   // 检测框列表
    std::vector<Mask> masks;   // 分割掩码列表

    SegmentRes() = default;
    SegmentRes(int num, const std::vector<int>& classes,
               const std::vector<float>& scores,
               const std::vector<Box>& boxes,
               const std::vector<Mask>& masks);
};
```

#### 使用示例

```cpp
trtyolo::SegmentRes result = segmenter->predict(image);
for (int i = 0; i < result.num; ++i) {
    const auto& mask = result.masks[i];
    // mask.data 包含 mask.width * mask.height 个浮点值
    // 值范围 [0, 1]，> 0.5 表示前景
}
```

---

### PoseRes

姿态估计结果。

```cpp
struct TRTYOLOAPI PoseRes : public BaseRes {
    std::vector<Box> boxes;                      // 人体检测框
    std::vector<std::vector<KeyPoint>> kpts;     // 关键点列表

    PoseRes() = default;
    PoseRes(int num, const std::vector<int>& classes,
            const std::vector<float>& scores,
            const std::vector<Box>& boxes,
            const std::vector<std::vector<KeyPoint>>& kpts);
};
```

#### 使用示例

```cpp
trtyolo::PoseRes result = pose_model->predict(image);
for (int i = 0; i < result.num; ++i) {
    std::cout << "Person " << i << " keypoints:" << std::endl;
    for (const auto& kp : result.kpts[i]) {
        std::cout << "  (" << kp.x << ", " << kp.y;
        if (kp.conf) std::cout << ", conf=" << *kp.conf;
        std::cout << ")" << std::endl;
    }
}
```

---

## 配置类

### InferOption

推理配置选项类，使用 PIMPL 模式隐藏实现细节。

```cpp
class TRTYOLOAPI InferOption {
public:
    InferOption();
    ~InferOption();

    // 设备配置
    void setDeviceId(int id);

    // 内存配置
    void enableCudaMem();          // 输入已在 GPU 显存中
    void enableManagedMemory();    // 使用统一内存（Jetson 推荐）

    // 性能配置
    void enablePerformanceReport();

    // 预处理配置
    void enableSwapRB();           // BGR ↔ RGB 转换
    void setBorderValue(float border_value);  // letterbox 填充值
    void setNormalizeParams(const std::vector<float>& mean,
                           const std::vector<float>& std);

    // 输入尺寸配置
    void setInputDimensions(int width, int height);
};
```

#### 方法详解

| 方法 | 描述 |
|-----|------|
| `setDeviceId(int id)` | 设置 GPU 设备 ID（多 GPU 场景） |
| `enableCudaMem()` | 声明输入数据已在 CUDA 显存中，跳过 Host→Device 拷贝 |
| `enableManagedMemory()` | 启用统一内存，适用于 Jetson 等集成 GPU 设备 |
| `enablePerformanceReport()` | 启用性能统计，可通过 `performanceReport()` 获取 |
| `enableSwapRB()` | 启用 R↔B 通道交换（OpenCV 默认 BGR，模型通常需要 RGB） |
| `setBorderValue(float)` | 设置 letterbox 填充像素值，默认 114 |
| `setNormalizeParams(mean, std)` | 设置归一化参数：`(x - mean) / std` |
| `setInputDimensions(w, h)` | 固定输入分辨率，用于视频分析等固定尺寸场景 |

#### 使用示例

```cpp
trtyolo::InferOption option;

// 基础配置
option.setDeviceId(0);
option.enableSwapRB();

// 分类模型归一化（ImageNet 标准）
option.setNormalizeParams(
    {0.485f, 0.456f, 0.406f},  // mean
    {0.229f, 0.224f, 0.225f}   // std
);

// Jetson 优化
option.enableManagedMemory();

// 性能监控
option.enablePerformanceReport();

// 固定分辨率（可选，用于 CUDA Graph 优化）
option.setInputDimensions(1920, 1080);
```

---

## 模型类

### BaseModel

所有模型的基类（不可直接实例化）。

```cpp
class TRTYOLOAPI BaseModel {
public:
    BaseModel();
    ~BaseModel();
    explicit BaseModel(const std::string& trt_engine_file,
                       const InferOption& infer_option);

    // 获取最大批量大小
    int batch() const;

    // 获取性能报告（需先启用 enablePerformanceReport）
    std::tuple<std::string, std::string, std::string> performanceReport();

protected:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

#### 方法说明

| 方法 | 返回类型 | 描述 |
|-----|---------|------|
| `batch()` | `int` | 返回模型支持的最大批量大小 |
| `performanceReport()` | `tuple<string, string, string>` | 返回 (吞吐量, CPU延迟, GPU延迟) |

---

### ClassifyModel

图像分类模型。

```cpp
class TRTYOLOAPI ClassifyModel : public BaseModel {
public:
    ClassifyModel();
    ~ClassifyModel();
    explicit ClassifyModel(const std::string& trt_engine_file,
                           const InferOption& infer_option);

    // 克隆模型（用于多线程）
    std::unique_ptr<ClassifyModel> clone() const;

    // 单张图像推理
    ClassifyRes predict(const Image& image);

    // 批量推理
    std::vector<ClassifyRes> predict(const std::vector<Image>& images);
};
```

---

### DetectModel

目标检测模型。

```cpp
class TRTYOLOAPI DetectModel : public BaseModel {
public:
    DetectModel();
    ~DetectModel();
    explicit DetectModel(const std::string& trt_engine_file,
                         const InferOption& infer_option);

    std::unique_ptr<DetectModel> clone() const;
    DetectRes predict(const Image& image);
    std::vector<DetectRes> predict(const std::vector<Image>& images);
};
```

---

### OBBModel

旋转目标检测模型。

```cpp
class TRTYOLOAPI OBBModel : public BaseModel {
public:
    OBBModel();
    ~OBBModel();
    explicit OBBModel(const std::string& trt_engine_file,
                      const InferOption& infer_option);

    std::unique_ptr<OBBModel> clone() const;
    OBBRes predict(const Image& image);
    std::vector<OBBRes> predict(const std::vector<Image>& images);
};
```

---

### SegmentModel

实例分割模型。

```cpp
class TRTYOLOAPI SegmentModel : public BaseModel {
public:
    SegmentModel();
    ~SegmentModel();
    explicit SegmentModel(const std::string& trt_engine_file,
                          const InferOption& infer_option);

    std::unique_ptr<SegmentModel> clone() const;
    SegmentRes predict(const Image& image);
    std::vector<SegmentRes> predict(const std::vector<Image>& images);
};
```

---

### PoseModel

姿态估计模型。

```cpp
class TRTYOLOAPI PoseModel : public BaseModel {
public:
    PoseModel();
    ~PoseModel();
    explicit PoseModel(const std::string& trt_engine_file,
                       const InferOption& infer_option);

    std::unique_ptr<PoseModel> clone() const;
    PoseRes predict(const Image& image);
    std::vector<PoseRes> predict(const std::vector<Image>& images);
};
```

---

## 使用示例

### 目标检测完整示例

```cpp
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

int main(int argc, char* argv[]) {
    try {
        // 配置选项
        trtyolo::InferOption option;
        option.enableSwapRB();
        option.enablePerformanceReport();

        // 创建检测器
        auto detector = std::make_unique<trtyolo::DetectModel>(
            "yolo11n.engine", option
        );

        std::cout << "Max batch size: " << detector->batch() << std::endl;

        // 加载图像
        cv::Mat frame = cv::imread("test.jpg");
        if (frame.empty()) {
            throw std::runtime_error("Failed to load image");
        }

        // 创建 Image 对象
        trtyolo::Image image(frame.data, frame.cols, frame.rows);

        // 执行推理
        trtyolo::DetectRes result = detector->predict(image);

        // 打印结果
        std::cout << "Detected " << result.num << " objects:" << std::endl;
        for (int i = 0; i < result.num; ++i) {
            auto xyxy = result.boxes[i].xyxy();
            std::cout << "  [" << i << "] class=" << result.classes[i]
                      << ", score=" << result.scores[i]
                      << ", box=(" << xyxy[0] << "," << xyxy[1]
                      << "," << xyxy[2] << "," << xyxy[3] << ")"
                      << std::endl;
        }

        // 性能报告
        auto [throughput, cpu_lat, gpu_lat] = detector->performanceReport();
        std::cout << throughput << std::endl;
        std::cout << cpu_lat << std::endl;
        std::cout << gpu_lat << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

### 批量推理示例

```cpp
// 准备批量图像
std::vector<cv::Mat> cv_images;
cv_images.push_back(cv::imread("image1.jpg"));
cv_images.push_back(cv::imread("image2.jpg"));
cv_images.push_back(cv::imread("image3.jpg"));

// 转换为 Image 对象
std::vector<trtyolo::Image> images;
for (auto& mat : cv_images) {
    images.emplace_back(mat.data, mat.cols, mat.rows);
}

// 批量推理
std::vector<trtyolo::DetectRes> results = detector->predict(images);

// 处理每个结果
for (size_t i = 0; i < results.size(); ++i) {
    std::cout << "Image " << i << ": " << results[i].num << " detections" << std::endl;
}
```

---

## 线程安全

### clone() 方法

TensorRT-YOLO **不支持**跨线程共享同一模型实例。多线程场景必须使用 `clone()` 方法：

```cpp
// 主线程创建模型
auto model = std::make_unique<trtyolo::DetectModel>("model.engine", option);

// 为每个工作线程创建克隆
std::vector<std::thread> workers;
for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back([&model]() {
        // 克隆模型（共享 ICudaEngine，独立 IExecutionContext）
        auto thread_model = model->clone();

        // 在本线程安全使用
        while (/* condition */) {
            auto result = thread_model->predict(image);
            // 处理结果...
        }
    });
}

for (auto& t : workers) t.join();
```

> **重要**: `clone()` 共享底层 TensorRT 引擎（节省 GPU 内存），但创建独立的执行上下文（保证线程安全）。

---

## 错误处理

TensorRT-YOLO 使用 C++ 异常进行错误处理：

```cpp
try {
    auto model = std::make_unique<trtyolo::DetectModel>("model.engine", option);
    auto result = model->predict(image);
} catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
} catch (const std::invalid_argument& e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

### 常见错误

| 错误类型 | 可能原因 |
|---------|---------|
| `std::runtime_error` | 引擎文件不存在、CUDA 错误、TensorRT 反序列化失败 |
| `std::invalid_argument` | 图像尺寸无效、参数配置错误 |

---

## 编译链接

### CMake 集成

```cmake
find_package(tensorrt-yolo REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE trtyolo)
```

### 手动链接

```bash
g++ -std=c++17 main.cpp -I/path/to/include -L/path/to/lib -ltrtyolo -o my_app
```

> **注意**: 使用时只需包含 `trtyolo.hpp`，无需链接 CUDA 或 TensorRT 库（已封装在 libtrtyolo 中）。
