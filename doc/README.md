# TensorRT-YOLO API Documentation

> **Version:** 6.4.0 | **License:** GPL-3.0

TensorRT-YOLO 是一款专为 NVIDIA 设备设计的高性能 YOLO 推理部署工具，提供 C++ 和 Python 双语言接口。

---

## 文档目录

| 文档 | 描述 |
|-----|------|
| [C++ API Reference](cpp_api.md) | C++ 接口完整参考，包含所有类、结构体和方法 |
| [Python API Reference](python_api.md) | Python 接口完整参考，包含 TRTYOLO 类和 supervision 集成 |
| [Examples](examples.md) | 各任务场景的完整代码示例 |

---

## 支持的任务

| 任务 | C++ 模型类 | Python 参数 | 返回类型 |
|-----|-----------|-------------|---------|
| 目标检测 | `DetectModel` | `task="detect"` | `DetectRes` / `sv.Detections` |
| 实例分割 | `SegmentModel` | `task="segment"` | `SegmentRes` / `sv.Detections` |
| 图像分类 | `ClassifyModel` | `task="classify"` | `ClassifyRes` / `sv.Classifications` |
| 姿态估计 | `PoseModel` | `task="pose"` | `PoseRes` / `sv.KeyPoints` |
| 旋转目标检测 | `OBBModel` | `task="obb"` | `OBBRes` / `sv.Detections` |

---

## 快速开始

### C++

```cpp
#include "trtyolo.hpp"

int main() {
    trtyolo::InferOption option;
    option.enableSwapRB();

    auto detector = std::make_unique<trtyolo::DetectModel>("yolo11n.engine", option);

    cv::Mat image = cv::imread("test.jpg");
    trtyolo::Image img(image.data, image.cols, image.rows);

    auto result = detector->predict(img);
    std::cout << "Detected: " << result.num << " objects" << std::endl;

    return 0;
}
```

### Python

```python
from trtyolo import TRTYOLO

model = TRTYOLO("yolo11n.engine", task="detect")
result = model.predict("test.jpg")
print(f"Detected: {len(result)} objects")
```

---

## 核心概念

### 1. 推理选项 (InferOption)

配置推理行为的核心类：

| 选项 | C++ 方法 | Python 参数 | 描述 |
|-----|---------|-------------|------|
| GPU 设备 | `setDeviceId(id)` | `device=0` | 选择 GPU |
| 通道交换 | `enableSwapRB()` | `swap_rb=True` | BGR↔RGB 转换 |
| 性能分析 | `enablePerformanceReport()` | `profile=True` | 启用延迟统计 |
| 填充值 | `setBorderValue(val)` | `border_value=114` | letterbox 填充 |
| 归一化 | `setNormalizeParams(mean, std)` | `mean=(), std=()` | 图像归一化 |
| 固定尺寸 | `setInputDimensions(w, h)` | `input_size=(w, h)` | CUDA Graph 优化 |

### 2. 模型克隆 (clone)

多线程场景下，**必须**使用 `clone()` 方法创建独立实例：

```cpp
// C++
auto thread_model = model->clone();
```

```python
# Python
thread_model = model.clone()
```

### 3. 结果结构

所有结果继承自 `BaseRes`，包含：

| 字段 | 类型 | 描述 |
|-----|------|------|
| `num` | `int` | 检测数量 |
| `classes` | `vector<int>` / `np.ndarray` | 类别 ID |
| `scores` | `vector<float>` / `np.ndarray` | 置信度 |

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python Application                         │
│                    (trtyolo.TRTYOLO)                            │
├─────────────────────────────────────────────────────────────────┤
│                      supervision                                │
│              (sv.Detections, sv.KeyPoints, ...)                 │
├─────────────────────────────────────────────────────────────────┤
│                    pybind11 Bindings                            │
│              (modules/trtyolo/binding/)                         │
├─────────────────────────────────────────────────────────────────┤
│                   C++ Core Library                              │
│              (DetectModel, SegmentModel, ...)                   │
├─────────────────────────────────────────────────────────────────┤
│                   TensorRT Backend                              │
│       (CUDA Graph, Letterbox Kernel, NMS Plugins)               │
├─────────────────────────────────────────────────────────────────┤
│                    NVIDIA TensorRT                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 系统要求

| 组件 | 版本要求 |
|-----|---------|
| CUDA | >= 11.0.1 |
| TensorRT | >= 8.6.1 |
| CMake | >= 3.18 |
| C++ 编译器 | C++17 支持 |
| Python | >= 3.8 |
| pybind11 | 用于 Python 绑定 |

---

## 编译安装

### C++ 库

```bash
# 安装 pybind11（用于 Python 绑定）
pip install "pybind11[global]"

# 配置
cmake -S . -B build \
    -D TRT_PATH=/path/to/tensorrt \
    -D BUILD_PYTHON=ON \
    -D CMAKE_INSTALL_PREFIX=./install

# 编译安装
cmake --build build -j$(nproc) --config Release --target install
```

### Python Wheel

```bash
pip install --upgrade build
python -m build --wheel
pip install dist/trtyolo-6.4.0-py3-none-any.whl
```

---

## 性能优化建议

| 场景 | 建议 |
|-----|------|
| 视频分析 | 使用 `input_size` 固定分辨率，启用 CUDA Graph |
| 批量处理 | 使用批量推理 API，充分利用 GPU 并行 |
| 多线程 | 使用 `clone()` 创建线程本地模型 |
| Jetson | 启用 `enableManagedMemory()` 使用统一内存 |
| 精度敏感 | 确保 `swap_rb` 与训练时一致 |

---

## 常见问题

### Q: 模型文件如何获取？

使用 `trtyolo-export` 工具（在 `export` 分支）：

```bash
git checkout export
pip install -e .
trtyolo-export --model yolo11n.pt --task detect --output yolo11n.engine
```

### Q: 为什么不能跨线程共享模型？

TensorRT 的 `IExecutionContext` 不是线程安全的。`clone()` 方法共享引擎但创建独立上下文。

### Q: 任务类型不匹配会怎样？

会导致推理失败或输出错误结果。`task` 参数必须与模型导出时一致。

---

## 相关链接

- [GitHub Repository](https://github.com/laugh12321/TensorRT-YOLO)
- [Issues & Bug Reports](https://github.com/laugh12321/TensorRT-YOLO/issues)
- [supervision Library](https://github.com/roboflow/supervision)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
