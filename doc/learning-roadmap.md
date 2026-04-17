# TensorRT-YOLO 学习路线 (面向 C++ / CUDA 部署初学者)

> 适用版本: 6.4.0 · 分支: yyq_develop · 目标读者: 有一定 C++/Python 基础,
> 对 CUDA、TensorRT、深度学习部署几乎零经验的开发者。

本文档把整个仓库拆成 7 个循序渐进的阶段。每个阶段给出:
- **目标** — 学完能做什么
- **前置知识** — 需要先补的知识 (附推荐资源方向)
- **仓库阅读顺序** — 具体到文件/函数
- **动手练习** — 可验证的小任务

建议节奏: 每周推进 1 个阶段, 不求一次读懂所有细节, 先跑通再深挖。

---

## 阶段 0 · 环境与背景知识 (入门前置)

### 目标
跑通一个最简单的 YOLO 检测 demo, 对"模型推理"建立感性认识。

### 前置知识清单
| 主题 | 需要达到的程度 | 推荐切入点 |
|------|----------------|-----------|
| 现代 C++ (C++17) | 智能指针 `unique_ptr`/`shared_ptr`、RAII、`std::vector`、lambda | 《Effective Modern C++》前 4 章 |
| CMake | 能看懂 `add_library` / `target_link_libraries` / `find_package` | CMake 官方 tutorial Step1~Step5 |
| CUDA 基础 | kernel 是什么、`<<<grid, block>>>`、`cudaMalloc`/`cudaMemcpy`、`cudaStream_t` | NVIDIA *CUDA C++ Programming Guide* 第 1~3 章 |
| TensorRT 概念 | Engine / Runtime / ExecutionContext / Binding 的区别 | NVIDIA *TensorRT Developer Guide* 第 1~2 章 |
| YOLO 基本原理 | 输入/输出张量的含义、NMS 的作用 | Ultralytics 官方文档 "Predict" 页 |

### 仓库阅读
1. [README.md](../README.md) — 跳过所有细节, 只看"支持的模型 / 安装 / 示例"三节。
2. [doc/examples.md](examples.md) — 了解仓库里每个 example 对应什么任务。

### 动手练习
- 安装 TensorRT + CUDA, 按 [AGENTS.md](../AGENTS.md) 的 `COMMANDS` 小节编译 Python wheel。
- 用 `trtyolo-export` 导出一个 YOLO11n detect 模型为 `.engine` 文件。
- 运行 [examples/detect/detect.py](../examples/detect/detect.py), 在一张自己的图上跑出检测框。

**阶段完成标志**: 能说出"engine 文件 = 已经在这张 GPU 上编译好的模型"。

---

## 阶段 1 · Python 层 API (俯视整个系统)

### 目标
从 Python 用户视角理解这个库暴露了什么, 以及输入/输出的形状。

### 仓库阅读
1. [trtyolo/__init__.py](../trtyolo/__init__.py) — **从头读到尾**。
   - 重点看 `TRTYOLO` 类: `__init__` 里怎么根据 `task` 字符串选不同的 C++ 模型类。
   - `predict` 如何把 `numpy.ndarray` 转成 C++ 层的 `Image` 结构。
   - `convert_to_sv` 如何把 C++ 返回的结果转成 `supervision` 的数据类型。
2. [examples/detect/detect.py](../examples/detect/detect.py) — 对照上一步,
   看调用方是怎么使用 `TRTYOLO` 的。
3. 浏览 [examples/](../examples/) 下其它任务 (classify / segment / pose / obb) 的 `.py`,
   注意它们**只有 `task=` 参数和结果字段不同**, 代码结构基本相同。

### 动手练习
- 在 Python 里 `import trtyolo; help(trtyolo.TRTYOLO)`, 记录每个方法的签名。
- 改写 [examples/detect/detect.py](../examples/detect/detect.py): 去掉 `supervision` 依赖,
  直接从 `model.predict(...)` 返回的原始对象里取 `boxes`、`scores`、`classes` 并手动用 OpenCV 画框。

**阶段完成标志**: 能画出一张"Python 调用 → pybind11 → C++ 模型 → TensorRT engine"的数据流示意图。

---

## 阶段 2 · C++ 公共 API 与 PIMPL 模式

### 目标
理解 C++ 侧用户如何使用这个库, 以及作者为什么把实现"藏"起来。

### 仓库阅读
1. [modules/trtyolo/infer/trtyolo.hpp](../modules/trtyolo/infer/trtyolo.hpp) — **唯一对外暴露的头文件**,
   必须逐行读懂。重点观察:
   - `struct Image / Mask / KeyPoint / Box / RotatedBox` — 纯数据结构。
   - `struct BaseRes` → `ClassifyRes / DetectRes / OBBRes / SegmentRes / PoseRes` 的继承关系。
   - `class InferOption` 和 `class BaseModel` 里的 `class Impl; std::unique_ptr<Impl> impl_;` —
     这就是 **PIMPL (Pointer to IMPLementation) 惯用法**。
   - 派生类 `DetectModel / SegmentModel / ...` 各自提供 `predict(...)` 和 `clone()`。
2. [examples/detect/detect.cpp](../examples/detect/detect.cpp) — 看 C++ 用户怎么调 `trtyolo.hpp`。
3. [modules/trtyolo/binding/trtyolo.cpp](../modules/trtyolo/binding/trtyolo.cpp) —
   这是 pybind11 粘合层, 把上面每个 C++ 类一一映射到 Python。不需要深究 pybind11 细节,
   只需确认"Python 的每个方法对应 C++ 的哪个方法"。

### 关键概念: PIMPL 为什么重要
- `InferOption` 只在公共头里声明 `class Impl`, 实现里才定义 `Impl`。
- 好处: **头文件里不需要 `#include <NvInferRuntime.h>`**,
  库使用者不必安装 TensorRT 开发包就能编译自己的程序。
- 代价: 多一层指针间接访问, 性能可忽略。

### 动手练习
- 写一个最小的 `main.cpp`: 只 include `trtyolo.hpp`, 加载一个 engine, 对一张用 OpenCV 读入的图做一次 `predict`, 打印 `DetectRes`。
- **不要链接 TensorRT**, 只链接 `trtyolo` 本身 — 验证 PIMPL 带来的编译隔离。

**阶段完成标志**: 能解释"为什么 `trtyolo.hpp` 里没有任何 TensorRT 头文件却能使用 TensorRT"。

---

## 阶段 3 · TensorRT 运行时封装 (core 模块)

### 目标
理解 TensorRT 的三件套 (`IRuntime` / `ICudaEngine` / `IExecutionContext`) 是如何被包装的,
以及 `clone()` 是怎么做到"多线程安全"的。

### 前置补课
阅读 NVIDIA 官方 *TensorRT Developer Guide*:
- "The TensorRT Runtime" 章节 (关注 `IRuntime::deserializeCudaEngine`)
- "Execution Contexts" 章节 (关注"一个 engine 可以拥有多个 context")

### 仓库阅读
1. [modules/trtyolo/core/core.hpp](../modules/trtyolo/core/core.hpp) — 先读 3 个类的接口:
   - `TRTLogger` — 继承 `nvinfer1::ILogger`, TensorRT 要求用户提供日志回调。
   - `TRTManager` — 包装 `IRuntime` + `ICudaEngine` + `IExecutionContext`。
     **注意** `engine_` 是 `shared_ptr`, 而 `context_` 和 `runtime_` 是 `unique_ptr`。
   - `CudaGraph` — 这个先跳过, 阶段 5 再细读。
2. [modules/trtyolo/core/core.cpp](../modules/trtyolo/core/core.cpp) — 重点看:
   - `TRTManager::initialize` 是如何从 engine 二进制反序列化出 engine 对象的。
   - `TRTManager::clone()` 返回的新对象**共享同一个 engine**, 但**持有独立 context**。

### 关键概念: 为什么 engine 共享 / context 独立
- `ICudaEngine` 是**只读**的编译结果, 几百 MB, 复制成本极高。
- `IExecutionContext` 保存**一次推理**的中间状态, 多线程不能共用。
- 因此 "`shared_ptr<engine>` + 每个克隆一份 `unique_ptr<context>`" 是标准做法。

### 动手练习
- 用最朴素的 TensorRT C++ API (不使用 trtyolo), 手写一个 200 行以内的程序:
  读取 engine → 创建 context → 分配输入/输出 GPU buffer → `enqueueV3` → 拷回结果。
- 然后对比 `TRTManager::initialize` 的实现, 理解作者做了哪些额外封装。

**阶段完成标志**: 能回答"为什么 `DetectModel::clone()` 调用几乎没有开销"。

---

## 阶段 4 · GPU 显存管理 (buffer 模块)

### 目标
搞懂 CPU 和 GPU 内存之间的几种拷贝/共享模式, 以及仓库为什么要区分多种 buffer 类型。

### 前置补课
- CUDA 内存种类: *pageable* / *pinned (page-locked)* / *managed (unified)* / *device*。
- `cudaMemcpyAsync` 必须搭配 *pinned host memory* 才能真正异步。
- Jetson 上 CPU/GPU 物理共享显存, 可以用 "zero-copy" 避免拷贝。

### 仓库阅读
1. [modules/trtyolo/core/buffer.hpp](../modules/trtyolo/core/buffer.hpp) — 读完整个继承体系:
   - `BaseBuffer` 抽象基类定义接口。
   - 派生类分别对应不同内存模型 (普通 device / pinned / managed)。
2. [modules/trtyolo/core/buffer.cpp](../modules/trtyolo/core/buffer.cpp) — 重点看每个派生类的
   `allocate` / `free` / `host` / `device` 方法, 对照 CUDA API 手册。
3. 回到 [backend.hpp](../modules/trtyolo/infer/backend.hpp), 观察 `TrtBackend` 里怎么持有
   `std::unique_ptr<BaseBuffer> inputs_buffer_`, 并根据 `InferOption` 选不同派生类。

### 关键概念: 零拷贝 (zero-copy)
- `InferOption::enableManagedMemory()` 打开 managed memory 模式。
- 在 Jetson 上, 这样可以让 CPU 写入的数据 GPU 直接读到, 不走 `cudaMemcpy`。
- 在桌面 GPU 上, managed memory 由驱动自动迁移, 对性能没有好处 (甚至更慢)。

### 动手练习
- 写一个小 benchmark: 1 MB 数据在普通 `cudaMemcpy` vs. pinned memory vs. managed memory 下的拷贝耗时。
- 观察 `TrtBackend::initialize` 里是如何根据 `zero_copy_` 标志选择 buffer 类型的。

**阶段完成标志**: 能画一张图, 标出"图像从 CPU → GPU → 模型输入"这条路径上都经过哪些内存。

---

## 阶段 5 · CUDA Kernel: letterbox 预处理

### 目标
读懂第一个自定义 CUDA kernel, 理解"一个 kernel 做完整张图的预处理"是什么意思。

### 前置补课
- CUDA 2D kernel 启动写法: `dim3 block(16,16); dim3 grid((W+15)/16, (H+15)/16); kernel<<<grid, block>>>(...)`.
- 双线性插值公式: 4 邻域像素按小数部分加权。
- YOLO 的 "letterbox" 预处理: 保持长宽比缩放 + 边缘填充到固定尺寸。

### 仓库阅读
1. [modules/trtyolo/infer/letterbox.hpp](../modules/trtyolo/infer/letterbox.hpp) — 先看
   `struct Transform` (affine 变换参数) 和 host 端接口函数。
2. [modules/trtyolo/infer/letterbox.cu](../modules/trtyolo/infer/letterbox.cu) — **这是全仓库最精华的 200 行 CUDA 代码**。
   逐段阅读:
   - `__global__ void` kernel 的线程索引计算 `int dx = blockIdx.x * blockDim.x + threadIdx.x;`。
   - 如何用**逆仿射变换**从输出像素坐标算出源像素坐标 (这样可以并行, 每个输出像素独立)。
   - 如何在**同一个 kernel** 里完成: 双线性采样 + BGR↔RGB 通道交换 + `/255` 归一化 + HWC→CHW 转置。
   - 这种"算子融合"是 GPU 性能优化的核心手段: 避免多次读写显存。
3. 对照 [backend.cpp](../modules/trtyolo/infer/backend.cpp) 里调用 letterbox 的地方,
   看输入指针是怎么传进去的。

### 动手练习
- 把 letterbox kernel 单独抽出来, 写一个最小可运行的 `.cu` 文件, 读一张 jpg → 调 kernel → 写出 CHW 的 float 张量到文件, 用 Python 校验结果和 `cv2` + numpy 版本一致。
- 用 Nsight Systems 或 `nvprof` 量一下这个 kernel 的耗时, 对比纯 OpenCV 的预处理耗时。

**阶段完成标志**: 能解释"为什么这个 kernel 把 4 个步骤融合成 1 个, 比分别做要快几倍"。

---

## 阶段 6 · 后端与 CUDA Graph (backend 模块)

### 目标
理解完整的推理循环: 预处理 → enqueueV3 → 后处理, 以及 CUDA Graph 如何加速静态 shape 推理。

### 前置补课
- CUDA Stream: 多条 stream 可并发, 同一条 stream 内按顺序执行。
- CUDA Graph (CUDA 10+): 把一串 stream 操作"录制"成一张图, 后续可以以极低的 CPU 开销重复执行。
  官方介绍: *CUDA C++ Programming Guide* 的 "CUDA Graphs" 小节。

### 仓库阅读
1. [modules/trtyolo/infer/backend.hpp](../modules/trtyolo/infer/backend.hpp) —
   `TrtBackend` 的成员变量是理解全貌的地图: stream / manager / buffer / cuda_graph / dynamic 标志等。
2. [modules/trtyolo/infer/backend.cpp](../modules/trtyolo/infer/backend.cpp) — 按方法读:
   - `TrtBackend(...)` 构造函数: 加载 engine → `getTensorInfo()` → `initialize()` → (可选) `captureCudaGraph()`。
   - `getTensorInfo()`: 遍历 engine 的每个 binding, 记录 shape / dtype / 是否输入。
   - `staticInfer(...)` vs `dynamicInfer(...)`: 静态 shape 走 graph 回放, 动态 shape 每次都要重新 `setInputShape`。
   - `captureCudaGraph()`: `beginCapture` → 跑一遍完整流程 → `endCapture`。
3. 回到 [core.hpp](../modules/trtyolo/core/core.hpp) 的 `CudaGraph` 类 —
   现在你应该能看懂 `updateKernelNodeParams` / `updateMemcpyNodeParams` 的用途了 (换输入地址时不用重录图)。
4. [modules/trtyolo/infer/trtyolo.cpp](../modules/trtyolo/infer/trtyolo.cpp) —
   最后读 `DetectModel::predict` / `SegmentModel::predict` 等方法里的**后处理**代码。
   重点观察: 它们如何从 TensorRT 输出张量的裸指针上直接解析出 `Box` / `Mask` / `KeyPoint`。

### 关键概念: CUDA Graph 为什么能加速
- 传统做法: 每次推理 CPU 都要提交几十个 kernel launch, 每次 launch ~5µs 的 driver 开销累计起来很可观。
- CUDA Graph: 把这一串 launch 录成 graph, 之后 `cudaGraphLaunch` 只有 1 次 CPU 开销。
- 限制: 录制时的内存地址、tensor shape 必须保持不变 → 所以只有**静态 shape 模型**才能用。

### 动手练习
- 在 Python 里创建模型时, 一次用 `profile=True`, 一次不开, 分别跑 100 次推理看耗时差异。
- 故意把一个静态 shape 模型和一个动态 shape 模型都加载一遍,
  断点看 `backend.cpp` 里 `dynamic` 标志的分支。

**阶段完成标志**: 能解释"为什么 CUDA Graph 只对静态 shape 模型有效, 对动态 shape 不能用"。

---

## 阶段 7 · TensorRT 插件 (plugin 模块 · 进阶)

### 目标
看懂自定义 NMS 插件是怎么以"动态库"形式被 TensorRT 加载的, 以及 NMS 的 GPU 实现思路。

### 前置补课
- TensorRT Plugin 机制: 继承 `IPluginV2DynamicExt`, 注册到 `IPluginRegistry`, engine 反序列化时自动查找。
- NMS 算法: 按 score 排序 → 依次保留最高分框 → 丢弃与之 IoU 超过阈值的其它框。
- GPU 并行 NMS 的难点: "依赖前一个结果"的循环很难并行, 需要用 mask 矩阵化的思路。

### 仓库阅读
1. [modules/plugin/AGENTS.md](../modules/plugin/AGENTS.md) — 作者写的 plugin 模块导读, 先读这个。
2. [modules/plugin/common/plugin.h](../modules/plugin/common/plugin.h) — 插件公共基类。
3. [modules/plugin/efficientIdxNMSPlugin/efficientIdxNMSPlugin.h](../modules/plugin/efficientIdxNMSPlugin/efficientIdxNMSPlugin.h) / `.cpp` —
   以最简单的 "带索引输出的 NMS" 为例, 看插件类的生命周期方法 (`configurePlugin`,
   `getOutputDimensions`, `enqueue` 等)。
4. [modules/plugin/efficientIdxNMSPlugin/efficientIdxNMSInference.cu](../modules/plugin/efficientIdxNMSPlugin/efficientIdxNMSInference.cu) —
   NMS 的 CUDA 实现, 用到 `cub` 库做排序。
5. [efficientRotatedNMSPlugin/](../modules/plugin/efficientRotatedNMSPlugin/) — 旋转框 NMS, 在普通 NMS 基础上把 IoU 计算换成旋转矩形求交 (SAT 算法)。

### 关键概念: 插件 ↔ 主库为什么"运行时"耦合
- 主 `trtyolo` 库**不直接链接** plugin, 它在启动时通过 `initLibNvInferPlugins(nullptr, "")` 让 TensorRT 加载已注册的插件。
- 好处: 插件可以独立编译、版本化, 不影响主库 ABI。
- 代价: 运行时如果插件 `.so` 没加载, engine 反序列化会失败并报"unknown plugin"。

### 动手练习
- 把 `modules/plugin/` 以 `Release` 编译, 看 `libnvinfer_plugin_trtyolo.so` 的符号表 (`nm -D | grep Efficient`)。
- 尝试删掉这个 so 再运行 detect 示例, 观察报错信息, 建立"插件缺失 ≠ 程序 bug"的直觉。

**阶段完成标志**: 能独立实现一个"空插件" (输出 = 输入的 `IPluginV2DynamicExt` 子类), 编译、注册、并在一个虚构的 ONNX 图里被 TensorRT 识别。

---

## 学完之后能做什么

- **改模型**: 给 `trtyolo.cpp` 增加一个新的 `XxxModel`, 对接一个新的 YOLO 变体。
- **改预处理**: 往 `letterbox.cu` 里加一个可选的 "center crop" 分支。
- **改后处理**: 用自己的 NMS 插件替换 `efficientIdxNMSPlugin`, 比如加入 Soft-NMS。
- **跨框架**: 把 `TrtBackend` 抽象成接口, 加一个 `OnnxRuntimeBackend` 实现, 让同一套前后处理代码支持 ORT。

## 仓库之外的延伸阅读

- NVIDIA **TensorRT Samples** 代码库 — 官方给的各种 plugin / API 用例, 是最好的参考实现。
- **cuBLAS / cuDNN** 官方文档 — 理解 TensorRT 底层用的数学库。
- **Triton Inference Server** 代码 — 看工业级的 TensorRT 服务化部署是怎么做的。
- **Nsight Systems / Nsight Compute** — 两个调优必备工具, 学会看 timeline 和 kernel metrics。

---

## 附录: 快速索引

| 想搞懂这个 | 去读这个文件 |
|-----------|-------------|
| Python 用户怎么调库 | [trtyolo/__init__.py](../trtyolo/__init__.py) |
| C++ 用户怎么调库 | [modules/trtyolo/infer/trtyolo.hpp](../modules/trtyolo/infer/trtyolo.hpp) |
| Python ↔ C++ 粘合层 | [modules/trtyolo/binding/trtyolo.cpp](../modules/trtyolo/binding/trtyolo.cpp) |
| TensorRT 运行时封装 | [modules/trtyolo/core/core.cpp](../modules/trtyolo/core/core.cpp) |
| 显存 buffer 管理 | [modules/trtyolo/core/buffer.cpp](../modules/trtyolo/core/buffer.cpp) |
| letterbox CUDA 实现 | [modules/trtyolo/infer/letterbox.cu](../modules/trtyolo/infer/letterbox.cu) |
| 完整推理循环 / CUDA Graph | [modules/trtyolo/infer/backend.cpp](../modules/trtyolo/infer/backend.cpp) |
| 各任务的后处理 | [modules/trtyolo/infer/trtyolo.cpp](../modules/trtyolo/infer/trtyolo.cpp) |
| 自定义 NMS 插件 | [modules/plugin/efficientIdxNMSPlugin/](../modules/plugin/efficientIdxNMSPlugin/) |
| 多线程 clone 用法 | [examples/mutli_thread/mutli_thread.cpp](../examples/mutli_thread/mutli_thread.cpp) |
