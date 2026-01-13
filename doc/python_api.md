# TensorRT-YOLO Python API Reference

> **Version:** 6.4.0 | **Package:** `trtyolo` | **依赖:** `supervision`, `opencv-python`, `numpy`

本文档详细介绍 TensorRT-YOLO 的 Python API，包括所有公共类、方法及其使用方式。

---

## 目录

- [安装](#安装)
- [快速入门](#快速入门)
- [TRTYOLO 类](#trtyolo-类)
  - [构造函数](#构造函数)
  - [属性](#属性)
  - [方法](#方法)
- [推理结果](#推理结果)
  - [sv.Detections](#svdetections)
  - [sv.KeyPoints](#svkeypoints)
  - [sv.Classifications](#svclassifications)
- [InferOption 类](#inferoption-类)
- [底层模型类](#底层模型类)
- [使用示例](#使用示例)
- [性能优化](#性能优化)
- [线程安全](#线程安全)
- [常见问题](#常见问题)

---

## 安装

### 从源码构建

```bash
# 1. 构建 C++ 库（需要 TensorRT）
pip install "pybind11[global]"
cmake -S . -B build -D TRT_PATH=/path/to/tensorrt -D BUILD_PYTHON=ON
cmake --build build -j$(nproc) --config Release --target install

# 2. 构建 Python wheel
pip install --upgrade build
python -m build --wheel
pip install dist/trtyolo-6.4.0-py3-none-any.whl
```

### 验证安装

```python
from trtyolo import TRTYOLO
print("TensorRT-YOLO installed successfully!")
```

---

## 快速入门

```python
import cv2
from trtyolo import TRTYOLO

# 1. 创建模型
model = TRTYOLO("yolo11n.engine", task="detect")

# 2. 加载图像并推理
image = cv2.imread("test.jpg")
result = model.predict(image)

# 3. 结果是 supervision.Detections 对象
print(f"检测到 {len(result)} 个目标")
print(f"类别: {result.class_id}")
print(f"置信度: {result.confidence}")
print(f"边界框: {result.xyxy}")
```

---

## TRTYOLO 类

`TRTYOLO` 是 TensorRT-YOLO 的统一 Python 接口，支持五种视觉任务：

- **detect** - 目标检测
- **segment** - 实例分割
- **classify** - 图像分类
- **pose** - 姿态估计
- **obb** - 旋转目标检测

### 构造函数

```python
class TRTYOLO:
    def __init__(
        self,
        model: Union[str, Path],
        task: str,
        device: Optional[int] = 0,
        swap_rb: Optional[bool] = True,
        profile: Optional[bool] = False,
        border_value: Optional[float] = None,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None
```

#### 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|-----|------|-------|------|
| `model` | `str \| Path` | 必填 | TensorRT 引擎文件路径（`.engine`） |
| `task` | `str` | 必填 | 任务类型：`"detect"`, `"segment"`, `"classify"`, `"pose"`, `"obb"` |
| `device` | `int` | `0` | GPU 设备 ID |
| `swap_rb` | `bool` | `True` | 是否在预处理时交换 R↔B 通道 |
| `profile` | `bool` | `False` | 是否启用性能分析 |
| `border_value` | `float` | `None` | letterbox 填充值（默认 114） |
| `mean` | `Tuple[float, float, float]` | `None` | 归一化均值，必须与 `std` 成对使用 |
| `std` | `Tuple[float, float, float]` | `None` | 归一化标准差，必须与 `mean` 成对使用 |
| `input_size` | `Tuple[int, int]` | `None` | 固定输入分辨率 `(width, height)`，用于视频分析等场景 |

#### 使用示例

```python
# 基础检测模型
detector = TRTYOLO("yolo11n.engine", task="detect")

# 指定 GPU 和性能分析
detector = TRTYOLO(
    "yolo11n.engine",
    task="detect",
    device=1,
    profile=True
)

# 分类模型（带 ImageNet 归一化）
classifier = TRTYOLO(
    "yolo11n-cls.engine",
    task="classify",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# 视频分析（固定分辨率优化）
video_detector = TRTYOLO(
    "yolo11n.engine",
    task="detect",
    input_size=(1920, 1080),  # 固定输入尺寸
    profile=True
)
```

---

### 属性

#### `batch`

```python
@property
def batch(self) -> int
```

获取模型支持的最大批量大小。

```python
model = TRTYOLO("yolo11n.engine", task="detect")
print(f"最大批量大小: {model.batch}")
```

#### `task_map`

```python
@property
def task_map(self) -> Dict[str, Any]
```

返回任务名称到底层模型类的映射。

```python
# {'classify': ClassifyModel, 'detect': DetectModel, ...}
print(model.task_map)
```

---

### 方法

#### `predict()`

执行推理。

```python
def predict(
    self,
    source: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
) -> Union[
    sv.Detections | sv.KeyPoints | sv.Classifications,
    List[sv.Detections | sv.KeyPoints | sv.Classifications],
]
```

##### 参数

| 参数 | 类型 | 描述 |
|-----|------|------|
| `source` | `str \| Path \| np.ndarray \| List[...]` | 输入图像，支持路径、NumPy 数组或列表 |

##### 返回值

根据任务类型返回不同的 `supervision` 对象：

| 任务 | 返回类型 |
|-----|---------|
| `detect` | `sv.Detections` |
| `segment` | `sv.Detections`（含 `mask`） |
| `classify` | `sv.Classifications` |
| `pose` | `sv.KeyPoints` |
| `obb` | `sv.Detections`（含 `data[ORIENTED_BOX_COORDINATES]`） |

##### 使用示例

```python
import cv2
import numpy as np
from trtyolo import TRTYOLO

model = TRTYOLO("yolo11n.engine", task="detect")

# 方式 1: 直接传入文件路径
result = model.predict("image.jpg")

# 方式 2: 传入 NumPy 数组
image = cv2.imread("image.jpg")
result = model.predict(image)

# 方式 3: 批量推理（路径列表）
results = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])

# 方式 4: 批量推理（数组列表）
images = [cv2.imread(f"img{i}.jpg") for i in range(5)]
results = model.predict(images)
```

---

#### `clone()`

创建模型的浅拷贝（用于多线程）。

```python
def clone(self) -> "TRTYOLO"
```

##### 返回值

| 类型 | 描述 |
|-----|------|
| `TRTYOLO` | 共享相同 TensorRT 引擎的新实例 |

##### 使用示例

```python
import threading

model = TRTYOLO("yolo11n.engine", task="detect")

def worker(model_instance, image_path):
    result = model_instance.predict(image_path)
    print(f"Thread {threading.current_thread().name}: {len(result)} detections")

# 为每个线程创建克隆
threads = []
for i in range(4):
    cloned = model.clone()  # 克隆模型
    t = threading.Thread(target=worker, args=(cloned, f"image{i}.jpg"))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

---

#### `profile()`

获取性能统计信息。

```python
def profile(self) -> Tuple[str, str, str]
```

##### 返回值

| 索引 | 类型 | 描述 |
|-----|------|------|
| 0 | `str` | 吞吐量，如 `"Throughput: 120.14 qps"` |
| 1 | `str` | CPU 延迟，如 `"CPU Latency: min = 8.32 ms, max = 8.35 ms, mean = 8.33 ms, ..."` |
| 2 | `str` | GPU 延迟，如 `"GPU Latency: min = 8.12 ms, max = 8.15 ms, mean = 8.13 ms, ..."` |

> **注意**: 必须在初始化时设置 `profile=True` 才能获取有效数据，否则返回空字符串。

##### 使用示例

```python
model = TRTYOLO("yolo11n.engine", task="detect", profile=True)

# 执行多次推理以收集统计数据
for i in range(100):
    model.predict(image)

# 获取性能报告
throughput, cpu_latency, gpu_latency = model.profile()
print(throughput)
print(cpu_latency)
print(gpu_latency)
```

---

#### `__call__()`

`predict()` 的别名，支持函数调用语法。

```python
def __call__(
    self,
    source: Union[str, Path, np.ndarray, List[...]],
) -> Union[sv.Detections, sv.KeyPoints, sv.Classifications, List[...]]
```

```python
model = TRTYOLO("yolo11n.engine", task="detect")
result = model("image.jpg")  # 等价于 model.predict("image.jpg")
```

---

## 推理结果

TensorRT-YOLO 的 Python API 自动将推理结果转换为 [supervision](https://github.com/roboflow/supervision) 格式，便于后续可视化和处理。

### sv.Detections

用于 `detect`、`segment`、`obb` 任务。

```python
from trtyolo import TRTYOLO
import supervision as sv

model = TRTYOLO("yolo11n.engine", task="detect")
result = model.predict(image)  # 返回 sv.Detections

# 访问检测结果
print(result.xyxy)        # np.ndarray, shape (n, 4), 边界框 [x1, y1, x2, y2]
print(result.confidence)  # np.ndarray, shape (n,), 置信度
print(result.class_id)    # np.ndarray, shape (n,), 类别 ID

# 遍历检测结果
for xyxy, confidence, class_id in result:
    x1, y1, x2, y2 = xyxy
    print(f"Class {class_id}: {confidence:.2f} at ({x1}, {y1}, {x2}, {y2})")
```

#### Segment 任务特有属性

```python
model = TRTYOLO("yolo11n-seg.engine", task="segment")
result = model.predict(image)

# 访问分割掩码
print(result.mask)  # np.ndarray, shape (n, H, W), bool 类型
```

#### OBB 任务特有属性

```python
model = TRTYOLO("yolo11n-obb.engine", task="obb")
result = model.predict(image)

# 访问旋转框顶点坐标
xyxyxyxy = result.data[sv.config.ORIENTED_BOX_COORDINATES]
print(xyxyxyxy)  # np.ndarray, shape (n, 4, 2), 四个顶点坐标
```

---

### sv.KeyPoints

用于 `pose` 任务。

```python
model = TRTYOLO("yolo11n-pose.engine", task="pose")
result = model.predict(image)  # 返回 sv.KeyPoints

# 访问关键点
print(result.xy)          # np.ndarray, shape (n, m, 2), 关键点坐标
print(result.confidence)  # np.ndarray, shape (n, m), 关键点置信度（可能为 None）
print(result.class_id)    # np.ndarray, shape (n,), 类别 ID
```

---

### sv.Classifications

用于 `classify` 任务。

```python
model = TRTYOLO("yolo11n-cls.engine", task="classify")
result = model.predict(image)  # 返回 sv.Classifications

# 访问分类结果
print(result.class_id)     # np.ndarray, shape (k,), Top-K 类别 ID
print(result.confidence)   # np.ndarray, shape (k,), Top-K 置信度
```

---

## InferOption 类

底层推理配置类（通常通过 `TRTYOLO` 构造函数参数间接使用）。

```python
from trtyolo import c_lib_wrap as C

option = C.option.InferOption()
option.set_device_id(0)           # 设置 GPU ID
option.enable_swap_rb()           # 启用 R↔B 交换
option.enable_profile()           # 启用性能分析
option.set_border_value(114.0)    # 设置 letterbox 填充值
option.set_normalize_params(      # 设置归一化参数
    (0.485, 0.456, 0.406),        # mean
    (0.229, 0.224, 0.225)         # std
)
option.set_input_dimensions(1920, 1080)  # 固定输入尺寸
```

---

## 底层模型类

如需更精细的控制，可直接使用底层模型类：

```python
from trtyolo import c_lib_wrap as C

# 创建配置
option = C.option.InferOption()
option.set_device_id(0)
option.enable_swap_rb()

# 创建底层模型
model = C.model.DetectModel("yolo11n.engine", option)

# 推理（返回 C++ 结果对象）
result = model.predict(image)  # image 必须是 np.ndarray

# 访问结果属性
print(result.xyxy)        # np.ndarray
print(result.confidence)  # np.ndarray
print(result.class_id)    # np.ndarray
```

### 可用模型类

| 类名 | 对应任务 | 结果类型 |
|-----|---------|---------|
| `C.model.ClassifyModel` | 图像分类 | `C.result.ClassifyRes` |
| `C.model.DetectModel` | 目标检测 | `C.result.DetectRes` |
| `C.model.OBBModel` | 旋转目标检测 | `C.result.OBBRes` |
| `C.model.SegmentModel` | 实例分割 | `C.result.SegmentRes` |
| `C.model.PoseModel` | 姿态估计 | `C.result.PoseRes` |

---

## 使用示例

### 目标检测与可视化

```python
import cv2
import supervision as sv
from trtyolo import TRTYOLO

# 初始化
model = TRTYOLO("yolo11n.engine", task="detect", profile=True)
image = cv2.imread("test.jpg")

# 推理
detections = model.predict(image)

# 可视化
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 准备标签
labels = [f"class_{cid}: {conf:.2f}" for cid, conf in zip(
    detections.class_id, detections.confidence
)]

# 绘制
annotated = box_annotator.annotate(image.copy(), detections)
annotated = label_annotator.annotate(annotated, detections, labels)

# 保存结果
cv2.imwrite("result.jpg", annotated)

# 打印性能
throughput, cpu_lat, gpu_lat = model.profile()
print(throughput)
```

### 实例分割

```python
import cv2
import supervision as sv
from trtyolo import TRTYOLO

model = TRTYOLO("yolo11n-seg.engine", task="segment")
image = cv2.imread("test.jpg")

detections = model.predict(image)

# 使用掩码进行可视化
mask_annotator = sv.MaskAnnotator()
annotated = mask_annotator.annotate(image.copy(), detections)
cv2.imwrite("segmentation.jpg", annotated)
```

### 姿态估计

```python
import cv2
import supervision as sv
from trtyolo import TRTYOLO

model = TRTYOLO("yolo11n-pose.engine", task="pose")
image = cv2.imread("test.jpg")

keypoints = model.predict(image)

# 绘制骨架
edge_annotator = sv.EdgeAnnotator()
vertex_annotator = sv.VertexAnnotator()

annotated = edge_annotator.annotate(image.copy(), keypoints)
annotated = vertex_annotator.annotate(annotated, keypoints)
cv2.imwrite("pose.jpg", annotated)
```

### 视频处理

```python
import cv2
from trtyolo import TRTYOLO
import supervision as sv

# 使用固定输入尺寸优化
model = TRTYOLO(
    "yolo11n.engine",
    task="detect",
    input_size=(1920, 1080),  # 视频分辨率
    profile=True
)

cap = cv2.VideoCapture("video.mp4")
box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = model.predict(frame)
    annotated = box_annotator.annotate(frame, detections)

    cv2.imshow("Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 打印性能统计
print(model.profile()[0])
```

### 批量处理

```python
from pathlib import Path
from trtyolo import TRTYOLO

model = TRTYOLO("yolo11n.engine", task="detect")

# 批量处理目录中的图像
image_dir = Path("images/")
image_paths = list(image_dir.glob("*.jpg"))

# 批量推理
results = model.predict(image_paths)

# 处理结果
for path, result in zip(image_paths, results):
    print(f"{path.name}: {len(result)} detections")
```

---

## 性能优化

### 1. 固定输入尺寸

对于固定分辨率的输入（如视频流），设置 `input_size` 可启用 CUDA Graph 优化：

```python
model = TRTYOLO(
    "model.engine",
    task="detect",
    input_size=(1920, 1080)  # 固定分辨率
)
```

### 2. 批量推理

利用批量推理提高吞吐量：

```python
# 准备批量图像
images = [cv2.imread(f"img{i}.jpg") for i in range(model.batch)]

# 批量推理
results = model.predict(images)
```

### 3. 禁用不必要的预处理

如果输入已经是 RGB 格式，可禁用通道交换：

```python
model = TRTYOLO("model.engine", task="detect", swap_rb=False)
```

### 4. 使用 profile 监控性能

```python
model = TRTYOLO("model.engine", task="detect", profile=True)

# 运行推理后获取统计
for _ in range(100):
    model.predict(image)

throughput, cpu_lat, gpu_lat = model.profile()
print(f"吞吐量: {throughput}")
print(f"CPU 延迟: {cpu_lat}")
print(f"GPU 延迟: {gpu_lat}")
```

---

## 线程安全

### 重要规则

**不要**在多个线程间共享同一个 `TRTYOLO` 实例！

### 正确做法：使用 clone()

```python
import threading
from queue import Queue
from trtyolo import TRTYOLO

# 主线程创建模型
model = TRTYOLO("model.engine", task="detect")

def worker(model_clone, task_queue, result_queue):
    while True:
        image = task_queue.get()
        if image is None:
            break
        result = model_clone.predict(image)
        result_queue.put(result)

# 创建工作线程
num_workers = 4
task_queue = Queue()
result_queue = Queue()
threads = []

for _ in range(num_workers):
    clone = model.clone()  # 每个线程一个克隆
    t = threading.Thread(target=worker, args=(clone, task_queue, result_queue))
    t.start()
    threads.append(t)

# 分发任务
for image in images:
    task_queue.put(image)

# 停止工作线程
for _ in range(num_workers):
    task_queue.put(None)

for t in threads:
    t.join()
```

---

## 常见问题

### Q: 如何获取类别名称？

A: TensorRT-YOLO 返回的是类别 ID，需要自行维护 ID 到名称的映射：

```python
class_names = {0: "person", 1: "car", 2: "dog", ...}

for cid, conf in zip(detections.class_id, detections.confidence):
    name = class_names.get(cid, f"class_{cid}")
    print(f"{name}: {conf:.2f}")
```

### Q: 模型文件从哪里来？

A: 使用 `trtyolo-export` 工具（在 `export` 分支）导出：

```bash
git checkout export
pip install -e .
trtyolo-export --model yolo11n.pt --task detect --output yolo11n.engine
```

### Q: 为什么 copy/deepcopy 不工作？

A: 模型包含 GPU 资源，不支持普通复制。请使用 `clone()` 方法：

```python
# 错误
import copy
model2 = copy.copy(model)  # 抛出 NotImplementedError

# 正确
model2 = model.clone()
```

### Q: 任务类型必须匹配吗？

A: **是的**。`task` 参数必须与模型导出时的任务类型一致，否则结果会错误或报错。

```python
# 正确
model = TRTYOLO("yolo11n-seg.engine", task="segment")

# 错误
model = TRTYOLO("yolo11n-seg.engine", task="detect")  # 任务不匹配！
```

### Q: 如何在 Jetson 上优化？

A: 使用统一内存（需在 C++ 层通过 `InferOption` 设置）：

```python
from trtyolo import c_lib_wrap as C

option = C.option.InferOption()
option.enable_managed_memory()  # 启用统一内存

model = C.model.DetectModel("model.engine", option)
```

---

## API 速查表

| 操作 | 代码 |
|-----|------|
| 创建模型 | `model = TRTYOLO("model.engine", task="detect")` |
| 推理单张 | `result = model.predict(image)` |
| 推理批量 | `results = model.predict([img1, img2, img3])` |
| 克隆模型 | `clone = model.clone()` |
| 性能分析 | `throughput, cpu_lat, gpu_lat = model.profile()` |
| 获取批量大小 | `batch_size = model.batch` |
