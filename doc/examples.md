# TensorRT-YOLO 使用示例

本文档提供各种任务场景的完整代码示例，包括 C++ 和 Python 两种语言。

---

## 目录

- [目标检测 (Detect)](#目标检测-detect)
- [实例分割 (Segment)](#实例分割-segment)
- [图像分类 (Classify)](#图像分类-classify)
- [姿态估计 (Pose)](#姿态估计-pose)
- [旋转目标检测 (OBB)](#旋转目标检测-obb)
- [批量推理](#批量推理)
- [多线程推理](#多线程推理)
- [视频处理](#视频处理)
- [性能分析](#性能分析)

---

## 目标检测 (Detect)

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""目标检测示例"""

import cv2
import supervision as sv
from trtyolo import TRTYOLO

def main():
    # 初始化模型
    model = TRTYOLO(
        "yolo11n.engine",
        task="detect",
        swap_rb=True,
        profile=True
    )
    
    # 加载图像
    image = cv2.imread("test.jpg")
    
    # 推理
    detections = model.predict(image)
    
    # 打印结果
    print(f"检测到 {len(detections)} 个目标")
    for i, (xyxy, conf, cls_id) in enumerate(zip(
        detections.xyxy, 
        detections.confidence, 
        detections.class_id
    )):
        print(f"  [{i}] class={cls_id}, conf={conf:.3f}, "
              f"box=({xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f})")
    
    # 可视化
    box_annotator = sv.BoxAnnotator()
    annotated = box_annotator.annotate(image.copy(), detections)
    cv2.imwrite("detect_result.jpg", annotated)
    
    # 性能报告
    throughput, cpu_lat, gpu_lat = model.profile()
    print(f"\n{throughput}")

if __name__ == "__main__":
    main()
```

### C++

```cpp
/**
 * @file detect.cpp
 * @brief 目标检测示例
 */

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

        // 加载图像
        cv::Mat cv_image = cv::imread("test.jpg");
        if (cv_image.empty()) {
            throw std::runtime_error("Failed to load image");
        }

        // 封装图像
        trtyolo::Image image(cv_image.data, cv_image.cols, cv_image.rows);

        // 推理
        trtyolo::DetectRes result = detector->predict(image);

        // 打印结果
        std::cout << "检测到 " << result.num << " 个目标" << std::endl;
        for (int i = 0; i < result.num; ++i) {
            auto xyxy = result.boxes[i].xyxy();
            std::cout << "  [" << i << "] class=" << result.classes[i]
                      << ", conf=" << result.scores[i]
                      << ", box=(" << xyxy[0] << ", " << xyxy[1]
                      << ", " << xyxy[2] << ", " << xyxy[3] << ")"
                      << std::endl;
        }

        // 可视化
        for (int i = 0; i < result.num; ++i) {
            auto xyxy = result.boxes[i].xyxy();
            cv::rectangle(cv_image,
                cv::Point(xyxy[0], xyxy[1]),
                cv::Point(xyxy[2], xyxy[3]),
                cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("detect_result.jpg", cv_image);

        // 性能报告
        auto [throughput, cpu_lat, gpu_lat] = detector->performanceReport();
        std::cout << "\n" << throughput << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

---

## 实例分割 (Segment)

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实例分割示例"""

import cv2
import numpy as np
import supervision as sv
from trtyolo import TRTYOLO

def main():
    # 初始化模型
    model = TRTYOLO("yolo11n-seg.engine", task="segment")
    
    # 加载图像
    image = cv2.imread("test.jpg")
    
    # 推理
    detections = model.predict(image)
    
    # 打印结果
    print(f"分割了 {len(detections)} 个实例")
    
    # 可视化掩码
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    
    annotated = mask_annotator.annotate(image.copy(), detections)
    annotated = box_annotator.annotate(annotated, detections)
    
    cv2.imwrite("segment_result.jpg", annotated)
    
    # 访问掩码数据
    if detections.mask is not None:
        for i, mask in enumerate(detections.mask):
            print(f"  Mask {i}: shape={mask.shape}, "
                  f"area={np.sum(mask)} pixels")

if __name__ == "__main__":
    main()
```

### C++

```cpp
/**
 * @file segment.cpp
 * @brief 实例分割示例
 */

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

int main() {
    try {
        trtyolo::InferOption option;
        option.enableSwapRB();

        auto segmenter = std::make_unique<trtyolo::SegmentModel>(
            "yolo11n-seg.engine", option
        );

        cv::Mat cv_image = cv::imread("test.jpg");
        trtyolo::Image image(cv_image.data, cv_image.cols, cv_image.rows);

        trtyolo::SegmentRes result = segmenter->predict(image);

        std::cout << "分割了 " << result.num << " 个实例" << std::endl;

        // 可视化每个掩码
        cv::Mat overlay = cv_image.clone();
        std::vector<cv::Scalar> colors = {
            {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
            {255, 255, 0}, {255, 0, 255}, {0, 255, 255}
        };

        for (int i = 0; i < result.num; ++i) {
            const auto& mask = result.masks[i];
            const auto& box = result.boxes[i];
            auto xyxy = box.xyxy();

            // 将掩码应用到图像
            cv::Mat mask_mat(mask.height, mask.width, CV_32F, 
                           const_cast<float*>(mask.data.data()));
            cv::Mat mask_resized;
            cv::resize(mask_mat, mask_resized, 
                      cv::Size(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]));

            cv::Scalar color = colors[i % colors.size()];
            for (int y = 0; y < mask_resized.rows; ++y) {
                for (int x = 0; x < mask_resized.cols; ++x) {
                    if (mask_resized.at<float>(y, x) > 0.5f) {
                        int px = xyxy[0] + x;
                        int py = xyxy[1] + y;
                        if (px >= 0 && px < overlay.cols && 
                            py >= 0 && py < overlay.rows) {
                            overlay.at<cv::Vec3b>(py, px) = 
                                overlay.at<cv::Vec3b>(py, px) * 0.5 +
                                cv::Vec3b(color[0], color[1], color[2]) * 0.5;
                        }
                    }
                }
            }
        }

        cv::imwrite("segment_result.jpg", overlay);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

---

## 图像分类 (Classify)

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图像分类示例"""

import cv2
from trtyolo import TRTYOLO

# ImageNet 类别名称（示例）
IMAGENET_CLASSES = {
    0: "tench", 1: "goldfish", 2: "great_white_shark",
    # ... 完整列表请参考 ImageNet
}

def main():
    # 初始化分类模型（带 ImageNet 归一化）
    model = TRTYOLO(
        "yolo11n-cls.engine",
        task="classify",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    # 加载图像
    image = cv2.imread("test.jpg")
    
    # 推理
    classifications = model.predict(image)
    
    # 打印 Top-K 结果
    print("分类结果:")
    for i, (cls_id, conf) in enumerate(zip(
        classifications.class_id, 
        classifications.confidence
    )):
        name = IMAGENET_CLASSES.get(cls_id, f"class_{cls_id}")
        print(f"  Top-{i+1}: {name} ({cls_id}), confidence={conf:.4f}")

if __name__ == "__main__":
    main()
```

### C++

```cpp
/**
 * @file classify.cpp
 * @brief 图像分类示例
 */

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

int main() {
    try {
        trtyolo::InferOption option;
        option.enableSwapRB();
        option.setNormalizeParams(
            {0.485f, 0.456f, 0.406f},  // mean
            {0.229f, 0.224f, 0.225f}   // std
        );

        auto classifier = std::make_unique<trtyolo::ClassifyModel>(
            "yolo11n-cls.engine", option
        );

        cv::Mat cv_image = cv::imread("test.jpg");
        trtyolo::Image image(cv_image.data, cv_image.cols, cv_image.rows);

        trtyolo::ClassifyRes result = classifier->predict(image);

        std::cout << "分类结果:" << std::endl;
        for (int i = 0; i < result.num; ++i) {
            std::cout << "  Top-" << (i + 1) << ": class=" << result.classes[i]
                      << ", confidence=" << result.scores[i] << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

---

## 姿态估计 (Pose)

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""姿态估计示例"""

import cv2
import supervision as sv
from trtyolo import TRTYOLO

# COCO 骨架连接
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上身
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16)  # 下身
]

def main():
    model = TRTYOLO("yolo11n-pose.engine", task="pose")
    
    image = cv2.imread("test.jpg")
    keypoints = model.predict(image)
    
    print(f"检测到 {len(keypoints.xy)} 个人体")
    
    # 可视化
    edge_annotator = sv.EdgeAnnotator(thickness=2)
    vertex_annotator = sv.VertexAnnotator(radius=4)
    
    annotated = edge_annotator.annotate(image.copy(), keypoints)
    annotated = vertex_annotator.annotate(annotated, keypoints)
    
    cv2.imwrite("pose_result.jpg", annotated)
    
    # 打印关键点详情
    for person_idx, (xy, conf) in enumerate(zip(
        keypoints.xy, 
        keypoints.confidence if keypoints.confidence is not None else [None] * len(keypoints.xy)
    )):
        print(f"\n人物 {person_idx}:")
        for kp_idx, point in enumerate(xy):
            c = conf[kp_idx] if conf is not None else 'N/A'
            print(f"  关键点 {kp_idx}: ({point[0]:.1f}, {point[1]:.1f}), conf={c}")

if __name__ == "__main__":
    main()
```

### C++

```cpp
/**
 * @file pose.cpp
 * @brief 姿态估计示例
 */

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

int main() {
    try {
        trtyolo::InferOption option;
        option.enableSwapRB();

        auto pose_model = std::make_unique<trtyolo::PoseModel>(
            "yolo11n-pose.engine", option
        );

        cv::Mat cv_image = cv::imread("test.jpg");
        trtyolo::Image image(cv_image.data, cv_image.cols, cv_image.rows);

        trtyolo::PoseRes result = pose_model->predict(image);

        std::cout << "检测到 " << result.num << " 个人体" << std::endl;

        // 绘制关键点和骨架
        std::vector<std::pair<int, int>> skeleton = {
            {0, 1}, {0, 2}, {1, 3}, {2, 4},
            {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
            {5, 11}, {6, 12}, {11, 12},
            {11, 13}, {13, 15}, {12, 14}, {14, 16}
        };

        for (int i = 0; i < result.num; ++i) {
            const auto& kpts = result.kpts[i];
            
            // 绘制关键点
            for (const auto& kp : kpts) {
                cv::circle(cv_image, cv::Point(kp.x, kp.y), 4, 
                          cv::Scalar(0, 255, 0), -1);
            }
            
            // 绘制骨架
            for (const auto& [start, end] : skeleton) {
                if (start < kpts.size() && end < kpts.size()) {
                    cv::line(cv_image,
                        cv::Point(kpts[start].x, kpts[start].y),
                        cv::Point(kpts[end].x, kpts[end].y),
                        cv::Scalar(255, 0, 0), 2);
                }
            }
        }

        cv::imwrite("pose_result.jpg", cv_image);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

---

## 旋转目标检测 (OBB)

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""旋转目标检测示例"""

import cv2
import numpy as np
import supervision as sv
from trtyolo import TRTYOLO

def main():
    model = TRTYOLO("yolo11n-obb.engine", task="obb")
    
    image = cv2.imread("aerial.jpg")  # 航拍图像
    detections = model.predict(image)
    
    print(f"检测到 {len(detections)} 个旋转目标")
    
    # 获取旋转框顶点
    if sv.config.ORIENTED_BOX_COORDINATES in detections.data:
        xyxyxyxy = detections.data[sv.config.ORIENTED_BOX_COORDINATES]
        
        # 绘制旋转框
        for i, points in enumerate(xyxyxyxy):
            # points shape: (4, 2) - 四个顶点
            pts = points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            
            # 标注类别和置信度
            center = np.mean(points, axis=0).astype(int)
            label = f"cls:{detections.class_id[i]} {detections.confidence[i]:.2f}"
            cv2.putText(image, label, tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite("obb_result.jpg", image)

if __name__ == "__main__":
    main()
```

### C++

```cpp
/**
 * @file obb.cpp
 * @brief 旋转目标检测示例
 */

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

int main() {
    try {
        trtyolo::InferOption option;
        option.enableSwapRB();

        auto obb_detector = std::make_unique<trtyolo::OBBModel>(
            "yolo11n-obb.engine", option
        );

        cv::Mat cv_image = cv::imread("aerial.jpg");
        trtyolo::Image image(cv_image.data, cv_image.cols, cv_image.rows);

        trtyolo::OBBRes result = obb_detector->predict(image);

        std::cout << "检测到 " << result.num << " 个旋转目标" << std::endl;

        // 绘制旋转框
        for (int i = 0; i < result.num; ++i) {
            auto vertices = result.boxes[i].xyxyxyxy();
            
            std::vector<cv::Point> pts = {
                {vertices[0], vertices[1]},
                {vertices[2], vertices[3]},
                {vertices[4], vertices[5]},
                {vertices[6], vertices[7]}
            };
            
            cv::polylines(cv_image, pts, true, cv::Scalar(0, 255, 0), 2);
        }

        cv::imwrite("obb_result.jpg", cv_image);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

---

## 批量推理

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量推理示例"""

import cv2
from pathlib import Path
from trtyolo import TRTYOLO

def main():
    model = TRTYOLO("yolo11n.engine", task="detect", profile=True)
    
    # 获取图像列表
    image_dir = Path("images/")
    image_paths = list(image_dir.glob("*.jpg"))
    
    print(f"模型批量大小: {model.batch}")
    print(f"待处理图像: {len(image_paths)} 张")
    
    # 方式 1: 传入路径列表
    results = model.predict(image_paths)
    
    # 方式 2: 传入数组列表
    # images = [cv2.imread(str(p)) for p in image_paths]
    # results = model.predict(images)
    
    # 处理结果
    for path, result in zip(image_paths, results):
        print(f"{path.name}: {len(result)} 个检测")
    
    # 性能统计
    throughput, _, _ = model.profile()
    print(f"\n{throughput}")

if __name__ == "__main__":
    main()
```

### C++

```cpp
/**
 * @file batch.cpp
 * @brief 批量推理示例
 */

#include <iostream>
#include <memory>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

namespace fs = std::filesystem;

int main() {
    try {
        trtyolo::InferOption option;
        option.enableSwapRB();
        option.enablePerformanceReport();

        auto detector = std::make_unique<trtyolo::DetectModel>(
            "yolo11n.engine", option
        );

        std::cout << "模型批量大小: " << detector->batch() << std::endl;

        // 加载图像
        std::vector<cv::Mat> cv_images;
        std::vector<std::string> filenames;
        
        for (const auto& entry : fs::directory_iterator("images/")) {
            if (entry.path().extension() == ".jpg") {
                cv_images.push_back(cv::imread(entry.path().string()));
                filenames.push_back(entry.path().filename().string());
            }
        }

        // 转换为 Image 对象
        std::vector<trtyolo::Image> images;
        for (auto& mat : cv_images) {
            images.emplace_back(mat.data, mat.cols, mat.rows);
        }

        // 批量推理
        auto results = detector->predict(images);

        // 处理结果
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << filenames[i] << ": " 
                      << results[i].num << " 个检测" << std::endl;
        }

        // 性能报告
        auto [throughput, cpu_lat, gpu_lat] = detector->performanceReport();
        std::cout << "\n" << throughput << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

---

## 多线程推理

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多线程推理示例"""

import cv2
import threading
from queue import Queue
from trtyolo import TRTYOLO

def worker(model_clone, task_queue, result_queue, thread_id):
    """工作线程函数"""
    while True:
        item = task_queue.get()
        if item is None:
            break
        
        image_path, image = item
        result = model_clone.predict(image)
        result_queue.put((thread_id, image_path, len(result)))
        task_queue.task_done()

def main():
    # 主线程创建模型
    model = TRTYOLO("yolo11n.engine", task="detect")
    
    # 准备任务和结果队列
    task_queue = Queue()
    result_queue = Queue()
    
    # 创建工作线程
    num_workers = 4
    threads = []
    
    for i in range(num_workers):
        # 每个线程获得模型克隆
        model_clone = model.clone()
        t = threading.Thread(
            target=worker,
            args=(model_clone, task_queue, result_queue, i)
        )
        t.start()
        threads.append(t)
    
    # 分发任务
    image_paths = [f"image_{i}.jpg" for i in range(20)]
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            task_queue.put((path, image))
    
    # 等待任务完成
    task_queue.join()
    
    # 停止工作线程
    for _ in range(num_workers):
        task_queue.put(None)
    
    for t in threads:
        t.join()
    
    # 收集结果
    while not result_queue.empty():
        thread_id, path, count = result_queue.get()
        print(f"Thread {thread_id}: {path} -> {count} detections")

if __name__ == "__main__":
    main()
```

### C++

```cpp
/**
 * @file multithread.cpp
 * @brief 多线程推理示例
 */

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

std::mutex cout_mutex;

void worker(std::unique_ptr<trtyolo::DetectModel> model, 
            const std::vector<std::string>& image_paths,
            int thread_id) {
    for (const auto& path : image_paths) {
        cv::Mat cv_image = cv::imread(path);
        if (cv_image.empty()) continue;

        trtyolo::Image image(cv_image.data, cv_image.cols, cv_image.rows);
        auto result = model->predict(image);

        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Thread " << thread_id << ": " << path 
                  << " -> " << result.num << " detections" << std::endl;
    }
}

int main() {
    try {
        // 主线程创建模型
        trtyolo::InferOption option;
        option.enableSwapRB();

        auto main_model = std::make_unique<trtyolo::DetectModel>(
            "yolo11n.engine", option
        );

        // 准备图像路径
        std::vector<std::vector<std::string>> thread_tasks = {
            {"img1.jpg", "img2.jpg", "img3.jpg"},
            {"img4.jpg", "img5.jpg", "img6.jpg"},
            {"img7.jpg", "img8.jpg", "img9.jpg"},
            {"img10.jpg", "img11.jpg", "img12.jpg"}
        };

        // 创建工作线程
        std::vector<std::thread> threads;
        for (size_t i = 0; i < thread_tasks.size(); ++i) {
            // 克隆模型给每个线程
            auto cloned_model = main_model->clone();
            threads.emplace_back(worker, std::move(cloned_model), 
                               thread_tasks[i], i);
        }

        // 等待所有线程完成
        for (auto& t : threads) {
            t.join();
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

---

## 视频处理

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""视频处理示例"""

import cv2
import supervision as sv
from trtyolo import TRTYOLO

def main():
    # 使用固定输入尺寸优化（适用于固定分辨率视频）
    model = TRTYOLO(
        "yolo11n.engine",
        task="detect",
        input_size=(1920, 1080),  # 视频分辨率
        profile=True
    )
    
    # 打开视频
    cap = cv2.VideoCapture("video.mp4")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    # 创建标注器
    box_annotator = sv.BoxAnnotator()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        detections = model.predict(frame)
        
        # 标注
        annotated = box_annotator.annotate(frame, detections)
        
        # 写入
        out.write(annotated)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧")
    
    cap.release()
    out.release()
    
    # 性能统计
    throughput, cpu_lat, gpu_lat = model.profile()
    print(f"\n处理完成: {frame_count} 帧")
    print(throughput)

if __name__ == "__main__":
    main()
```

---

## 性能分析

### Python

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""性能分析示例"""

import cv2
import time
from trtyolo import TRTYOLO

def main():
    # 启用性能分析
    model = TRTYOLO(
        "yolo11n.engine",
        task="detect",
        profile=True,
        input_size=(640, 640)  # 固定尺寸以获得更稳定的性能
    )
    
    image = cv2.imread("test.jpg")
    
    # 预热
    print("预热中...")
    for _ in range(10):
        model.predict(image)
    
    # 性能测试
    print("性能测试中...")
    num_iterations = 100
    
    start_time = time.time()
    for _ in range(num_iterations):
        result = model.predict(image)
    end_time = time.time()
    
    # 计算统计
    total_time = end_time - start_time
    avg_time = total_time / num_iterations * 1000  # ms
    fps = num_iterations / total_time
    
    print(f"\n=== 性能报告 ===")
    print(f"总迭代次数: {num_iterations}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均耗时: {avg_time:.2f} ms/张")
    print(f"吞吐量: {fps:.2f} FPS")
    
    # 内置性能报告
    throughput, cpu_lat, gpu_lat = model.profile()
    print(f"\n=== 内置报告 ===")
    print(throughput)
    print(cpu_lat)
    print(gpu_lat)

if __name__ == "__main__":
    main()
```

---

## CMakeLists.txt 示例

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_trtyolo_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# 查找 tensorrt-yolo 包
find_package(tensorrt-yolo REQUIRED)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 创建可执行文件
add_executable(detect detect.cpp)
target_link_libraries(detect PRIVATE trtyolo ${OpenCV_LIBS})

add_executable(segment segment.cpp)
target_link_libraries(segment PRIVATE trtyolo ${OpenCV_LIBS})

add_executable(pose pose.cpp)
target_link_libraries(pose PRIVATE trtyolo ${OpenCV_LIBS})
```

---

## 编译运行

```bash
# 配置
cmake -S . -B build

# 编译
cmake --build build -j8

# 运行
./build/detect
./build/segment
./build/pose
```
