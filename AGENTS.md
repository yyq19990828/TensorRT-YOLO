# AGENTS.md - TensorRT-YOLO

> **Generated:** 2026-01-13 | **Commit:** b3ffeb3 | **Branch:** yyq_develop

High-performance YOLO inference toolkit for NVIDIA. C++/Python bindings for TensorRT-accelerated detection, segmentation, classification, pose, OBB.

**Stack:** TensorRT, CUDA, pybind11, CMake, Python

## STRUCTURE

```
TensorRT-YOLO/
├── modules/
│   ├── trtyolo/           # Core inference library
│   │   ├── infer/         # TensorRT backend + CUDA kernels
│   │   │   ├── trtyolo.hpp    # Single public header (C++ entry)
│   │   │   ├── backend.*      # TensorRT + CUDA Graph logic
│   │   │   └── letterbox.cu   # Fused preprocessing kernel
│   │   ├── core/          # Buffer + memory management
│   │   ├── binding/       # pybind11 → Python
│   │   └── utils/         # Timers, macros, configs
│   └── plugin/            # TensorRT NMS plugins (see plugin/AGENTS.md)
├── trtyolo/               # Python package
│   └── __init__.py        # TRTYOLO class → supervision integration
├── examples/              # C++ and Python demos per task
└── pyproject.toml         # Ruff config, dependencies
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new model task | `modules/trtyolo/infer/trtyolo.cpp` | Add postProcess*, update binding |
| Modify preprocessing | `modules/trtyolo/infer/letterbox.cu` | CUDA kernel, perf-critical |
| Change inference flow | `modules/trtyolo/infer/backend.cpp` | CUDA Graph logic here |
| Python API changes | `trtyolo/__init__.py` | Update task_map, convert_to_sv |
| Custom NMS plugins | `modules/plugin/` | See `modules/plugin/AGENTS.md` |
| Add example | `examples/{task}/` | Mirror C++ and Python |

## COMMANDS

```bash
# Build C++ library
cmake -S . -B build -D TRT_PATH=/path/to/tensorrt -D BUILD_PYTHON=ON
cmake --build build -j$(nproc) --config Release --target install

# Build Python wheel
pip install --upgrade build && python -m build --wheel
pip install dist/trtyolo-6.*-py3-none-any.whl

# Lint & Format
ruff check .          # Lint
ruff format .         # Format (preserves quotes)
ruff check --fix .    # Auto-fix

# Verify (no formal tests - use examples)
python examples/detect/detect.py -e model.engine -i image.jpg -o output/
```

## ANTI-PATTERNS (THIS PROJECT)

| Rule | Violation | Consequence |
|------|-----------|-------------|
| **NEVER** share model across threads | Use same instance in multiple threads | Race conditions, crashes |
| **ALWAYS** use `model.clone()` | Direct assignment for multi-threading | Unsafe shared state |
| **NEVER** use `as any`, `@ts-ignore` | Type suppression in bindings | Hidden bugs |
| **NEVER** compare types directly | `type(x) == T` in Python | Use `isinstance()` |
| **NEVER** mismatch task type | `task="detect"` with segment model | Inference failure |
| **DO NOT** edit raw pointers | Manual memory in C++ | Use `std::unique_ptr` |
| **DO NOT** skip PIMPL | Expose TensorRT headers publicly | ABI breakage |

## CONVENTIONS

### Python (Ruff)
- Line length: **140** | Quote style: **preserve** | isort: `combine-as-imports`
- Import order: stdlib → third-party (`cv2`, `numpy`, `supervision`) → local (`trtyolo`)

### C++ (.clang-format)
- Base: **Google** | Indent: **4 spaces** | Pointer: **Left** (`int* ptr`)
- Classes: `PascalCase` | Methods: `camelCase` | Private members: `snake_case_`
- Single header: users only include `trtyolo.hpp`

### Memory
- PIMPL idiom for `InferOption`, `BaseModel` (ABI stability)
- `std::unique_ptr` for ownership
- Zero-copy via `enableManagedMemory()` on Jetson

## COMPLEXITY HOTSPOTS

| File | Complexity | Risk |
|------|------------|------|
| `backend.cpp` | CUDA Graph capture/replay | Graph breaks if non-deterministic ops added |
| `letterbox.cu` | Fused bilinear + normalize + swapRB | Precision-sensitive, uses intrinsics |
| `trtyolo.cpp` | postProcess* methods | Direct pointer arithmetic on TRT outputs |
| `plugin/*.cu` | NMS CUDA kernels | Shared memory, cub sorting, tile-based IoU |

## KEY APIs

### Python
```python
from trtyolo import TRTYOLO
model = TRTYOLO("model.engine", task="detect", swap_rb=True, profile=True)
result = model.predict(image)  # Returns sv.Detections
model.clone()  # Thread-safe copy
```

### C++
```cpp
#include "trtyolo.hpp"
trtyolo::InferOption option;
option.enableSwapRB();
auto model = std::make_unique<trtyolo::DetectModel>("model.engine", option);
auto result = model->predict(image);  // Returns DetectRes
model->clone();  // Thread-safe copy
```

## ARCHITECTURE NOTES

1. **Plugin ↔ Core decoupled**: `modules/plugin` not linked at compile time; TensorRT registry loads plugins at runtime
2. **CUDA Graph**: Static-shape models capture preprocess→infer→copy into replayable graph
3. **Clone pattern**: Shares `ICudaEngine`, creates independent `IExecutionContext` per thread
4. **Supervision integration**: Python results auto-convert to `sv.Detections`/`sv.KeyPoints`/`sv.Classifications`

## PREREQUISITES

- CUDA >= 11.0.1
- TensorRT >= 8.6.1
- CMake >= 3.18
- C++17 compiler
- `pybind11[global]` for Python bindings

## GOTCHAS

- **Model export**: Use `trtyolo-export` from `export` branch (not vanilla ultralytics export)
- **No CI**: Verification via `examples/` scripts only
- **Engine compatibility**: Task type in code must match export-time task
- **Windows**: Set `CMAKE_INSTALL_PREFIX` in PATH for `find_package` to work
