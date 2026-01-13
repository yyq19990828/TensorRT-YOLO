# AGENTS.md - TensorRT Custom Plugins

> Parent: [../../AGENTS.md](../../AGENTS.md)

Custom TensorRT plugins for accelerated NMS. Required for **OBB**, **Segment**, **Pose** tasks.

## STRUCTURE

```
plugin/
├── common/                    # Shared plugin infrastructure
│   ├── plugin.cpp             # Plugin registration macros
│   ├── vfcCommon.cpp          # Workspace + error checking
│   └── common.cu              # Shared CUDA utilities
├── efficientIdxNMSPlugin/     # Standard NMS with indices
│   ├── efficientIdxNMSPlugin.cpp   # IPluginV2DynamicExt impl
│   └── efficientIdxNMSInference.cu # CUDA NMS kernels
└── efficientRotatedNMSPlugin/ # OBB NMS (ProbIoU)
    ├── efficientRotatedNMSPlugin.cpp
    └── efficientRotatedNMSInference.cu
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new plugin | Copy existing plugin dir | Follow `IPluginV2DynamicExt` pattern |
| Modify NMS logic | `*Inference.cu` | CUDA kernels, shared memory |
| Change I/O tensors | `*Plugin.cpp` | Update getNbOutputs, getOutputDimensions |
| Fix serialization | `*Plugin.cpp` | serialize/deserialize must match |

## ANTI-PATTERNS

- **NEVER** mismatch batch/boxes/classes dims across inputs
- **NEVER** add host sync inside `enqueue()` (breaks CUDA Graph)
- **DO NOT** change tensor order without updating `trtyolo.cpp` postProcess*

## PLUGIN LIFECYCLE

```
1. getOutputDimensions() — Called during engine build
2. supportsFormatCombination() — Validate FP32/FP16/INT32 combos
3. configurePlugin() — Store runtime shapes
4. getWorkspaceSize() — Return temp memory needs
5. enqueue() — Execute CUDA kernels (called per inference)
6. serialize() — Save plugin state to engine file
7. deserialize() — Restore plugin from engine file
```

## KEY ALGORITHMS

| Algorithm | File | Notes |
|-----------|------|-------|
| Tile-based NMS | `efficientIdxNMSInference.cu` | Parallelized IoU across box tiles |
| ProbIoU (Rotated) | `efficientRotatedNMSInference.cu` | Gaussian-based IoU for OBB |
| Radix sort | Both `.cu` files | Uses `cub::DeviceSegmentedRadixSort` |

## BUILD

Built as separate `libcustom_plugins.so`. Only needed when:
- Using `trtexec` to build OBB/Segment/Pose engines
- Loading engines that embed these plugins

```bash
# Plugins built automatically with main project
cmake --build build -j$(nproc) --config Release

# For trtexec usage
trtexec --onnx=model.onnx --staticPlugins=lib/libcustom_plugins.so
```

## CONVENTIONS

- Plugin classes: `Efficient*NMSPlugin`
- Plugin names: `EfficientIdxNMS_TRT`, `EfficientRotatedNMS_TRT`
- Output format: Always `BoxCorner` (x1,y1,x2,y2) regardless of input

## COMPLEXITY

| File | Risk | Details |
|------|------|---------|
| `*Inference.cu` | HIGH | Shared memory coordination, warp-level ops |
| `*Plugin.cpp` | MEDIUM | Shape inference, workspace calculation |
| `common/` | LOW | Stable infrastructure code |
