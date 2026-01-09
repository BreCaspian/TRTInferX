# TRTInferX For YOLOv11 🚀

High-performance TensorRT inference engine for YOLOv11 INT8 PTQ, designed for **North China University of Science and Technology HORIZON Team ROBOMASTER 2026 Radar Group Radar Localization System and Anti-Drone Laser Tracking System**.

![Demo](docs/result.gif)

*KITTI video demo: TRTInferX real-world video inference on YOLOv11 across multiple FP16/INT8 static and dynamic batch configurations, with the best-performing setup reaching 301.9 FPS and 0.57 ms GPU time (INT8 dynamic batch=16, Video summary FPS record).*

*References*: [`Performance report`](docs/PerformanceReport.md) | [`Performance log`](docs/performance.txt) | [`Runtime log`](docs/log.txt)

---

## Features

- A high-performance inference engine for YOLOv11 **object detection**, supporting FP16 and INT8 with static and dynamic batch, compatible with both `nms=True` (EfficientNMS inside engine) and `nms=False` (raw output + internal NMS) export paths. Preprocess, decode, and coordinate restoration are done on CUDA, and the input format is consistent with the default Ultralytics pipeline for both performance and portability.

- Measured (KITTI video, RTX 3060 Laptop GPU) highest stable about **301.9 FPS** (INT8 Dynamic batch=16); by measurement, maximum end-to-end throughput under full load is about **1522.66 FPS** (INT8 batch=32, infStreams=2, with transfers) / **746.28 FPS** (FP16 batch=64, infStreams=1, with transfers); theoretical compute upper bound (`trtexec --noDataTransfers`) can reach about **1858 FPS** (INT8 batch=128), which is used to measure pure inference ceiling; end-to-end performance is impacted by H2D/D2H.

- Note: The test environment did not fully exploit the inference engine's ceiling; **real performance should be close to the measured estimates** (the platform had significant unrelated load to obtain the metrics). Also, the test model is from Ultralytics official [yolov11n.pt](https://docs.ultralytics.com/zh/models/yolo11/), with no structure changes. Optimizing the model structure can further improve the performance ceiling.
- All test models (.pt/.onnx/.engine), test data (videos, images), and test results (videos, images) are available in Releases.

---

## Dependencies

- CUDA Toolkit >= 11.8 (12.x recommended; must match GPU driver)
- TensorRT >= 10.0 (runtime and build versions must match)
- nvinfer_plugin must match TensorRT major version (e.g., 10.x with 10.x)
- nvonnxparser (required only for building ONNX → engine)
- OpenCV >= 4.5 (examples)
- CMake >= 3.18
- C++17 compiler (GCC 9+/Clang 10+)

---

## Test Environment

- Computer: Lenovo Legion Y9000P IAH7H
- CPU: 12th Gen Intel Core i9-12900H
- GPU: NVIDIA GA106M (GeForce RTX 3060 Mobile / Max-Q)
- OS: Ubuntu 22.04.5 LTS
- CUDA: 13.0 (nvcc 13.0.48, Driver 580.95.05, CUDA runtime 13.0)
- TensorRT: 10.14.1 (system packages, libnvinfer/libnvinfer_plugin)
- OpenCV: 4.5.4 (system), 4.12.0 (conda/python)

---

## Directory Structure (Please strictly follow this layout)

```
yolov11/
├── TRTInferX/
│   ├── include/                     # Public headers and API definitions
│   │   ├── api.h                    # External unified API (ImageInput/Det/Api)
│   │   ├── Inference.h              # TRT inference wrapper and runtime context
│   │   ├── preprocess.h             # Preprocess interface declarations
│   │   ├── postprocess.h            # Postprocess/NMS interface declarations
│   │   ├── logging.h                # Logging and debug
│   │   └── macros.h                 # Common macros and error checks
│   ├── src/                         # Main inference flow implementation
│   │   ├── Api.cpp                  # API implementation (load/infer/inferWithInfo)
│   │   └── Inference.cpp            # TRT inference main flow (IO/scheduling/postprocess)
│   ├── kernel/                      # CUDA preprocess and postprocess kernels
│   │   ├── preprocess.cu            # letterbox + normalize
│   │   └── postprocess.cu           # raw decode + NMS / coordinate restore
│   ├── examples/
│   │   └── yolo11/
│   │       ├── main.cpp             # Example entry
│   │       └── include/main.h
│   ├── scripts/                     # Export/calibration scripts
│   ├── CMakeLists.txt               # Main build script
│   └── build/                       # Build outputs
├── models/
│   ├── initial/                     # Original .pt weights
│   └── exports/                     # Engine and calibration outputs
│       ├── best_fp16.engine
│       ├── best_int8.engine
│       ├── best.onnx / best_raw.onnx
│       ├── calib.bin / trtexec.cache
└── test/
    ├── images/coco128/images/       # Test input images
    ├── videos/                      # Test videos
    └── output/                      # Test output images/videos
```

Note: The top-level folder name can be customized, but keep the rest of the structure exactly the same; otherwise, the engine export/calibration scripts may fail.

---

## Build

```bash
cd TRTInferX
mkdir -p build
cd build
cmake .. \
  -DTRT_INCLUDE_DIR=/path/to/TensorRT/include \
  -DTRT_LIB_DIR=/path/to/TensorRT/lib \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j
```

> If `CMAKE_CUDA_ARCHITECTURES` is not set, it will be detected via `nvidia-smi` and set automatically; otherwise default is `86`.

---

The project prioritizes auto-detected GPU architecture for build optimization. You can override manually:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
```

Supported architectures (`compute_XX / SM_XX`) example list:

| Arch | Description |
| --- | --- |
| **SM60 / compute_60** | Pascal: Quadro GP100, Tesla P100, DGX-1 |
| **SM61 / compute_61** | Pascal: GTX 10 series, Titan Xp, Tesla P4/P40 |
| **SM62 / compute_62** | Jetson TX2 |
| **SM70 / compute_70** | Volta: Tesla V100 |
| **SM72 / compute_72** | Xavier / Xavier NX |
| **SM75 / compute_75** | Turing: RTX 20 series, Tesla T4 |
| **SM80 / compute_80** | Ampere: A100 |
| **SM86 / compute_86** | Ampere: RTX 3060/3070/3080/3090 |
| **SM87 / compute_87** | Jetson Orin |
| **SM89 / compute_89** | Lovelace: RTX 4090/4080 |
| **SM90 / compute_90** | Hopper: H100 |

---

## Export FP16/INT8 .engine from .pt

Export strategy selection:
- FP16: `nms=True` (engine NMS) + static `batch=1` is the most stable and lowest latency.
- INT8: `nms=False` + `trtexec` calibration + TRTInferX internal NMS is usually easiest to succeed; for maximum performance you can try INT8 with `nms=True`, but success depends on model and calibration data.

Default export strategy (scripts):
- FP16: `nms=True`, packed output `[B,300,6]` with engine NMS.
- INT8: `nms=False` raw output, `trtexec` calibration, then TRTInferX NMS on GPU.

> Note: FP16 does not require `nms=True`, but engine NMS is recommended (packed output is more stable and postprocess is simpler). `nms=False` can also use TRTInferX internal NMS, but it adds decode+NMS and is more complex with little benefit.

> Reminder: If you change `--imgsz` or `--int8-batch`, regenerate `calib.bin`. Static batch engines must run with the same batch they were built with.

**Activate Conda environment, run from repo root** (e.g., `/home/yao/TEST/yolov11`) (~~grab a coffee~~; export can take a while depending on system performance):

```bash
TRTInferX/scripts/export_all.sh \
  --pt models/initial/yolo11n.pt \
  --images test/images/coco128/images \
  --out-dir models/exports \
  --imgsz 640 \
  --fp16-batch 1 \
  --int8-batch 1
```

Prerequisites:
- Python environment has `ultralytics` installed.
- `test/images/coco128/images` image count >= `--int8-batch` (default 8) for INT8 PTQ calibration inputs.

Export separately:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

PYTHON_BIN=$(which python) TRTInferX/scripts/export_fp16_engine.sh \
  --pt models/initial/yolo11n.pt \
  --out-dir models/exports \
  --imgsz 640 \
  --batch 1 \
  --dynamic 1 \
  --nms 1
```

```bash
PYTHON_BIN=$(which python) TRTInferX/scripts/export_int8_engine.sh \
  --pt models/initial/yolo11n.pt \
  --images test/images/coco128/images \
  --out-dir models/exports \
  --imgsz 640 \
  --batch 1 \
  --dynamic 1
```

> You can override Python/TRT via env vars:
> `PYTHON_BIN=/path/to/python TRTEXEC=/path/to/trtexec TRTInferX/scripts/export_all.sh ...`

> Export scripts rely on the exact project structure above.

---

## Example Run

Commands below assume `TRTInferX/build`:

```bash
cd TRTInferX/build
./trt_yolo_example \
  --engine ../../models/exports/best_fp16.engine \
  --image ../../test/images/coco128/images/000000000036.jpg \
  --classes 1 \
  --conf 0.25 \
  --nms-score 0.25 \
  --nms-iou 0.45 \
  --raw-sigmoid \
  --raw-xyxy \
  --batch 4 \
  --streams 2 \
  --auto-streams \
  --min-streams 1
```

> `--raw-sigmoid/--raw-xyxy` are needed only for raw output engines; packed NMS engines do not need them.

You can also build the example in `examples/yolo11/` separately.

> If the engine is **static batch** (e.g., built with `min=opt=max=16`), runtime must use the same batch. The example fills the batch by repeating the same image.

**Camera mode (for real-time inference and correctness testing; built-in camera is not a valid performance metric):**

```bash
./trt_yolo_example --engine ../../models/exports/best_fp16.engine --camera --camera-id 0
```

> Window overlay shows FPS/Infer/GPU/detection count; press `q` to exit. Use `--no-display` for inference only (FPS will increase).

**Video mode (run on video and save output):**

```bash
./trt_yolo_example \
  --engine ../../models/exports/best_fp16.engine \
  --video ../../test/videos/input.mp4 \
  --video-out ../../test/output/output.mp4 \
  --no-display
```

> Video mode defaults to batch=1. You can use `--video-batch 1/4/8/16` to aggregate frames for offline throughput testing (tail frames are dropped). FPS/Infer/GPU/Det count are drawn on frames.

---

## Performance Benchmarking (trtexec)

Run in `TRTInferX/build`. `--noDataTransfers` excludes H2D/D2H and measures pure GPU inference throughput:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=../../models/exports/best_int8_b128.engine \
  --shapes=images:128x3x640x640 \
  --warmUp=200 \
  --duration=10 \
  --noDataTransfers \
  --infStreams=1
```

End-to-end measurement with data transfers:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=../../models/exports/best_int8_b128.engine \
  --shapes=images:128x3x640x640 \
  --warmUp=200 \
  --duration=10 \
  --infStreams=1
```

Practical conclusions:
- Peak pure inference throughput: INT8 batch=128 (infStreams=1, NoDataTransfers) about 1858 FPS.
- Peak end-to-end throughput: INT8 batch=32 (infStreams=2, with transfers) about 1523 FPS.
- Lowest end-to-end latency: FP16 batch=1 (infStreams=1, with transfers) is more stable and lower.
- For large batch, transfer overhead is the main bottleneck; INT8 raw output D2H cost is significantly higher than FP16 packed output.
- infStreams=2 helps small batch clearly, and may provide no benefit or negative benefit for large batch with linear VRAM increase.

References:
- Performance report: [`docs/PerformanceReport.md`](docs/PerformanceReport.md)
- Performance log: [`docs/performance.txt`](docs/performance.txt)
- Runtime log: [`docs/log.txt`](docs/log.txt)

---

## Notes

- For **different GPU architectures**, rebuild `.engine` on the target device for best performance.
- INT8 PTQ engines can be generated by Ultralytics export scripts and loaded directly.
- If **GPU NMS** is needed, `nms=True` packed output is recommended (`output0`, like `[B, max_det, 6]`).
- Use `--streams` to set 2-4 CUDA streams for round-robin inference to improve throughput (default 1).
- Use `--auto-streams` to enable adaptive dynamic streams based on recent throughput.
- **Important**: **Engines exported by Python and the C++ runtime must use the same TensorRT**. If Python in Conda uses TensorRT while C++ links system `/lib/x86_64-linux-gnu` TensorRT, deserialization fails (magicTag mismatch). Fix by linking Conda TRT in C++ (`-DTRT_INCLUDE_DIR=... -DTRT_LIB_DIR=...`) or ensure Python uses system TRT for export.
- Reference build (C++ with Conda TRT):
  ```bash
  cmake .. \
    -DTRT_INCLUDE_DIR=$CONDA_PREFIX/include \
    -DTRT_LIB_DIR=$CONDA_PREFIX/lib
  cmake --build . -j
  ```
- FP16/INT8 choice: use FP16 to verify accuracy and flow; use INT8 for peak performance with calibration data, and adjust thresholds for your task.
- Cannot create NMS engine (workspace too small): at high batch, the NMS plugin needs larger scratch/workspace; current limit is too small and all tactics are skipped. Increase `setMemoryPoolLimit(kWORKSPACE, …)` in `TRTInferX/src/Inference.cpp` and rebuild.

---

## Common Pitfalls (Validated)

### 1) Engine deserialization failed (magicTag mismatch)

Symptom: `trtexec --loadEngine` or TRTInferX fails to load.  
Cause: engine and runtime TensorRT mismatch, or engine is corrupted.  
Fix: rebuild using **system TRT** from ONNX (example assumes `models/exports/best.onnx` exists).

FP16 validation flow (assume running under `TRTInferX/build`):

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=../../models/exports/best.onnx \
  --saveEngine=../../models/exports/best_fp16.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640

./trt_yolo_example \
  --engine ../../models/exports/best_fp16.engine \
  --image ../../test/images/coco128/images/000000000036.jpg \
  --classes 1 --conf 0.25 --batch 1 --streams 1 --no-display \
  --output ../../test/output/output.jpg
```

### 2) INT8 calibration failed (engine becomes 0 MiB)

Cause: ONNX with NMS often triggers shape errors during `trtexec` calibration.  
Recommendation: use `nms=False` ONNX and perform GPU NMS inside TRTInferX.

**Generate calibration input (calib.bin):**

```bash
cd ../../
$CONDA_PREFIX/bin/python - <<'PY'
import os, glob, cv2, numpy as np
src = "./test/images/coco128/images"
out = "./models/exports/calib.bin"
imgsz = 640
batch = 16
def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_shape - nw, new_shape - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im
files = sorted(glob.glob(os.path.join(src, "*")))[:batch]
assert len(files) == batch, "not enough calibration images"
buf = []
for f in files:
    im = cv2.imread(f)
    im = letterbox(im, imgsz)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2,0,1))
    buf.append(im)
arr = np.stack(buf, axis=0)
arr.tofile(out)
print("saved", out, arr.shape)
PY
cd TRTInferX/build
```

The following is equivalent and more convenient:

```
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

python TRTInferX/scripts/gen_calib_bin.py \
  --images test/images/coco128/images \
  --out models/exports/calib.bin \
  --imgsz 640 \
  --batch 4
```

> Note: `calib.bin` **data size must exactly match the trtexec input shape**.  
> For example `--min/opt/maxShapes=images:4x3x640x640` requires `calib.bin` to be **4×3×640×640** float32 data.  
> If you generated `calib.bin` with batch=1 and build a new INT8 engine with batch=4, you will see "Unexpected file size".  
> **For INT8 static batch models, if you change batch or imgsz, regenerate `calib.bin`.**

---

**nms=False + INT8 (recommended stable PTQ INT8 path)**

When using `trtexec` for INT8 calibration, **ONNX with NMS often fails**. Recommended flow:

1) Export **nms=False** ONNX  
2) Build INT8 engine with `trtexec`  
3) TRTInferX completes **decode + EfficientNMS** on GPU

Example:

```bash
PYTHON_BIN=$(which python) TRTInferX/scripts/export_onnx.py \
  --pt models/initial/yolo11n.pt \
  --out models/exports/best_raw.onnx \
  --imgsz 640 \
  --batch 16 \
  --dynamic
```

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/exports/best_raw.onnx \
  --saveEngine=models/exports/best_int8.engine \
  --int8 --fp16 \
  --loadInputs=images:models/exports/calib.bin \
  --calib=models/exports/trtexec.cache \
  --minShapes=images:16x3x640x640 \
  --optShapes=images:16x3x640x640 \
  --maxShapes=images:16x3x640x640
```

> TRTInferX will auto-detect `output0` as raw output and perform NMS on GPU.

**TRTInferX validation:**

```bash
./trt_yolo_example \
  --engine ../../models/exports/best_int8.engine \
  --image ../../test/images/coco128/images/000000000036.jpg \
  --classes 1 --conf 0.25 --batch 16 --streams 1 --no-display \
  --output ../../test/output/output.jpg
```

### 3) Dynamic batch engine

When `min=opt=max=16`, the engine is **static batch=16** and must run with `--batch 16`.  
If you need dynamic batch 1~16, rebuild:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=../../models/exports/best.onnx \
  --saveEngine=../../models/exports/best_int8.engine \
  --int8 --fp16 \
  --loadInputs=images:../../models/exports/calib.bin \
  --calib=../../models/exports/trtexec.cache \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:8x3x640x640 \
  --maxShapes=images:16x3x640x640
```

> Note: `min/opt/max` defines a range. If 4 is within 1–16, you can run `batch=4`.  
> `optShapes` is the preferred batch for optimization, not the only supported batch. If you mainly use batch=4, set `optShapes` to 4 for better performance.

> Performance note: static batch is usually faster (fixed tactic, better cache). Dynamic batch provides flexibility but **shape switching introduces extra overhead and noticeable jitter**. For maximum performance, prefer fixed batch.

### 4) Raw output format (target support)

> [!TIP]
> With `nms=False` raw output, both `4+cls` and `4+obj+cls` heads are supported; if obj exists, scores are fused as `score = obj * cls`, default **cxcywh** format.  
> For multi-class models, pass the correct `--classes` (e.g., COCO=80), otherwise fallback logs will appear and scores/filters become unreliable.

Common raw shapes (as `[B, C, N]`):

- `C=5`: single-class fused score (`cx,cy,w,h,score`)
- `C=6`: single-class `obj+cls` (`cx,cy,w,h,obj,cls`)
- `C=84`: COCO80 (`cx,cy,w,h,cls[80]`, no obj)
- `C=85`: COCO80 (`cx,cy,w,h,obj,cls[80]`)

> `[B, N, C]` layout is also supported; it is auto-detected and normalized.

```bash
./trt_yolo_example \
  --engine ../../models/exports/best_int8.engine \
  --image ../../test/images/coco128/images/000000000036.jpg \
  --classes 1 --conf 0.25 \
  --nms-score 0.25 --nms-iou 0.45 \
  --raw-sigmoid \
  --batch 16 --streams 1 --no-display \
  --output ../../test/output/output.jpg
```

### 5) Thresholds and scores (fix screen-full boxes)

- Ultralytics ONNX outputs usually already apply sigmoid. Default `raw_score_sigmoid=false` avoids turning 0 logits into 0.5, which causes too many candidates to pass the threshold.
- If you need probabilities, add `--raw-sigmoid`, but increase `--conf/--nms-score` to 0.4~0.5, otherwise you will see dense false boxes.
- With raw logits only, 0.08 maps to about 0.52 after sigmoid; INT8 quantization shifts logits slightly, so apply sigmoid and set a reasonable threshold.
- If you see many ~0.5-score boxes, it is because near-0 logits were sigmoid'ed. Two fixes:
  1) Do not sigmoid (omit `--raw-sigmoid`) and keep a low threshold.
  2) If sigmoid is required, raise thresholds, e.g. `--conf 0.5 --nms-score 0.5` (or higher).
- Accuracy stability depends on correct mapping between score definition and thresholds: sigmoid switch and thresholds must match. INT8 shifts scores slightly; sweep thresholds on a validation set (e.g., try 0.25/0.35/0.5 in sigmoid mode) and then fix them.
- If raw output is **score-only** (`raw channels=1 has_obj=0`, i.e., `cx,cy,w,h,score`), do not use `--raw-sigmoid`, or scores will compress to ~0.5 and flood the image. Use raw score thresholds (e.g., 0.08~0.15). If all boxes are filtered, lower the threshold (e.g., 0.03) or **use INT8 with `nms=True` (recommended)**.

> Timing: `infer(ms)` is end-to-end and may include sync or video write; `gpu(ms)` is CUDA event timing around TRT enqueue only. For pure inference, use `gpu(ms)`.

---

## Unified API

Upstream input:
- CPU/GPU dual paths, must explicitly declare stride as **byte stride** to avoid HWC/GPU confusion.
- Recommended fields: `mem{CPU/GPU}, data, width/height, stride_bytes, color{BGR/RGB/GRAY}, layout{HWC/CHW}, dtype{UINT8/FP16/FP32}, prep{LETTERBOX/RESIZE}, target_w/h`, plus `device_id`, `cuda_stream` for GPU, optional `timestamp_ms`, `roi`.

Downstream output:
- `Det { x1,y1,x2,y2 (original image coords), score, cls, batch, mask/pose optional }`.
- `PreprocInfo { scale, scale_x, scale_y, padw, padh, src_w, src_h }` for downstream coordinate mapping.

Engine config/options:
- `EngineConfig { engine_path, device, max_batch, streams, auto_streams, prep, target_w/h, out_mode(AUTO/PACKED_NMS/RAW_WITH_NMS/RAW_ONLY), nms_score, nms_iou }`
- `InferOptions { conf, iou, apply_sigmoid=false, max_det, stream_override, box_fmt(cxcywh/xyxy) }`
- Internal auto-selection for nms=True/False, packed/raw; static batch enforces alignment, dynamic batch uses setInputShape.
- Raw path NMS thresholds are fixed at load time (use `EngineConfig.nms_score/nms_iou`); `InferOptions` thresholds do not affect raw NMS.
- C++ external API (provided in `include/api.h`/`src/Api.cpp`):
  - `Api::load(cfg)` loads engine, `infer(batch, opt)` returns unified `Det`.
  - Current implementation supports CPU/GPU input (BGR/HWC/uint8); GPU path does CUDA preprocess from device ptr (zero-copy).
  - `LETTERBOX` keeps aspect ratio and pads; `RESIZE` stretches to input size with different coordinate semantics.
  - **GPU input notes (zero-copy GPU input critical for end-to-end speed)**:
    - Only supports `BGR/HWC/uint8`.
    - `stride_bytes` must be **byte** stride, otherwise GPU row access will be wrong.
    - If upstream uses another CUDA stream, pass `cuda_stream`, otherwise synchronize yourself.

---

<p align="center">
  <img src="docs/Horizon.png" width="200" alt="Horizon Team">
</p>

<div align="center">

Copyright © 2026 ROBOMASTER · 华北理工大学 HORIZON 战队 · 雷达组 - YAOYUZHUO<br/>
Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).<br/>
Use, modification, and redistribution are permitted under the terms of AGPL-3.0.<br/>
The complete corresponding source must be made available.<br/>
2026 年 01 月 08 日

</div>
