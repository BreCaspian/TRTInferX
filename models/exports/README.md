## Download

Prebuilt models and inference artifacts are available at:

🔗 https://github.com/BreCaspian/TRTInferX/releases

---

## Model Files Structure

After downloading and extracting, the model directory structure is as follows:

```text
models/
├── exports/
│   ├── best_fp16_b1.engine
│   ├── best_fp16_b4.engine
│   ├── best_fp16_b8.engine
│   ├── best_fp16_b16.engine
│   ├── best_fp16_b32.engine
│   ├── best_fp16_b64.engine
│   ├── best_fp16_b128.engine
│   ├── best_fp16_b1_16_dynamic.engine
│   ├── best_int8_b1.engine
│   ├── best_int8_b4.engine
│   ├── best_int8_b8.engine
│   ├── best_int8_b16.engine
│   ├── best_int8_b32.engine
│   ├── best_int8_b64.engine
│   ├── best_int8_b128.engine
│   ├── best_int8_b1_16_dynamic.engine
│   ├── best.onnx
│   ├── best_raw.onnx
│   └── calib.bin
└── initial/
    ├── yolo11n.onnx
    └── yolo11n.pt
```


------

## Notes

- `exports/` contains exported ONNX models and TensorRT engines for different
  precisions (`FP16`, `INT8`) and batch sizes.
- Dynamic batch engines support batch sizes in the range `1–16`.
- `calib.bin` is required for INT8 inference.
- `initial/` contains the original training checkpoints and ONNX model.

For model export or regeneration, please refer to the instructions in the README.

---

<p align="center">
  <img src="../../docs/Horizon.png" width="200" alt="Horizon Team">
</p>

<div align="center">

Copyright © 2026 ROBOMASTER · 华北理工大学 HORIZON 战队 · 雷达组 - YAOYUZHUO<br/>
Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).<br/>
Use, modification, and redistribution are permitted under the terms of AGPL-3.0.<br/>
The complete corresponding source must be made available.<br/>
2026 年 01 月 08 日

</div>