## Test Data Structure

Test data and inference results can be downloaded from:

🔗 https://github.com/BreCaspian/TRTInferX/releases

---

## Directory Structure

After downloading and extracting, the `test/` directory structure is as follows:

```text
test/
├── images/
│   └── coco128/
│       ├── images/
│       ├── labels/
│       ├── LICENSE
│       └── README.txt
├── output/
│   ├── kitti-inference-vid_fp16_b1.mp4
│   ├── kitti-inference-vid_fp16_b4.mp4
│   ├── kitti-inference-vid_fp16_b8.mp4
│   ├── kitti-inference-vid_fp16_b16.mp4
│   ├── kitti-inference-vid_fp16_b64.mp4
│   ├── kitti-inference-vid_fp16_b1_16_dynamic_b1.mp4
│   ├── kitti-inference-vid_fp16_b1_16_dynamic_b4.mp4
│   ├── kitti-inference-vid_fp16_b1_16_dynamic_b8.mp4
│   ├── kitti-inference-vid_fp16_b1_16_dynamic_b16.mp4
│   ├── kitti-inference-vid_int8_b1.mp4
│   ├── kitti-inference-vid_int8_b4.mp4
│   ├── kitti-inference-vid_int8_b8.mp4
│   ├── kitti-inference-vid_int8_b16.mp4
│   ├── kitti-inference-vid_int8_b32.mp4
│   ├── kitti-inference-vid_int8_b1_16_dynamic_b1.mp4
│   ├── kitti-inference-vid_int8_b1_16_dynamic_b4.mp4
│   ├── kitti-inference-vid_int8_b1_16_dynamic_b8.mp4
│   ├── kitti-inference-vid_int8_b1_16_dynamic_b16.mp4
│   └── output.jpg
└── videos/
    ├── anpr-demo-video.mp4
    └── kitti-inference-vid.mp4
```

------

## Notes

- `images/` contains the COCO128 dataset used for testing.
- `videos/` contains the original input videos for inference.
- `output/` contains inference result videos generated using different
  precisions (`FP16`, `INT8`) and batch sizes.
- Dynamic batch results correspond to batch sizes in the range `1–16`.

These files are provided for testing, benchmarking, and result comparison.

---

<p align="center">
  <img src="../docs/Horizon.png" width="200" alt="Horizon Team">
</p>

<div align="center">

Copyright © 2026 ROBOMASTER · 华北理工大学 HORIZON 战队 · 雷达组 - YAOYUZHUO<br/>
Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).<br/>
Use, modification, and redistribution are permitted under the terms of AGPL-3.0.<br/>
The complete corresponding source must be made available.<br/>
2026 年 01 月 08 日

</div>