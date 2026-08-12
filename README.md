# meridian_seg

FastSAM segmentation for the Meridian perception pipeline. Labels each colour
frame with a class-agnostic instance id map. Promptless, no text conditioning,
no depth.

Preprocessing, TensorRT inference and postprocessing all happen in
`meridian_seg/seg_node.py`; ultralytics is not used at runtime.

## I/O

| Direction | Topic | Type | Notes |
| --- | --- | --- | --- |
| subscribe | `/camera/rgb` | `sensor_msgs/Image` | `encoding == "rgb8"` |
| publish | `/segment_image` | `sensor_msgs/Image` | `encoding == "mono8"` |

`header` is copied from the source colour frame, so `header.stamp` is the
capture time that keys the frame and pairs it with `/camera/depth`,
`/camera/info` and `/pose`.

Pixel `0` is background. `1..255` are frame-local segment ids, matching the
`uint8` width of `meridian_msgs/SegmentRef.segment_id`. A segment id is an index
within one frame and carries no relation to the same value in the next frame —
stable identity is `object_id` only.

## Output geometry

The label image is **not** the camera resolution. It is the model's prototype
grid with the letterbox padding removed:

```
camera 480 x 640  ──x1.6──►  model 1024 x 1024  (768 x 1024 content, 128 rows padded)
model 1024  ──/4──►  prototype 256 x 256
prototype 256  ──crop 32 rows──►  label 192 x 256
```

Both scalings compose to a single factor, and the aspect ratio is preserved
(`192:256 = 480:640`):

```
camera → label :  1.6 / 4 = 1 / 2.5
```

A consumer that needs source-image coordinates multiplies by **2.5**. A consumer
that back-projects the label image with depth scales the intrinsics instead:

```
(fx, fy, cx, cy) → (fx/2.5, fy/2.5, cx/2.5, cy/2.5)
```

The padding is already removed, so **no row offset is added to `cy`**.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `color_topic` | `/camera/rgb` | Input topic |
| `segment_topic` | `/segment_image` | Output topic |
| `model_path` | `""` | Engine path; empty means search (see below) |
| `conf_th` | `0.4` | Confidence threshold on the candidate table |
| `iou_th` | `0.9` | Box-IoU threshold for NMS |
| `area_min` | `16` | Minimum mask area in the label grid |
| `postprocess_mode` | `eager` | `eager` / `fixed` / `graph` / `graph_full` |
| `mask_dedup` | `false` | Drop duplicates by mask-pixel IoU |
| `mask_dedup_th` | `0.7` | Mask-IoU threshold for the above |
| `dedup_fp32` | `true` | `false` uses fp16: ~2x cheaper, rarely differs |
| `k1` | `384` | Fixed slot count for confidence candidates |
| `lanes` | `56` | Fixed slot count for NMS survivors |
| `nms_iters` | `6` | Fixed-iteration count for NMS |
| `compile_masks` | `true` | Fuse the mask-assembly chain with `torch.compile` |

`iou_th` governs **box** overlap only. Two masks can overlap far more than their
boxes suggest. `mask_dedup` is the answer to that: it compares the assembled
masks pixel-wise and drops the lower-confidence member of each duplicate pair.
Pair it with a permissive `iou_th` (0.7) so box NMS only caps capacity and the
mask stage makes the real decision.

## Postprocess modes

The default `eager` path is the reference implementation. The others keep every
intermediate tensor at a fixed size so no CPU-GPU synchronisation is needed,
which is what makes CUDA graph capture possible.

| Mode | What it does |
| --- | --- |
| `eager` | Reference. Variable shapes, one sync per boolean index and NMS |
| `fixed` | Fixed shapes only. **Slower than `eager`** — see below |
| `graph` | `fixed` plus CUDA graph capture of postprocessing |
| `graph_full` | Also captures preprocessing and the engine |

Measured on an RTX 2070, 640x480 input, ~21 segments:

| Mode | Postprocess | Kernel launches | Syncs |
| --- | ---: | ---: | ---: |
| `eager` | 1.22 ms | 143 | 10.1 |
| `fixed` | 2.27 ms | 65 | 3.1 |
| `graph` | 0.96 ms | 65 | 1.1 |
| `graph_full` | 0.96 ms | 1 | 1.1 |

`fixed` on its own is slower because fixed shapes always do the maximum amount
of work: 56 lanes are assembled even when 21 segments are present. It only pays
off once CUDA graph removes the launch overhead. **`fixed` is a prerequisite,
not an optimisation.**

The fixed-slot constants come from measured maxima on live scenes:
confidence candidates 268, NMS survivors 49, greedy chain depth 4. Overflowing
a slot does not silently truncate — a counter increments and a warning is
logged, so raise `lanes` if you see one.

`compile_masks` costs 2-3 seconds at startup and saves ~0.3 ms per frame, so it
only pays off for runs longer than a few minutes. It needs `triton`; without it
the node warns once and falls back to eager evaluation rather than failing.
Set `TORCHINDUCTOR_COMPILE_THREADS=1` if compilation hangs.

## Prerequisites

Not declared in `package.xml` because no rosdep key resolves correctly on both
x86_64 desktop and Jetson — install them yourself:

- PyTorch and torchvision (CUDA build; on Jetson use the JetPack wheels)
- TensorRT Python bindings (`import tensorrt`)
- A CUDA-capable GPU — the node raises at startup without one

## Building the engine

`weights/` ships the checkpoint and the exported ONNX:

```
weights/FastSAM-s.pt           22.7 MiB   upstream checkpoint
weights/FastSAM-s-1024.onnx    45.4 MiB   portable, build engines from this
```

The `.engine` is **not** in the repository and never should be. It is compiled
against one GPU architecture, TensorRT version and CUDA version, so an engine
built on a desktop GPU will fail to deserialize on Jetson and vice versa. Build
one on each machine — it takes a minute or two:

```bash
# needs trtexec on PATH
python3 scripts/build_engine.py --onnx weights/FastSAM-s-1024.onnx --fp16
```

Re-exporting the ONNX is only needed for a different input size, and requires a
`FastSAM_official` clone for the vendored ultralytics:

```bash
python3 scripts/export_fastsam_onnx.py --weights weights/FastSAM-s.pt --imgsz 1024
```

The node locates the engine in this order, stopping at the first hit:

1. the `model_path` parameter, if set
2. `$MERIDIAN_SEG_ENGINE`
3. `<package share>/weights/FastSAM-s-1024.engine`
4. `<source tree>/weights/FastSAM-s-1024.engine`

If none exist it raises and lists every path it tried.

## Run

```bash
ros2 run meridian_seg seg_node

# or point it at an engine explicitly
ros2 run meridian_seg seg_node --ros-args -p model_path:=/path/to/FastSAM-s-1024.engine
```
