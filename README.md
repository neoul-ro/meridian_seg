# meridian_seg

카메라 컬러 이미지를 받아 **물체별로 나눈 라벨 이미지**를 내보내는 ROS 2 노드입니다.

FastSAM을 TensorRT로 돌립니다. 프롬프트 없이 화면에 있는 것을 전부 나누며,
"이게 무엇인지"는 판단하지 않습니다 — 그건 뒷단(CLIP 등)의 몫입니다.

```
/camera/rgb  ──►  seg_node  ──►  /segment_image  ──►  geobuilder_node  ──►  /instance_3d_set
 (rgb8)                          (mono8, 192x256)      + depth, camera_info
```

이 패키지에는 노드가 **둘** 들어 있습니다.

| 노드 | 하는 일 | 필수 여부 |
|---|---|---|
| `seg_node` | RGB → 라벨 이미지 (FastSAM + TensorRT) | 항상 |
| `geobuilder_node` | 라벨 이미지 + depth → 3D 인스턴스 | 선택 (`with_geobuilder:=true`) |

---

## 무엇이 나오는가

`/segment_image`는 **한 장의 흑백 이미지**인데, 밝기 값이 곧 물체 번호입니다.

| 픽셀 값 | 뜻 |
|---|---|
| `0` | 배경 (또는 측정 불가) |
| `1 ~ 255` | 물체 번호 (segment_id) |

같은 값을 가진 픽셀들이 한 물체입니다.

### ⚠️ 물체 번호는 매 프레임 새로 매겨집니다

1번 프레임의 `3`번과 2번 프레임의 `3`번은 **다른 물체일 수 있습니다.**
번호는 그 프레임 안에서만 유효합니다. 프레임을 넘어 물체를 추적하려면
뒷단의 `object_id`를 쓰셔야 합니다.

---

## 출력 크기가 카메라와 다릅니다

카메라가 640×480인데 **결과는 256×192**로 나옵니다. 정확히 **1/2.5**입니다.

```
카메라 640x480  →  모델 1024x1024  →  결과 256x192
                    (1.6배 확대)      (1/4로 축소 후 여백 제거)
```

그래서 결과 이미지의 좌표를 원본 카메라 좌표로 바꾸려면 **2.5를 곱하면**
됩니다.

### depth와 함께 3D로 만들 때

카메라 파라미터(intrinsics)를 같은 비율로 나눠 쓰시면 됩니다.

```python
fx, fy, cx, cy = fx/2.5, fy/2.5, cx/2.5, cy/2.5
```

> **여백(padding)은 이미 잘라냈습니다.** 그래서 `cy`에 `+32` 같은 보정을
> 더하면 **안 됩니다.** 더하면 1.5m 거리에서 20cm쯤 어긋납니다.

`geobuilder_node`는 이 환산을 **스스로 합니다.** depth와 camera_info는
카메라 원본 해상도 그대로 주시면 됩니다.

---

## 어떤 weight를 쓰는가

기본으로 쓰는 건 **`FastSAM-s-1024.engine`** 하나입니다 (`-s` = small, 1024×1024).

만들어지는 순서는 이렇습니다.

```
FastSAM-s.pt  ──export──►  FastSAM-s-1024.onnx  ──build──►  FastSAM-s-1024.engine
   24MB                          48MB                            27MB
 저장소에 있음                  저장소에 있음                  ★ 직접 만드셔야 함
```

### 저장소에서 받아지는 것 / 아닌 것

| 파일 | 저장소 | 이유 |
|---|---|---|
| `FastSAM-s.pt` | ✅ 들어 있음 | 24MB. 다른 크기로 다시 내보낼 때 필요 |
| `FastSAM-s-1024.onnx` | ✅ 들어 있음 | 48MB. 기계 사이 이식 가능 — 엔진의 재료 |
| `FastSAM-s-1024.engine` | ❌ **없음** | GPU 아키텍처 + TensorRT 버전 + CUDA 버전에 묶여 **옮겨도 안 돕니다** |

> **`.pt`와 `.onnx`는 `git clone`만 하면 그냥 따라옵니다.** 따로 받으실 것
> 없습니다. 만들어야 하는 건 `.engine` 하나뿐입니다.

### 엔진 만들기 (처음 한 번, 10~20분)

```bash
python3 scripts/build_engine.py --onnx weights/FastSAM-s-1024.onnx
```

`trtexec`로 직접 하셔도 됩니다.

```bash
cd weights
trtexec --onnx=FastSAM-s-1024.onnx --saveEngine=FastSAM-s-1024.engine --fp16
```

**젯슨에서도 똑같이 다시 만드셔야 합니다.** 데스크톱에서 만든 엔진은
동작하지 않습니다. 젯슨이면 전력 모드부터 올리세요.

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

### 큰 모델(`-x`)이 필요하면

품질 기준이 필요할 때 쓰는 `FastSAM.pt`(-x, 145MB)와 그 `.onnx`(289MB)는
**저장소에 넣을 수 없습니다.** GitHub는 파일 하나에 100MB가 한계입니다.
`-x`는 `-s`보다 **6배 느립니다** — 실시간으로는 못 씁니다.

필요하시면 upstream FastSAM 프로젝트에서 `FastSAM.pt`를 받아
`weights/`에 두고 직접 내보내십시오.

```bash
python3 scripts/export_fastsam_onnx.py --weights weights/FastSAM.pt --imgsz 1024
python3 scripts/build_engine.py --onnx weights/FastSAM-1024.onnx
ros2 launch meridian_seg seg.launch.py model_path:=weights/FastSAM-1024.engine
```

`export_fastsam_onnx.py`는 `~/FastSAM_official` 저장소가 있어야 돕니다
(vendored ultralytics 8.0.120을 씁니다). `-s`를 다른 크기로 다시 내보낼
때도 같은 방법입니다 — `--weights weights/FastSAM-s.pt`.

### 엔진을 어디서 찾는가

`model_path`를 비워두면 이 순서로 찾습니다.

1. `MERIDIAN_SEG_ENGINE` 환경변수
2. `share/meridian_seg/weights/FastSAM-s-1024.engine`
3. 소스 트리의 `weights/FastSAM-s-1024.engine`

셋 다 없으면 찾아본 경로를 전부 찍고 종료합니다. 조용히 실패하지 않습니다.

> `weights/`는 일부러 **설치하지 않습니다** (엔진 하나가 26~142MB).
> `--symlink-install`이면 3번이 그대로 잡히고, 아니면 `model_path`나
> `MERIDIAN_SEG_ENGINE`으로 지정하십시오.

---

## 실행

가장 흔한 경우 — RealSense를 따로 띄우고, 이 launch를 겁니다.

```bash
# 터미널 1: 카메라
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true enable_sync:=true \
  align_depth.enable:=true enable_rgbd:=false

# 터미널 2: 세그멘테이션
ros2 launch meridian_seg seg.launch.py
```

3D 인스턴스까지 필요하면:

```bash
ros2 launch meridian_seg seg.launch.py with_geobuilder:=true
```

노드 하나만 직접 띄우려면:

```bash
ros2 run meridian_seg seg_node --ros-args -p color_topic:=/camera/camera/color/image_raw
```

launch 파일은 **`seg.launch.py` 하나**입니다. geobuilder는 별도 파일이 아니라
`with_geobuilder` 인자로 켭니다. 밖에서 두 노드를 따로 켜고 끄고 싶으면
이 파일을 include하면서 그 인자를 넘기면 됩니다.

---

## 조절할 수 있는 값

`seg.launch.py`에 선언된 인자는 **launch 명령줄에서 `이름:=값`으로** 넘깁니다.

```bash
ros2 launch meridian_seg seg.launch.py conf_th:=0.6 iou_th:=0.8 with_geobuilder:=true
```

`ros2 run`으로 노드를 직접 띄울 때는 `--ros-args -p 이름:=값`입니다.
**두 경우 기본값이 다릅니다** — launch는 실사용에 맞게 조정한 값을 넘기고,
노드 자체 기본값은 그보다 보수적입니다. 아래 표에 둘 다 적었습니다.

### 잘라내기 — 물체를 얼마나 남길지

| 인자 | launch 기본 | 노드 기본 | 뜻 |
|---|---|---|---|
| `conf_th` | `0.5` | `0.4` | 이 점수 미만은 물체로 안 봅니다. **올리면** 확실한 것만 남고 물체 수가 줍니다. **내리면** 애매한 것까지 잡습니다 |
| `iou_th` | `0.7` | `0.9` | 상자 NMS 임계값. `mask_dedup`이 켜져 있으면 진짜 중복 판정은 마스크가 하므로 **0.7처럼 관대하게** 두는 게 짝입니다 |
| `area_min` | `64` | `16` | 192×256 격자에서 이보다 작은 조각은 버립니다. **올리면** 자잘한 파편이 사라집니다. 64는 원본 기준 400픽셀쯤 |
| `mask_dedup` | `true` | `true` | 상자는 안 겹치는데 픽셀은 겹치는 중복을 지웁니다. `iou_th`를 낮게 두는 것과 **같이** 쓰세요 |

> **처음 만지실 값은 `conf_th`입니다.** 물체가 너무 잘게 쪼개지면 올리고,
> 있어야 할 게 안 잡히면 내리세요. 0.3~0.7 밖으로는 잘 안 나갑니다.

### 토픽 이름

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `color_topic` | `/camera/camera/color/image_raw` | 입력 RGB. 계약 이름은 `/camera/rgb` |
| `segment_topic` | `/segment_image` | 출력 라벨 이미지. geobuilder가 이걸 받습니다 |
| `depth_topic` | `/camera/camera/aligned_depth_to_color/image_raw` | geobuilder 전용 |
| `camera_info_topic` | `/camera/camera/aligned_depth_to_color/camera_info` | geobuilder 전용 |

launch 기본값이 계약 이름(`/camera/rgb`)이 **아닌** 이유는, RealSense 드라이버가
자기 이름으로 발행하기 때문입니다. 드라이버를 계약 이름으로 remap해서 쓰신다면
이 인자들을 계약 이름으로 넘기세요.

### 속도 — 결과는 그대로, 시간만 달라집니다

| 인자 | launch 기본 | 노드 기본 | 뜻 |
|---|---|---|---|
| `postprocess_mode` | `graph_full` | `graph_full` | 아래 표 참고 |
| `lanes` | `56` | `56` | NMS 생존 마스크를 담는 고정 슬롯 수 |

#### postprocess_mode

**네 값 모두 결과가 같습니다.** 속도만 다릅니다.

| 값 | 설명 |
|---|---|
| `eager` | 가장 단순한 경로. **문제가 생기면 여기로 되돌리세요** |
| `fixed` | 중간 텐서를 고정 shape으로 유지해 CPU–GPU sync를 없앤 경로 |
| `graph` | `fixed` 위에 CUDA graph 캡처를 얹습니다 |
| `graph_full` | 전처리·추론까지 묶습니다. **기본값** |

실측(실시간 RealSense, 물체 39개): `eager` 9.5ms → `graph_full` 8.9ms.

#### lanes — 경고가 뜨면 올리는 값

```
NMS 생존자가 lanes(56)를 채웠습니다
```

이 경고가 뜨면 물체가 슬롯보다 많다는 뜻입니다. `lanes:=72`처럼 올리세요.

**공짜가 아닙니다.** 마스크 텐서가 `(lanes, 192, 256)`이라 메모리 트래픽이
정비례합니다. 72 → 56으로 줄이면 후처리가 1.048ms → 0.774ms(26% 감소)가
됩니다. 실측 최대 생존 수가 49라 **49 밑으로는 내리지 마세요.** 최대 255.

> 넘쳐도 **조용히 잘리지 않습니다.** overflow 카운터가 올라가고 경고가 나갑니다.

### geobuilder 쪽 (`with_geobuilder:=true`일 때만)

| 인자 | launch 기본 | 노드 기본 | 뜻 |
|---|---|---|---|
| `minimum_points` | `16` | `100` | 이보다 점이 적은 인스턴스는 버립니다 |
| `pose_topic` | `/pose` | `/pose` | world_T_camera 자세. **안 들어오면 카메라 좌표계로** 냅니다 |
| `depth_topic` | RealSense 실제 토픽 | `/camera/depth` | 카메라 원본 해상도 그대로 주세요 |
| `camera_info_topic` | RealSense 실제 토픽 | `/camera/info` | intrinsics는 노드가 환산합니다 |

> ⚠️ **`minimum_points`는 반드시 launch 기본값(16)을 쓰세요.** 노드 기본값
> `100`은 640×480 기준으로 잡힌 값입니다. depth가 192×256으로 6.25배
> 줄어들기 때문에, 100을 그대로 두면 원본 기준 625픽셀을 요구하는 셈이
> 되어 **작은 물체가 전부 탈락합니다.** 노드가 시작할 때 권고값을 로그로
> 알려줍니다.

---

## launch에 없는 값들

아래는 `seg.launch.py`가 넘기지 않습니다. 필요하면 `ros2 run`으로 직접
띄우면서 `-p`로 주거나, launch 파일에 추가하세요.

<details>
<summary>seg_node</summary>

| 파라미터 | 기본값 | 뜻 |
|---|---|---|
| `k1` | `384` | conf 통과 후보 고정 슬롯. 실측 max 268 |
| `nms_iters` | `6` | NMS 고정 반복 횟수. 실측 사슬 깊이 max 4 |
| `mask_dedup_th` | `0.7` | 마스크 겹침을 중복으로 볼 임계값 |
| `dedup_iters` | `4` | dedup 반복 횟수 |
| `dedup_fp32` | `true` | dedup을 fp32로. 끄면 빨라지지만 경계가 흔들립니다 |
| `compile_masks` | `true` | 마스크 조립을 `torch.compile`로 융합. triton이 없으면(젯슨) 자동으로 eager로 내려앉습니다 |

</details>

<details>
<summary>geobuilder_node</summary>

| 파라미터 | 기본값 | 뜻 |
|---|---|---|
| `pose_topic` | `/pose` | 카메라 위치. 없으면 항등 pose를 씁니다 |
| `output_topic` | `/instance_3d_set` | 3D 인스턴스 출력 |
| `erosion_px` | `1` | 마스크 경계에서 전경/배경 depth가 섞인 픽셀을 깎아냅니다. 192×256의 1픽셀 = 원본 2.5픽셀 |
| `voxel_size_m` | `0.0` | 0 이하면 다운샘플 안 함. `0.02`를 쓰면 점 수가 거리에 무관해집니다(같은 컵을 0.8m/1.5m에서: 105점 대 123점, 원본은 1147 대 320) |
| `minimum_depth_m` | `0.1` | 이보다 가까운 depth는 버립니다 |
| `maximum_depth_m` | `10.0` | 이보다 먼 depth는 버립니다 |
| `depth_scale_m` | `0.001` | depth 원값 → 미터. RealSense는 mm이라 0.001 |
| `sync_tolerance_ms` | `2.0` | depth와 라벨 이미지의 stamp 허용 오차 |
| `use_identity_pose_when_missing` | `true` | pose가 안 오면 항등으로 진행 |
| `verbose` | `false` | 프레임마다 INFO를 찍습니다. **켜면 느려집니다** (rclpy 로거가 호출마다 스택을 뒤집니다 — 프레임당 lstat 366회 실측) |

</details>

---

## 알아두면 좋은 것

**한 프레임에 물체는 최대 255개**입니다. 흑백 한 장에 담을 수 있는 값이
그만큼이기 때문입니다. 넘치면 작은 것부터 남기고 경고를 냅니다.

**겹침은 이미 정리돼 있습니다.** 한 픽셀은 한 물체에만 속하고, 겹쳤던
자리는 더 작은 물체가 이깁니다. 그래서 "누가 누구를 가렸는지"는 이
이미지로 알 수 없습니다.

**벽이나 바닥도 물체로 잡힙니다.** 무엇이든 나누는 모델이라 그렇습니다.
걸러내는 건 쓰는 쪽에서 하셔야 합니다.

**`header.stamp`은 카메라가 찍은 시각 그대로**입니다. depth나 pose와
짝을 맞출 때 이 값을 열쇠로 쓰시면 됩니다.

---

## 성능 (RTX 2070 기준)

한 프레임 **6.4ms**, 초당 약 156장 처리할 수 있습니다.

| 단계 | ms |
|---|---:|
| 전처리 | 0.32 |
| **신경망 추론** | **4.87** |
| 후처리 | 1.16 |
| 발행 | 0.04 |

시간의 76%는 신경망 자체가 씁니다. 프레임당 1000억 번쯤 계산하기 때문에
어쩔 수 없는 부분이고, 나머지 단계는 이미 충분히 얇습니다.

---

## scripts/

런타임에는 필요 없고, 엔진을 만들거나 검증할 때 쓰는 것들입니다.

| 파일 | 하는 일 |
|---|---|
| `export_fastsam_onnx.py` | 체크포인트(`.pt`) → `.onnx` 내보내기 |
| `build_engine.py` | `.onnx` → `.engine` 빌드 |
| `wrap_engine_for_ultralytics.py` | 엔진에 메타데이터 헤더를 붙임. **`build_engine.py`가 import해서 씁니다** — 따로 실행할 일은 없지만 지우면 엔진 빌드가 깨집니다 |

`seg_node`는 이 중 **아무것도 import하지 않습니다.** 엔진을 직접 호출하므로
`FastSAM_official` 저장소도, ultralytics도 런타임에 필요 없습니다. 반대로
`export_fastsam_onnx.py`는 `~/FastSAM_official`이 있어야 돕니다 — 모델을 다시
내보낼 때만 필요합니다.
