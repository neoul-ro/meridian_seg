#!/usr/bin/env python3
"""FastSAM: 카메라 수신 -> 전처리 -> TensorRT 엔진 -> 후처리 -> SegmentImage.

전처리부터 후처리까지 이 한 파일에서 끝난다. FastSAM은 RGB만 먹으므로 depth는
이 파일이 다루지 않는다.

변수명은 Segmentor 아키텍처 문서를 그대로 따른다: rgb_raw / img_rgb /
img_preprocessed / pred_raw / proto / pred_filter / pred_conf / boxes_xyxy /
nms_idx / pred / mask_coeff / bbox_proto / masks. 문서와 코드를 나란히 놓고
읽을 수 있게 하는 것이 목적이다.

ultralytics를 쓰지 않는 이유
---------------------------
8.0.120의 예측기에 미리 letterbox한 텐서를 source로 주면 preprocess가 letterbox와
/255를 건너뛰고(정상), retina_masks 경로가 orig_img.shape[:2]를 참조하는데 그 orig_img가
(1, 3, 1024, 1024) 텐서라서 (1, 3)이 나온다. 즉 전처리를 직접 하는 것과 예측기를
유지하는 것은 함께 성립하지 않는다. 그래서 엔진을 직접 호출한다. 결과적으로
ultralytics도, FastSAM_official 저장소도 런타임에 필요 없다.

출력 공간
--------
마스크는 proto 공간에서 letterbox 패딩을 잘라낸 (192, 256)이다. 원본 (480, 640)을
정확히 1/2.5로 줄인 것과 같은 격자이므로, back-projection 하는 쪽은 intrinsics를
(fx/2.5, fy/2.5, cx/2.5, cy/2.5)로 환산하면 된다 -- 패딩을 이미 제거했으므로
cy에 +32 오프셋을 더하면 안 된다.
"""
from __future__ import annotations

import array
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
import tensorrt as trt
import torch
import torch.nn.functional as F
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image as ImageMsg
from torchvision.ops import box_convert, nms

ENGINE_FILENAME = "FastSAM-s-1024.engine"
ENGINE_ENV_VAR = "MERIDIAN_SEG_ENGINE"


def engine_candidates() -> list[Path]:
    """엔진 파일을 찾을 위치를 우선순위 순으로 돌려준다.

    엔진은 GPU 아키텍처와 TensorRT 버전에 묶여 이식되지 않으므로 저장소에
    포함하지 않는다. 각 머신이 .onnx에서 직접 빌드한 파일을 여기서 찾는다.

    한 곳만 보면 안 되는 이유: --symlink-install이면 __file__이 소스 트리를
    가리키지만, 평범한 colcon build면 install 트리를 가리킨다. 후자에는
    weights/가 없다.
    """
    found: list[Path] = []

    env = os.environ.get(ENGINE_ENV_VAR, "").strip()
    if env:
        found.append(Path(env).expanduser())

    try:
        share = Path(get_package_share_directory("meridian_seg"))
        found.append(share / "weights" / ENGINE_FILENAME)
    except (PackageNotFoundError, KeyError):
        pass

    # 소스 트리 (--symlink-install 또는 저장소에서 직접 실행)
    found.append(Path(__file__).resolve().parents[1] / "weights" / ENGINE_FILENAME)
    return found

MODEL_SIZE = 1024
PROTO_STRIDE = 4  # FastSAM 설계 상수: proto는 MODEL_SIZE/4
PROTO_CHANNELS = 32  # 공유 기저 장수. 모델 설계 상수

# ultralytics LetterBox와 같은 회색. 0~255 공간의 값이라 정규화 뒤에 패딩할 때는
# 반드시 255로 나눠서 넣어야 한다(그냥 114를 넣으면 초백색이 된다).
PAD_VALUE_U8 = 114.0

# mono8 label 한 장에 담을 수 있는 마스크 수. 넘치면 조용히 자르지 않고 경고한다.
MAX_SEGMENTS = 255


def assemble_masks(
    logit: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    columns: torch.Tensor,
    rows: torch.Tensor,
    lane_alive: torch.Tensor,
) -> torch.Tensor:
    """마스크 이진화 + bbox 크롭 + 죽은 레인 제거를 한 식으로 묶는다.

    eager로 두면 연산자마다 별도 커널이 (LANES, 192, 256) bool 중간 텐서를
    메모리에 썼다가 다시 읽는다. graph는 launch를 없애줄 뿐 이 왕복은 그대로
    남으므로, 남은 비용의 본체가 여기다. torch.compile로 한 커널에 융합하면
    픽셀당 계산이 레지스터에서 끝나고 최종 bool만 한 번 쓰인다.
    """
    return (
        (logit > 0)
        & (columns >= x1)
        & (columns < x2)
        & (rows >= y1)
        & (rows < y2)
        & lane_alive[:, None, None]
    )

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
}


class Geometry:
    """한 소스 해상도에 대한 letterbox / proto 상수 묶음.

    MODEL_SIZE와 PROTO_STRIDE 두 개에서만 파생된다. 640x480이면 scale=1.6,
    pad=128, proto 크롭 (32, 192)이 자동으로 나오고, 해상도가 바뀌어도 손으로
    맞출 곳이 없다.
    """

    def __init__(self, src_h: int, src_w: int) -> None:
        self.src_h = src_h
        self.src_w = src_w

        # resize가 아니라 scale 보존: 긴 축이 MODEL_SIZE에 닿게 맞춘다
        self.scale = min(MODEL_SIZE / src_h, MODEL_SIZE / src_w)
        self.new_h = int(round(src_h * self.scale))
        self.new_w = int(round(src_w * self.scale))

        pad_h = MODEL_SIZE - self.new_h
        pad_w = MODEL_SIZE - self.new_w
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left

        # proto 공간 = letterbox를 PROTO_STRIDE로 나눈 것
        self.proto_size = MODEL_SIZE // PROTO_STRIDE
        self.crop_top = self.pad_top // PROTO_STRIDE
        self.crop_left = self.pad_left // PROTO_STRIDE
        self.mask_h = self.new_h // PROTO_STRIDE
        self.mask_w = self.new_w // PROTO_STRIDE

    def describe(self) -> str:
        return (
            f"({self.src_h}, {self.src_w}) x{self.scale:g} -> "
            f"({self.new_h}, {self.new_w}) + pad top {self.pad_top} "
            f"left {self.pad_left} -> ({MODEL_SIZE}, {MODEL_SIZE})  |  "
            f"proto {self.proto_size} crop top {self.crop_top} -> "
            f"mask ({self.mask_h}, {self.mask_w})"
        )


class SegNode(Node):
    def __init__(self) -> None:
        super().__init__("seg_node")

        self.declare_parameter("color_topic", "/camera/rgb")
        self.declare_parameter("segment_topic", "/segment_image")
        self.declare_parameter("model_path", "")
        self.declare_parameter("conf_th", 0.4)
        self.declare_parameter("iou_th", 0.9)
        self.declare_parameter("area_min", 16)

        # 후처리 경로. "eager"가 기존 동작이고 "fixed"는 모든 중간 텐서를 고정
        # shape으로 유지해 CPU-GPU sync를 없앤 경로다. 결과는 같아야 한다.
        # 손해가 나면 파라미터 하나로 되돌릴 수 있게 둘 다 남긴다.
        # 상수는 RealSense 실장면 200프레임 실측값 기준이다. 넘치면 조용히
        # 잘리는 게 아니라 overflow 카운터가 올라가고 경고가 나간다.
        #
        #   항목        실측 max   채택   여유
        #   K 통과        268      384    +43%
        #   NMS 생존       49       56    +14%
        #   사슬 깊이       4        6    +50%
        #
        # ★ lanes가 비용을 지배한다. 마스크 텐서가 (lanes, 192, 256)이라
        #   메모리 트래픽이 정비례한다. 72 -> 56만으로 후처리가 1.048 ->
        #   0.774ms가 된다(26%). 다만 여유가 얇아지므로, 물건이 많은 장면에서
        #   경고가 뜨면 올려야 한다. 49보다 낮추면 안 된다.
        self.declare_parameter("postprocess_mode", "graph_full")
        self.declare_parameter("k1", 384)         # conf 후보 고정 슬롯
        self.declare_parameter("lanes", 56)       # NMS 생존 고정 슬롯
        self.declare_parameter("nms_iters", 6)    # 고정 반복 횟수

        # ② mask dedup. box NMS(iou_th)는 상자만 보므로 용량 제어만 맡기고,
        # 진짜 중복 판정은 조립된 마스크의 픽셀 겹침으로 한다. 이때 iou_th는
        # 관대하게(0.7) 두는 것이 스펙의 짝이다.
        self.declare_parameter("mask_dedup", True)
        self.declare_parameter("mask_dedup_th", 0.7)
        self.declare_parameter("dedup_iters", 4)
        self.declare_parameter("dedup_fp32", True)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.conf_th = float(self.get_parameter("conf_th").value)
        self.iou_th = float(self.get_parameter("iou_th").value)
        self.area_min = int(self.get_parameter("area_min").value)

        self.postprocess_mode = str(
            self.get_parameter("postprocess_mode").value
        ).strip().lower()
        if self.postprocess_mode not in (
            "eager", "fixed", "graph", "graph_full"
        ):
            raise ValueError(
                f"postprocess_mode는 eager / fixed / graph / graph_full여야 "
                f"합니다: {self.postprocess_mode}"
            )
        self.static_rgb = None
        self.static_input = None
        # graph는 fixed 위에 CUDA graph 캡처를 얹은 것이다. 첫 프레임에서 잡는다.
        self.graph: torch.cuda.CUDAGraph | None = None
        self.g_label = self.g_status = None

        # ④ 마스크 조립 체인을 한 커널로 융합한다.
        #
        # 실제 컴파일은 첫 호출 때 일어나므로 여기서 torch.compile()을 감싸는
        # try는 아무것도 못 잡는다. 준비는 prepare_assemble()에서 첫 실행까지
        # 시켜보고 거기서 실패를 잡는다 -- Jetson에 triton이 없을 때 노드가
        # 죽지 않고 eager로 내려앉게 하기 위해서다.
        #
        # eager 모드는 assemble을 쓰지 않으므로 컴파일하지 않는다. 켜두고
        # eager로 돌리면 3.7초를 그냥 버리게 된다.
        self.declare_parameter("compile_masks", True)
        self.compile_masks = bool(self.get_parameter("compile_masks").value)
        self.assemble = assemble_masks
        self.assemble_ready = False
        self.k1 = int(self.get_parameter("k1").value)
        self.lanes = int(self.get_parameter("lanes").value)
        self.nms_iters = int(self.get_parameter("nms_iters").value)
        self.mask_dedup = bool(self.get_parameter("mask_dedup").value)
        self.mask_dedup_th = float(self.get_parameter("mask_dedup_th").value)
        self.dedup_iters = int(self.get_parameter("dedup_iters").value)
        self.dedup_fp32 = bool(self.get_parameter("dedup_fp32").value)

        # 고정 슬롯을 넘긴 프레임 수. 넘치면 조용히 잘리는 게 아니라 여기 쌓이고
        # 경고가 나가야 한다 -- 무손실이 우선이기 때문이다.
        self.overflow_k1 = 0
        self.overflow_lanes = 0
        self.overflow_nms = 0

        if self.lanes > MAX_SEGMENTS:
            raise ValueError(
                f"lanes({self.lanes})는 MAX_SEGMENTS({MAX_SEGMENTS})를 넘을 수 "
                "없습니다. 넘으면 label map 단계에서 잘립니다."
            )

        # 비워두면 알려진 위치들을 순서대로 찾는다. 지정하면 그 경로만 쓴다.
        configured = str(self.get_parameter("model_path").value).strip()

        if configured:
            self.model_path = Path(configured).expanduser()
            if not self.model_path.exists():
                raise FileNotFoundError(f"엔진을 찾을 수 없습니다: {self.model_path}")
        else:
            tried = engine_candidates()
            self.model_path = next((p for p in tried if p.exists()), None)
            if self.model_path is None:
                raise FileNotFoundError(
                    "엔진을 찾을 수 없습니다. 찾아본 곳:\n  "
                    + "\n  ".join(str(p) for p in tried)
                    + f"\n\nmodel_path 파라미터로 직접 지정하거나 {ENGINE_ENV_VAR}"
                    " 환경변수를 설정하십시오. 엔진은 GPU 아키텍처와 TensorRT"
                    " 버전에 묶이므로 각 머신에서 .onnx로부터 빌드해야 합니다."
                )
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT 엔진은 CUDA가 필요합니다")
        self.device = "cuda"

        self.load_engine(self.model_path)

        # 소스 해상도는 첫 프레임에서 알아내 캐시한다. 하드코딩하지 않는다.
        self.geometry: Geometry | None = None

        # 다른 코드가 꺼내 쓸 수 있게 최근 프레임의 중간값을 들고 있는다.
        # masks / conf / bbox_proto / area는 행 순서가 서로 대응한다.
        self.img_rgb: torch.Tensor | None = None
        self.img_preprocessed: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.conf: torch.Tensor | None = None
        self.bbox_proto: torch.Tensor | None = None
        self.area: torch.Tensor | None = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.segment_pub = self.create_publisher(
            ImageMsg, str(self.get_parameter("segment_topic").value), qos
        )
        self.create_subscription(ImageMsg, self.color_topic, self.on_rgb, qos)

        self.frame_count = 0
        self.get_logger().info(
            f"SAM ready: {self.color_topic} -> "
            f"{self.get_parameter('segment_topic').value}, "
            f"conf_th={self.conf_th}, iou_th={self.iou_th}, "
            f"area_min={self.area_min}, 겹침=작은 마스크 우선, "
            + (f"후처리={self.postprocess_mode}(k1={self.k1}, "
               f"lanes={self.lanes}, nms_iters={self.nms_iters}"
               + (f", dedup {'fp32' if self.dedup_fp32 else 'fp16'}"
                  if self.mask_dedup else "") + ")"
               if self.postprocess_mode != "eager" else "후처리=eager")
        )

    # ------------------------------------------------------------------
    # 엔진
    # ------------------------------------------------------------------

    def load_engine(self, path: Path) -> None:
        """엔진을 역직렬화하고 출력 버퍼를 한 번만 할당한다.

        build_engine.py가 ultralytics용 metadata 헤더(4바이트 길이 + JSON)를 앞에
        붙여두므로 그만큼 건너뛴다. 헤더가 없는 순수 엔진도 받아들인다.
        """
        raw = path.read_bytes()
        head = int.from_bytes(raw[:4], byteorder="little", signed=True)
        if 0 < head < 4096 and raw[4:5] == b"{":
            metadata = json.loads(raw[4:4 + head].decode("utf-8"))
            blob = raw[4 + head:]
            self.get_logger().info(
                f"엔진: {path.name} ({len(raw) / 1e6:.1f} MB), "
                f"imgsz={metadata.get('imgsz')}"
            )
        else:
            blob = raw
            self.get_logger().info(
                f"엔진: {path.name} ({len(raw) / 1e6:.1f} MB), metadata 헤더 없음"
            )

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(
                f"엔진 역직렬화 실패: {path}. 다른 GPU 아키텍처나 다른 TensorRT "
                "버전에서 빌드된 파일일 수 있습니다(엔진은 이식되지 않습니다)."
            )
        # rclpy Node에 읽기 전용 context 속성이 있어 self.context는 쓸 수 없다.
        self.trt_context = self.engine.create_execution_context()

        # enqueueV3를 기본 스트림에 걸면 TensorRT가 자체 동기화를 끼워넣고 그걸
        # 경고한다. 전용 스트림을 쓰고 이벤트로만 순서를 맞춘다.
        self.stream = torch.cuda.Stream()

        self.input_name = None
        self.pred_raw_name = self.proto_name = None
        self.pred_raw_buffer = self.proto_buffer = None

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = shape
                self.input_dtype = dtype
                continue

            buffer = torch.empty(shape, dtype=dtype, device=self.device)
            # 이름에 의존하지 않고 rank로 가른다: 후보표 pred_raw는 (1, 37, 21504),
            # 프로토타입 proto는 (1, 32, 256, 256)이다.
            if len(shape) == 3:
                self.pred_raw_name, self.pred_raw_buffer = name, buffer
            elif len(shape) == 4:
                self.proto_name, self.proto_buffer = name, buffer
            else:
                raise RuntimeError(f"예상 못한 출력 rank: {name} {shape}")
            self.trt_context.set_tensor_address(name, int(buffer.data_ptr()))

        missing = [
            label
            for label, value in (
                ("input", self.input_name),
                ("pred_raw", self.pred_raw_name),
                ("proto", self.proto_name),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(f"엔진 I/O를 찾지 못했습니다: {missing}")

        if self.input_shape[2] != MODEL_SIZE or self.input_shape[3] != MODEL_SIZE:
            raise RuntimeError(
                f"이 노드는 {MODEL_SIZE}x{MODEL_SIZE} 엔진을 전제합니다. "
                f"엔진 입력은 {self.input_shape}입니다."
            )

        self.get_logger().info(
            f"엔진 I/O: {self.input_name}{self.input_shape} {self.input_dtype} -> "
            f"pred_raw{tuple(self.pred_raw_buffer.shape)}, "
            f"proto{tuple(self.proto_buffer.shape)}"
        )
        # --fp16은 커널 정밀도이고 I/O dtype은 ONNX 그대로다. 하드코딩하지 않고
        # 엔진이 실제로 요구하는 dtype으로 전처리 출력을 맞춘다.
        self.engine_dtype = self.input_dtype

    def forward(
        self, img_preprocessed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(1,3,1024,1024) -> (pred_raw, proto). 둘 다 GPU에 남는다."""
        if tuple(img_preprocessed.shape) != self.input_shape:
            raise ValueError(
                f"입력 shape {tuple(img_preprocessed.shape)}이 "
                f"엔진의 {self.input_shape}과 다릅니다"
            )

        self.trt_context.set_tensor_address(
            self.input_name, int(img_preprocessed.data_ptr())
        )
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        if not self.trt_context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("execute_async_v3 실패")
        current.wait_stream(self.stream)
        return self.pred_raw_buffer, self.proto_buffer

    # ------------------------------------------------------------------
    # 전처리 — ROS 없이 호출/테스트할 수 있도록 staticmethod로 둔다
    # ------------------------------------------------------------------

    @staticmethod
    def image_to_array(msg: ImageMsg) -> np.ndarray:
        """msg.data를 복사 한 번으로 (H, W, 3) uint8로 만든다. step을 존중한다."""
        if msg.encoding != "rgb8":
            raise ValueError(f"rgb8을 기대했는데 {msg.encoding}이 왔습니다")
        flat = np.frombuffer(msg.data, dtype=np.uint8)
        # step은 바이트 단위다. 행마다 뒤에 패딩이 붙어 있을 수 있다.
        rows = flat.reshape(msg.height, msg.step)
        return rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)

    @staticmethod
    def to_img_rgb(rgb_raw: np.ndarray, device: str) -> torch.Tensor:
        """(H, W, 3) uint8 -> (1, 3, H, W) fp32 [0, 1] on device."""
        # ascontiguousarray는 여기서 사실상 no-op다. rclpy가 주는 msg.data는
        # array.array('B')라 frombuffer 뷰도 쓰기 가능하고, step == width*3이면
        # 이미 C-연속이라 같은 객체가 그대로 돌아온다(실측 0.0002ms).
        # 행 패딩이 있는 카메라에서만 실제 복사가 일어나고, 그때는 필요한 복사다.
        # ★ "read-only라 복사가 필요하다"는 예전 주석은 틀렸다. 그 말을 믿고
        #   .copy()를 넣으면 프레임마다 900KB 복사가 새로 생긴다.
        tensor = torch.from_numpy(np.ascontiguousarray(rgb_raw))
        tensor = tensor.to(device, non_blocking=True)
        # uint8을 올린 뒤 GPU에서 나누는 편이 fp32를 올리는 것보다 전송량이 4배 적다
        return tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

    @staticmethod
    def letterbox(
        img_rgb: torch.Tensor, geometry: Geometry, dtype: torch.dtype
    ) -> torch.Tensor:
        """(1,3,H,W) fp32 [0,1] -> (1,3,1024,1024). resize가 아니라 scale+pad."""
        scaled = F.interpolate(
            img_rgb,
            size=(geometry.new_h, geometry.new_w),
            mode="bilinear",
            align_corners=False,
        )
        # img_rgb가 이미 [0,1]로 정규화돼 있으므로 패딩 값도 정규화해서 넣는다.
        return F.pad(
            scaled,
            (
                geometry.pad_left,
                geometry.pad_right,
                geometry.pad_top,
                geometry.pad_bottom,
            ),
            value=PAD_VALUE_U8 / 255.0,
        ).to(dtype).contiguous()

    # ------------------------------------------------------------------
    # 후처리 — 전부 GPU
    # ------------------------------------------------------------------

    @staticmethod
    def masks_to_label_map(masks: torch.Tensor) -> torch.Tensor:
        """(N, H, W) bool -> (H, W) uint8. 겹치면 면적이 작은 마스크가 이긴다.

        면적 내림차순으로 1..N을 부여하면 가장 작은 마스크가 가장 큰 id를 갖는다.
        그 상태에서 amax를 취하면 겹친 픽셀에서 큰 id, 곧 작은 마스크가 남는다.
        파이썬 루프로 순차 대입하는 것과 정확히 같은 결과이고 커널은 한 번이다.
        """
        if masks.shape[0] == 0:
            return torch.zeros(
                masks.shape[1:], dtype=torch.uint8, device=masks.device
            )

        areas = masks.flatten(1).sum(1)
        # 오름차순으로 정렬해 앞에서 MAX_SEGMENTS개를 남긴다. 잘라야 할 때는
        # 작은 것을 우선 보존한다(겹침 규칙과 같은 방향).
        ascending = torch.argsort(areas, stable=True)[:MAX_SEGMENTS]
        descending = ascending.flip(0)

        segment_ids = torch.arange(
            1, len(descending) + 1, dtype=torch.uint8, device=masks.device
        ).view(-1, 1, 1)
        return (masks[descending].to(torch.uint8) * segment_ids).amax(0)

    def clear_outputs(self, height: int, width: int) -> None:
        """masks / conf / bbox_proto / area를 빈 텐서로 초기화한다."""
        self.masks = torch.zeros(
            (0, height, width), dtype=torch.bool, device=self.device
        )
        self.conf = torch.zeros(0, device=self.device)
        self.bbox_proto = torch.zeros((0, 4), device=self.device)
        self.area = torch.zeros(0, dtype=torch.long, device=self.device)

    def fixed_core(
        self, pred_raw: torch.Tensor, proto: torch.Tensor, geometry: Geometry
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """postprocess_fixed의 GPU 부분. sync가 하나도 없어야 한다.

        CUDA graph는 캡처 중 CPU가 GPU 값을 읽는 것을 허용하지 않는다. 그래서
        경고·개수 추출을 전부 GPU 텐서로 내보내고, 판단은 replay 뒤 바깥에서
        한다. 파이썬 분기도 GPU 값에 의존해서는 안 된다.

        반환 (label_map, count, k1_full, lanes_full) — 뒤 셋은 GPU 스칼라다.
        """
        height, width = geometry.mask_h, geometry.mask_w

        table = pred_raw[0].transpose(0, 1).float()      # (21504, 37)
        k1 = min(self.k1, table.shape[0])
        top_score, top_idx = table[:, 4].topk(k1)
        cand = table[top_idx]                             # (K1, 37)
        valid = top_score > self.conf_th

        boxes_xyxy = box_convert(cand[:, :4], "cxcywh", "xyxy")

        # 고정 반복 NMS. 조기 종료(torch.equal)는 sync라서 여기서는 못 쓴다.
        # 사슬 깊이 실측 max가 4이고 nms_iters 기본값이 8이라 여유가 있다.
        n = boxes_xyxy.shape[0]
        area_b = ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0)
                  * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0))
        lt = torch.max(boxes_xyxy[:, None, :2], boxes_xyxy[None, :, :2])
        rb = torch.min(boxes_xyxy[:, None, 2:], boxes_xyxy[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        iou = inter / (area_b[:, None] + area_b[None, :] - inter + 1e-9)
        idx = torch.arange(n, device=boxes_xyxy.device)
        outranks = (top_score[None, :] > top_score[:, None]) | (
            (top_score[None, :] == top_score[:, None])
            & (idx[None, :] < idx[:, None])
        )
        conflict = (iou > self.iou_th) & outranks

        alive = valid.clone()
        for _ in range(self.nms_iters):
            killed = (conflict & alive[None, :]).any(dim=1)
            alive = valid & ~killed

        lane_score = torch.where(
            alive, top_score, torch.full_like(top_score, float("-inf"))
        )
        lanes = min(self.lanes, k1)
        _, lane_idx = lane_score.topk(lanes)
        lane_alive = alive[lane_idx]

        pred = cand[lane_idx]
        conf = pred[:, 4]
        mask_coeff = pred[:, 5:]

        bbox_proto = boxes_xyxy[lane_idx] / PROTO_STRIDE
        # ★ [0, 2] 같은 리스트 인덱싱은 CPU 인덱스 텐서를 만들어 GPU로 복사하므로
        #   graph 캡처가 거부한다. 열 0,2 / 1,3은 스트라이드 슬라이싱으로 집는다.
        bbox_proto[:, 0::2] -= geometry.crop_left
        bbox_proto[:, 1::2] -= geometry.crop_top

        proto_cropped = proto[0][
            :,
            geometry.crop_top:geometry.crop_top + height,
            geometry.crop_left:geometry.crop_left + width,
        ].contiguous()
        masks = (
            mask_coeff @ proto_cropped.float().view(PROTO_CHANNELS, -1)
        ).view(-1, height, width)

        x1, y1, x2, y2 = torch.chunk(bbox_proto[:, :, None], 4, dim=1)
        columns = torch.arange(
            width, device=masks.device, dtype=bbox_proto.dtype
        ).view(1, 1, width)
        rows = torch.arange(
            height, device=masks.device, dtype=bbox_proto.dtype
        ).view(1, height, 1)
        masks = self.assemble(
            masks, x1, y1, x2, y2, columns, rows, lane_alive
        )

        area = masks.sum(dim=(1, 2))
        keep = (area >= self.area_min) & lane_alive

        # ② 최종 심판 = mask dedup. box NMS는 상자만 보므로 상자는 안 겹치는데
        # 픽셀은 겹치는 중복이 남는다. 실제 마스크 IoU로 한 번 더 거른다.
        # box 임계를 관대하게(0.7) 두고 용량 제어만 맡긴 뒤, 진짜 판정을 여기서
        # 하는 것이 스펙의 의도다.
        if self.mask_dedup:
            # ★ fp16은 2048을 넘는 정수를 정확히 표현하지 못한다. 마스크 픽셀
            #   수가 수천~수만이면 inter/union 합산이 뭉개져 경계 IoU 판정이
            #   뒤집힌다(실장면 1/100 프레임에서 관측). 기본은 fp32로 둔다.
            flat = masks.view(masks.shape[0], -1).to(
                torch.float32 if self.dedup_fp32 else torch.float16
            )
            inter = flat @ flat.T                     # (LANES, LANES)
            asum = flat.sum(1)
            miou = inter / (asum[:, None] + asum[None, :] - inter + 1e-6)

            # conf 높은 쪽이 이긴다. 동점은 레인 인덱스가 앞선 쪽.
            lane_conf = conf
            lane_i = torch.arange(
                masks.shape[0], device=masks.device
            )
            outranks_m = (lane_conf[None, :] > lane_conf[:, None]) | (
                (lane_conf[None, :] == lane_conf[:, None])
                & (lane_i[None, :] < lane_i[:, None])
            )
            conflict_m = (miou > self.mask_dedup_th) & outranks_m

            # NMS와 같은 고정점. 여기서도 매 라운드 keep에서 다시 계산해야
            # 사슬이 제대로 전파된다.
            alive_m = keep.clone()
            for _ in range(self.dedup_iters):
                killed_m = (conflict_m & alive_m[None, :]).any(dim=1)
                alive_m = keep & ~killed_m
            keep = alive_m

        masks = masks & keep[:, None, None]
        area = torch.where(keep, area, torch.zeros_like(area))

        # label map. lanes <= MAX_SEGMENTS이므로 절단이 없고, 죽은 레인은 면적
        # 0이라 오름차순 맨 앞 -> flip 뒤 맨 뒤로 가서 살아있는 마스크의
        # segment_id 배정을 밀지 않는다.
        ascending = torch.argsort(area, stable=True)
        descending = ascending.flip(0)
        segment_ids = torch.arange(
            1, lanes + 1, dtype=torch.uint8, device=masks.device
        ).view(-1, 1, 1)
        label = (masks[descending].to(torch.uint8) * segment_ids).amax(0)

        # ★ 개수와 overflow 신호를 텐서 하나로 묶는다. 따로 두면 int()/bool()
        #   호출마다 별도 D2H가 일어나 프레임당 sync가 3회가 된다. 묶어서
        #   한 번에 읽으면 1회다.
        #     [0] 살아남은 개수  [1] k1 가득참  [2] lanes 가득참
        status = torch.stack([
            keep.sum().to(torch.int32),
            valid.all().to(torch.int32),
            lane_alive.all().to(torch.int32),
        ])

        # masks/conf/bbox_proto/area/keep도 같이 낸다 -- postprocess_fixed가
        # 같은 함수를 쓰게 해서 구현이 둘로 갈라지지 않도록 하기 위해서다.
        return (label, status, masks, conf, bbox_proto, area, keep)

    def prepare_assemble(self, geometry: Geometry) -> None:
        """compile을 켜고 실제로 한 번 돌려본다. 실패하면 eager로 되돌린다.

        torch.compile은 첫 호출에서 컴파일하므로, 여기서 실행까지 시켜봐야
        triton 부재 같은 실패를 잡을 수 있다. 실패해도 결과는 같은 식이므로
        정확도에는 영향이 없고 속도만 원래대로 돌아간다.
        """
        if self.assemble_ready or not self.compile_masks:
            self.assemble_ready = True
            return

        lanes, h, w = self.lanes, geometry.mask_h, geometry.mask_w
        dev = self.device
        probe = (
            torch.zeros((lanes, h, w), device=dev),
            *[torch.zeros((lanes, 1, 1), device=dev) for _ in range(4)],
            torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, w),
            torch.arange(h, device=dev, dtype=torch.float32).view(1, h, 1),
            torch.ones(lanes, dtype=torch.bool, device=dev),
        )
        candidate = torch.compile(
            assemble_masks, fullgraph=True, dynamic=False
        )
        try:
            started = time.perf_counter()
            got = candidate(*probe)
            expected = assemble_masks(*probe)
            if not torch.equal(got, expected):
                raise RuntimeError("compile 결과가 eager와 다릅니다")
            torch.cuda.synchronize()
            self.assemble = candidate
            self.get_logger().info(
                f"마스크 조립 커널 융합 완료 "
                f"({(time.perf_counter() - started):.1f}초)"
            )
        except Exception as exc:      # noqa: BLE001
            self.get_logger().warn(
                f"torch.compile 실패, eager로 진행합니다 (결과는 동일, "
                f"속도만 원래대로): {type(exc).__name__}: {exc}"
            )
            self.assemble = assemble_masks
        self.assemble_ready = True

    def build_postprocess_graph(self, geometry: Geometry) -> None:
        """fixed_core를 CUDA graph로 캡처한다.

        입력은 engine이 매 프레임 같은 주소(pred_raw_buffer / proto_buffer)에
        쓰므로 그대로 쓰면 되고, 출력은 캡처가 잡아준 고정 텐서를 replay가
        제자리 갱신한다. 프레임당 CPU 개입은 replay 1회 + count 읽기 1회다.
        """
        # compile은 캡처 전에 끝나 있어야 한다. 캡처 중 컴파일이 일어나면
        # 그 과정까지 graph에 말려든다.
        self.prepare_assemble(geometry)

        # 캡처 전 warmup. 첫 호출에서 잡히는 lazy 초기화나 workspace 할당이
        # graph 안에 들어가면 안 된다.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.fixed_core(
                    self.pred_raw_buffer, self.proto_buffer, geometry
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            (self.g_label, self.g_status,
             self.g_masks, self.g_conf, self.g_bbox,
             self.g_area, self.g_keep) = self.fixed_core(
                self.pred_raw_buffer, self.proto_buffer, geometry
            )
        self.get_logger().info(
            f"CUDA graph 캡처 완료: 후처리 (k1={self.k1}, lanes={self.lanes}, "
            f"nms_iters={self.nms_iters})"
        )

    def full_core(self, geometry: Geometry) -> tuple:
        """전처리 + engine + 후처리를 한 덩어리로. sync가 없어야 한다.

        입력은 self.static_rgb (고정 주소)이고, engine 입력도 self.static_input
        에 고정해 둔다. TensorRT는 shape과 주소가 고정이면 캡처할 수 있다.
        """
        img = (
            self.static_rgb.permute(2, 0, 1).unsqueeze(0)
            .float().div(255.0)
        )
        x = self.letterbox(img, geometry, self.input_dtype)
        self.static_input.copy_(x)

        # 주소는 캡처 전에 이미 set_tensor_address로 박아뒀다. 여기서는 실행만
        # 한다. 캡처 중에는 현재 스트림이 곧 캡처 스트림이다.
        if not self.trt_context.execute_async_v3(
            torch.cuda.current_stream().cuda_stream
        ):
            raise RuntimeError("execute_async_v3 실패 (graph 캡처 중)")

        return self.fixed_core(
            self.pred_raw_buffer, self.proto_buffer, geometry
        )

    def build_full_graph(self, geometry: Geometry) -> None:
        """full_core를 캡처한다. 전처리·engine·후처리가 replay 1회로 끝난다."""
        h, w = geometry.src_h, geometry.src_w
        self.static_rgb = torch.zeros(
            (h, w, 3), dtype=torch.uint8, device=self.device
        )
        self.static_input = torch.zeros(
            self.input_shape, dtype=self.input_dtype, device=self.device
        )
        # engine 입력 주소를 고정한다. 캡처 이후에는 절대 바꾸지 않는다.
        self.trt_context.set_tensor_address(
            self.input_name, int(self.static_input.data_ptr())
        )
        self.prepare_assemble(geometry)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.full_core(geometry)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            (self.g_label, self.g_status,
             self.g_masks, self.g_conf, self.g_bbox,
             self.g_area, self.g_keep) = self.full_core(geometry)
        self.get_logger().info(
            f"CUDA graph 캡처 완료: 전처리+engine+후처리 "
            f"(k1={self.k1}, lanes={self.lanes})"
        )

    def fall_back_to_eager(self, why: str) -> None:
        """graph 캡처가 안 되는 환경이면 조용히 죽지 말고 eager로 돌아간다.

        graph_full이 기본값이므로, TensorRT 버전이나 드라이버 조합에 따라 캡처가
        실패할 수 있다. 결과는 eager와 같으므로 속도만 원래대로 돌아갈 뿐이다.
        """
        self.get_logger().warn(
            f"CUDA graph 캡처 실패, eager로 진행합니다 (결과는 동일, 속도만 "
            f"원래대로): {why}"
        )
        self.postprocess_mode = "eager"
        self.graph = None

    def process_graph_full(
        self, rgb: np.ndarray, geometry: Geometry
    ) -> tuple[torch.Tensor, int]:
        """원본 RGB를 고정 버퍼에 올리고 replay 한 번으로 끝낸다."""
        if self.graph is None:
            try:
                self.build_full_graph(geometry)
            except Exception as exc:      # noqa: BLE001
                self.fall_back_to_eager(f"{type(exc).__name__}: {exc}")
                x = self.letterbox(
                    self.to_img_rgb(rgb, self.device), geometry,
                    self.engine_dtype)
                pred_raw, proto = self.forward(x)
                return self.postprocess(pred_raw, proto, geometry)

        src = torch.from_numpy(np.ascontiguousarray(rgb))
        self.static_rgb.copy_(src, non_blocking=True)
        self.graph.replay()

        count = self.read_status()      # 프레임당 유일한 sync
        return self.g_label.clone(), count

    def read_status(self) -> int:
        """graph 출력 status를 한 번의 D2H로 읽고 overflow를 알린다.

        이것이 프레임당 유일한 CPU-GPU 동기화 지점이다. 값을 따로따로 읽으면
        읽는 횟수만큼 D2H가 늘어난다.
        """
        host = self.g_status.cpu()          # <-- 프레임당 sync 1회
        count = int(host[0])
        if int(host[1]):
            self.overflow_k1 += 1
            self.get_logger().warn(
                f"conf 통과 후보가 k1({self.k1})을 채웠습니다. 잘렸을 수 "
                f"있습니다 (누적 {self.overflow_k1}프레임). k1을 올리세요.",
                throttle_duration_sec=5.0,
            )
        if int(host[2]):
            self.overflow_lanes += 1
            self.get_logger().warn(
                f"NMS 생존자가 lanes({self.lanes})를 채웠습니다. 잘렸을 수 "
                f"있습니다 (누적 {self.overflow_lanes}프레임). lanes를 올리세요.",
                throttle_duration_sec=5.0,
            )
        return count

    def postprocess_graph(
        self, pred_raw: torch.Tensor, proto: torch.Tensor, geometry: Geometry
    ) -> tuple[torch.Tensor, int]:
        """캡처된 graph를 replay한다. pred_raw/proto는 주소가 같아야 한다."""
        if self.graph is None:
            try:
                self.build_postprocess_graph(geometry)
            except Exception as exc:      # noqa: BLE001
                self.fall_back_to_eager(f"{type(exc).__name__}: {exc}")
                return self.postprocess(pred_raw, proto, geometry)

        self.graph.replay()
        count = self.read_status()      # 프레임당 유일한 sync

        # graph 출력은 다음 replay가 덮어쓴다. 소비자가 프레임을 넘겨 들고
        # 있어도 안전하도록 복사해서 내보낸다.
        return self.g_label.clone(), count

    def postprocess_fixed(
        self, pred_raw: torch.Tensor, proto: torch.Tensor, geometry: Geometry
    ) -> tuple[torch.Tensor, int]:
        """fixed_core를 graph 없이 그대로 실행한다.

        graph 모드와 같은 함수를 쓴다. 예전에는 같은 로직을 두 벌 들고 있었는데,
        mask dedup을 한쪽에만 넣는 실수가 바로 나왔다. 구현은 하나여야 한다.
        """
        height, width = geometry.mask_h, geometry.mask_w
        self.clear_outputs(height, width)
        self.prepare_assemble(geometry)

        (label, status, masks, conf, bbox_proto,
         area, keep) = self.fixed_core(pred_raw, proto, geometry)
        host = status.cpu()          # graph 없는 경로도 읽기는 1회
        count_t = int(host[0])
        k1_full, lanes_full = int(host[1]), int(host[2])

        if bool(k1_full):
            self.overflow_k1 += 1
            self.get_logger().warn(
                f"conf 통과 후보가 k1({self.k1})을 채웠습니다. 잘렸을 수 "
                f"있습니다 (누적 {self.overflow_k1}프레임). k1을 올리세요.",
                throttle_duration_sec=5.0,
            )
        if bool(lanes_full):
            self.overflow_lanes += 1
            self.get_logger().warn(
                f"NMS 생존자가 lanes({self.lanes})를 채웠습니다. 잘렸을 수 "
                f"있습니다 (누적 {self.overflow_lanes}프레임). lanes를 올리세요.",
                throttle_duration_sec=5.0,
            )

        count = int(count_t)
        if count == 0:
            return torch.zeros(
                (height, width), dtype=torch.uint8, device=self.device
            ), 0

        compact = keep.nonzero(as_tuple=True)[0]
        self.masks = masks[compact]
        self.conf = conf[compact]
        self.bbox_proto = bbox_proto[compact]
        self.area = area[compact]
        return label, count

    def postprocess(
        self, pred_raw: torch.Tensor, proto: torch.Tensor, geometry: Geometry
    ) -> tuple[torch.Tensor, int]:
        """엔진 출력 -> (192, 256) uint8 label map, 생존 객체 수."""
        height, width = geometry.mask_h, geometry.mask_w
        empty = torch.zeros(
            (height, width), dtype=torch.uint8, device=self.device
        )

        # 조기 반환하는 프레임에서 이전 프레임 값이 남아 있으면 소비자가 낡은
        # 마스크를 최신으로 착각한다. 먼저 비운다.
        self.clear_outputs(height, width)

        # Table 형식 만들기: 1 row = 후보 1개
        pred_raw = pred_raw[0].transpose(0, 1).float()  # (21504, 37)

        # conf 1차 필터: 21504 -> 수백
        pred_filter = pred_raw[:, 4] > self.conf_th  # (21504,) bool
        pred_conf = pred_raw[pred_filter]  # (K, 37)
        if pred_conf.shape[0] == 0:
            return empty, 0

        # bbox 형식 변환 + NMS. 좌표는 1024x1024 공간이다.
        boxes_xyxy = box_convert(pred_conf[:, :4], "cxcywh", "xyxy")  # (K, 4)
        nms_idx = nms(boxes_xyxy, pred_conf[:, 4], self.iou_th)  # (N,)

        pred = pred_conf[nms_idx]  # (N, 37)
        conf = pred[:, 4]  # (N,) -- 문서의 pred[:, :4]는 bbox 열이라 오타로 본다
        mask_coeff = pred[:, 5:]  # (N, 32)

        # 1024 -> proto(256) 좌표로 내리고, letterbox 패딩만큼 원점을 옮긴다.
        # 아래에서 proto를 잘라내므로 박스도 같은 원점을 써야 한다.
        bbox_proto = boxes_xyxy[nms_idx] / PROTO_STRIDE  # (N, 4)
        bbox_proto[:, [0, 2]] -= geometry.crop_left
        bbox_proto[:, [1, 3]] -= geometry.crop_top

        # Mask 생성: 공유 기저 32장의 weighted mean.
        # proto에서 패딩 영역을 먼저 떼어 원본 비율 격자로 만든다.
        proto_cropped = proto[0][
            :,
            geometry.crop_top:geometry.crop_top + height,
            geometry.crop_left:geometry.crop_left + width,
        ].contiguous()  # (32, 192, 256)
        masks = (
            mask_coeff @ proto_cropped.float().view(PROTO_CHANNELS, -1)
        ).view(-1, height, width)  # (N, 192, 256) logit

        # bbox 밖은 0으로 + 이진화. sigmoid는 계산하지 않는다 --
        # sigmoid(x) > 0.5 는 x > 0 과 동치이고, (N, 49152) sigmoid는 순수 낭비다.
        x1, y1, x2, y2 = torch.chunk(bbox_proto[:, :, None], 4, dim=1)
        columns = torch.arange(
            width, device=masks.device, dtype=bbox_proto.dtype
        ).view(1, 1, width)
        rows = torch.arange(
            height, device=masks.device, dtype=bbox_proto.dtype
        ).view(1, height, 1)
        inside = (columns >= x1) & (columns < x2) & (rows >= y1) & (rows < y2)
        masks = (masks > 0) & inside  # (N, 192, 256) bool

        # Final filtering: 너무 작은 조각은 버린다. masks만 남기지 않고 conf ·
        # bbox_proto · area도 같은 boolean으로 걸러 행 대응을 유지한다 -- k번째
        # 마스크와 k번째 conf가 같은 객체를 가리켜야 나중에 쓸 수 있다.
        area = masks.sum(dim=(1, 2))  # (N,)
        size_ok = area >= self.area_min
        masks = masks[size_ok]
        conf = conf[size_ok]
        bbox_proto = bbox_proto[size_ok]
        area = area[size_ok]

        # 다른 코드가 꺼내 쓸 수 있게 들고 있는다. 행 순서는 서로 대응한다.
        self.masks = masks
        self.conf = conf
        self.bbox_proto = bbox_proto
        self.area = area

        if masks.shape[0] == 0:
            return empty, 0

        if masks.shape[0] > MAX_SEGMENTS:
            self.get_logger().warn(
                f"{masks.shape[0]}개 마스크는 mono8 한 장({MAX_SEGMENTS})을 넘습니다. "
                "면적이 작은 것부터 보존하고 큰 것을 버립니다.",
                throttle_duration_sec=2.0,
            )
        return self.masks_to_label_map(masks), int(masks.shape[0])

    # ------------------------------------------------------------------
    # ROS
    # ------------------------------------------------------------------

    def on_rgb(self, msg: ImageMsg) -> None:
        started = time.perf_counter()
        try:
            if self.geometry is None:
                self.geometry = Geometry(msg.height, msg.width)
                self.get_logger().info(self.geometry.describe())

            rgb_raw = self.image_to_array(msg)  # (480, 640, 3) uint8

            if self.postprocess_mode == "graph_full":
                # 전처리부터 후처리까지 replay 1회. CPU가 하는 일은 원본을
                # 고정 버퍼에 올리는 것과 count를 읽는 것뿐이다.
                with torch.inference_mode():
                    labels, count = self.process_graph_full(
                        rgb_raw, self.geometry
                    )
            else:
                self.img_rgb = self.to_img_rgb(rgb_raw, self.device)
                self.img_preprocessed = self.letterbox(
                    self.img_rgb, self.geometry, self.engine_dtype
                )
                with torch.inference_mode():
                    pred_raw, proto = self.forward(self.img_preprocessed)
                    run = {
                        "graph": self.postprocess_graph,
                        "fixed": self.postprocess_fixed,
                    }.get(self.postprocess_mode, self.postprocess)
                    labels, count = run(pred_raw, proto, self.geometry)

            self.publish_segments(labels, msg.header)

            self.frame_count += 1
            if self.frame_count == 1 or self.frame_count % 10 == 0:
                elapsed = (time.perf_counter() - started) * 1000.0
                self.get_logger().info(
                    f"frame={self.frame_count}, objects={count}, "
                    f"segments={count}, "
                    f"size={labels.shape[1]}x{labels.shape[0]}, {elapsed:.1f} ms, "
                    f"mode={self.postprocess_mode}"
                )
        except Exception as exc:
            self.get_logger().error(
                f"SAM 추론 실패: {type(exc).__name__}: {exc}"
            )

    def publish_segments(self, labels: torch.Tensor, header) -> None:
        """mono8 [H, W]. 픽셀 0은 background, 1..255가 frame-local segment_id다.

        header.stamp은 source camera frame의 capture time이어야 하므로 입력
        메시지의 헤더를 그대로 물려준다. 이것이 frame key다.
        """
        labels_u8 = np.ascontiguousarray(
            labels.detach().to("cpu", torch.uint8).numpy()
        )

        # image.data에는 array.array("B")를 넣는다. bytes를 넣으면 rclpy가 시퀀스
        # 검증 경로를 타면서 48KiB짜리 라벨맵 하나에 4.3ms를 쓴다(실측 458배 차이).
        holder = array.array("B")
        holder.frombytes(memoryview(labels_u8).cast("B"))

        image = ImageMsg()
        image.header = header
        image.height, image.width = labels_u8.shape
        image.encoding = "mono8"
        image.is_bigendian = 0
        image.step = image.width
        image.data = holder
        self.segment_pub.publish(image)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
