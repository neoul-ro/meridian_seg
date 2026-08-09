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
trt10_compat도, FastSAM_official 저장소도 런타임에 필요 없다.

출력 공간
--------
마스크는 proto 공간에서 letterbox 패딩을 잘라낸 (192, 256)이다. 원본 (480, 640)을
정확히 1/2.5로 줄인 것과 같은 격자이므로, back-projection 하는 쪽은 intrinsics를
(fx/2.5, fy/2.5, cx/2.5, cy/2.5)로 환산하면 된다 -- 패딩을 이미 제거했으므로
cy에 +32 오프셋을 더하면 안 된다.
"""
from __future__ import annotations

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


class SamNode(Node):
    def __init__(self) -> None:
        super().__init__("sam_node")

        self.declare_parameter("color_topic", "/camera/rgb")
        self.declare_parameter("segment_topic", "/segment_image")
        self.declare_parameter("model_path", "")
        self.declare_parameter("conf_th", 0.4)
        self.declare_parameter("iou_th", 0.9)
        self.declare_parameter("area_min", 16)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.conf_th = float(self.get_parameter("conf_th").value)
        self.iou_th = float(self.get_parameter("iou_th").value)
        self.area_min = int(self.get_parameter("area_min").value)

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
        self.get_logger().info(f"엔진: {self.model_path}")
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
            f"area_min={self.area_min}, 겹침=작은 마스크 우선"
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
        # frombuffer 결과는 read-only라서 from_numpy 전에 복사가 필요하다.
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
            self.img_rgb = self.to_img_rgb(rgb_raw, self.device)
            self.img_preprocessed = self.letterbox(
                self.img_rgb, self.geometry, self.engine_dtype
            )

            with torch.inference_mode():
                pred_raw, proto = self.forward(self.img_preprocessed)
                labels, count = self.postprocess(pred_raw, proto, self.geometry)

            self.publish_segments(labels, msg.header)

            self.frame_count += 1
            if self.frame_count == 1 or self.frame_count % 10 == 0:
                elapsed = (time.perf_counter() - started) * 1000.0
                self.get_logger().info(
                    f"frame={self.frame_count}, objects={count}, "
                    f"segments={int(labels.max())}, "
                    f"size={labels.shape[1]}x{labels.shape[0]}, {elapsed:.1f} ms"
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
        array = labels.detach().to("cpu", torch.uint8).numpy()

        image = ImageMsg()
        image.header = header
        image.height, image.width = array.shape
        image.encoding = "mono8"
        image.is_bigendian = 0
        image.step = image.width
        image.data = np.ascontiguousarray(array).tobytes()
        self.segment_pub.publish(image)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
