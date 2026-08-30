#!/usr/bin/env python3
import os

# ★ numpy를 import하기 전에 실행돼야 한다. numpy/BLAS는 import 시점에 스레드
#   수를 정해버리므로, 그 뒤에 바꾸면 아무 효과가 없다.
#
# numpy는 기본적으로 CPU 코어를 전부 써서 배열 연산을 병렬화한다. 그런데 여기서
# 다루는 배열은 192x256이라 병렬화 이득이 거의 없고, 대신 같은 기계에서 도는
# seg 노드가 GPU에 일을 던질 CPU 여유를 빼앗는다.
#
# 실측(실시간 RealSense, 두 노드 동시): sam 계산 13.5ms -> 8.8ms.
#
# 환경에서 이미 지정했다면 그 값을 존중한다(setdefault).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import array  # noqa: E402
import math  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
from copy import deepcopy  # noqa: E402

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from geometry_msgs.msg import Point, PoseWithCovarianceStamped  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from meridian_msgs.msg import Instance3DSet  # noqa: E402


def key(stamp):
    return int(stamp.sec), int(stamp.nanosec)


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = x / n, y / n, z / n, w / n

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def segment_id_to_rgb(segment_id):
    """같은 segment_id에 항상 같은 RGB 색을 부여한다."""
    sid = int(segment_id)
    r = (sid * 53 + 80) % 256
    g = (sid * 97 + 120) % 256
    b = (sid * 193 + 160) % 256
    return int(r), int(g), int(b)


def pack_rgb_float(r, g, b):
    """PointCloud2 FLOAT32 rgb 필드용 packed RGB 값."""
    rgb_uint32 = (
        ((int(r) & 0xFF) << 16)
        | ((int(g) & 0xFF) << 8)
        | (int(b) & 0xFF)
    )
    return struct.unpack("<f", struct.pack("<I", rgb_uint32))[0]


# label map이 mono8이므로 segment_id는 0..255다. bincount 길이 기준.
MAX_SEGMENT_ID = 255

XYZRGB_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
]


def pack_cloud_bytes(header, xyzrgb):
    """(N, 4) float32 버퍼를 그대로 PointCloud2 바이트로 넘긴다.

    msg.data에는 반드시 array.array("B")를 넣어야 한다. bytes를 넣으면 rclpy가
    시퀀스 검증 경로를 타면서 오히려 2배 느려진다(실측 30ms vs 64ms).
    필드 값은 sensor_msgs_py의 create_cloud가 채우는 것과 동일하게 맞춘다.
    """
    n = xyzrgb.shape[0]

    holder = array.array("B")
    holder.frombytes(memoryview(xyzrgb).cast("B"))

    cloud = PointCloud2(
        header=header,
        height=1,
        width=n,
        fields=XYZRGB_FIELDS,
        is_bigendian=sys.byteorder != "little",
        point_step=16,
        row_step=16 * n,
        is_dense=False,
    )
    cloud.data = holder
    return cloud


class GeobuilderNode(Node):
    def __init__(self):
        super().__init__("geobuilder_node")

        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("segment_topic", "/segment_image")
        self.declare_parameter("camera_info_topic", "/camera/info")
        self.declare_parameter("pose_topic", "/pose")
        self.declare_parameter("output_topic", "/instance_3d_set")
        self.declare_parameter(
            "debug_cloud_topic", "/meridian/debug/instances_cloud"
        )
        self.declare_parameter(
            "centroid_marker_topic", "/meridian/debug/instance_centroids"
        )
        self.declare_parameter("minimum_points", 100)
        # 마스크 경계에서 전경/배경 depth가 섞인 픽셀(mixed pixel)을 버린다.
        # 192x256 격자에서 1픽셀은 원본 480x640의 2.5픽셀에 해당한다.
        self.declare_parameter("erosion_px", 1)
        # 0 이하면 다운샘플을 하지 않고 모든 점을 낸다. 2cm를 쓰면 점 수가 물체의
        # 물리적 크기에 묶여 거리에 무관해진다(같은 컵을 0.8m/1.5m에서 봤을 때
        # 105점 대 123점, 원본은 1147점 대 320점) -- 거리 불변성이 필요하면 켜라.
        self.declare_parameter("voxel_size_m", 0.0)
        self.declare_parameter("voxel_warn_cells", 8000)
        self.declare_parameter("minimum_depth_m", 0.1)
        self.declare_parameter("maximum_depth_m", 10.0)
        self.declare_parameter("depth_scale_m", 0.001)
        self.declare_parameter("sync_tolerance_ms", 2.0)
        self.declare_parameter("use_identity_pose_when_missing", True)

        # 프레임마다 찍는 INFO를 끌 수 있게 한다. rclpy 로거는 호출할 때마다
        # inspect로 스택을 뒤지고 realpath로 경로를 푸는데(프레임당 lstat 366회
        # 실측), throttle에 걸려 출력되지 않아도 그 비용은 그대로 든다.
        self.declare_parameter("verbose", False)
        self.verbose = bool(self.get_parameter("verbose").value)

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.logged_resize = False
        self.depth_cache = {}
        self.segment_cache = {}
        self.pose_cache = {}
        self.max_cache = 30

        self.minimum_points = int(self.get_parameter("minimum_points").value)
        self.erosion_px = int(self.get_parameter("erosion_px").value)
        self.voxel_size_m = float(self.get_parameter("voxel_size_m").value)
        self.voxel_warn_cells = int(
            self.get_parameter("voxel_warn_cells").value
        )
        self.minimum_depth_m = float(self.get_parameter("minimum_depth_m").value)
        self.maximum_depth_m = float(self.get_parameter("maximum_depth_m").value)
        self.depth_scale_m = float(self.get_parameter("depth_scale_m").value)
        self.sync_tolerance_ns = int(
            float(self.get_parameter("sync_tolerance_ms").value) * 1_000_000
        )
        self.identity_ok = bool(
            self.get_parameter("use_identity_pose_when_missing").value
        )

        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_topic").value,
            self.on_depth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter("segment_topic").value,
            self.on_segment,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.on_pose,
            10,
        )

        self.pub = self.create_publisher(
            Instance3DSet,
            self.get_parameter("output_topic").value,
            10,
        )
        self.debug_cloud_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter("debug_cloud_topic").value,
            10,
        )
        self.centroid_marker_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter("centroid_marker_topic").value,
            10,
        )
        voxel = (f"{self.voxel_size_m * 100:g}cm" if self.voxel_size_m > 0.0
                 else "off (모든 점 발행)")
        self.get_logger().info(
            "Instance Builder node started: "
            f"depth={self.get_parameter('depth_topic').value}, "
            f"segments={self.get_parameter('segment_topic').value}, "
            f"voxel={voxel}, erosion={self.erosion_px}px, "
            f"sync_tolerance_ms={self.sync_tolerance_ns / 1_000_000:.3f}"
        )

    def on_camera_info(self, msg):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def trim(self, cache, cache_name):
        while len(cache) > self.max_cache:
            stale_key = next(iter(cache))
            del cache[stale_key]
            self.get_logger().warn(
                f"Dropped unmatched {cache_name} frame at "
                f"{stale_key[0]}.{stale_key[1]:09d}",
                throttle_duration_sec=1.0,
            )

    def on_depth(self, msg):
        k = key(msg.header.stamp)
        self.depth_cache[k] = msg
        self.trim(self.depth_cache, "depth")
        self.try_process_near(msg.header.stamp)

    def on_segment(self, msg):
        # /segment_image는 이제 mono8 sensor_msgs/Image다. header.stamp이 frame key다.
        k = key(msg.header.stamp)
        self.segment_cache[k] = msg
        self.trim(self.segment_cache, "segment")
        self.try_process_near(msg.header.stamp)

    def on_pose(self, msg):
        # /pose는 이제 geometry_msgs/PoseWithCovarianceStamped다.
        k = key(msg.header.stamp)
        self.pose_cache[k] = msg
        self.trim(self.pose_cache, "pose")
        self.try_process_near(msg.header.stamp)

    def nearest_key(self, cache, stamp):
        if not cache:
            return None

        target_ns = stamp_to_ns(stamp)
        candidate = min(
            cache,
            key=lambda candidate_key: abs(
                candidate_key[0] * 1_000_000_000 + candidate_key[1] - target_ns
            ),
        )
        candidate_ns = candidate[0] * 1_000_000_000 + candidate[1]
        if abs(candidate_ns - target_ns) > self.sync_tolerance_ns:
            return None
        return candidate

    def try_process_near(self, stamp):
        depth_key = self.nearest_key(self.depth_cache, stamp)
        segment_key = self.nearest_key(self.segment_cache, stamp)
        if depth_key is None or segment_key is None:
            return
        if any(v is None for v in (self.fx, self.fy, self.cx, self.cy)):
            self.get_logger().warn(
                "Waiting for CameraInfo...",
                throttle_duration_sec=5.0,
            )
            return

        pose = self.pose_cache.get(segment_key)
        if pose is None and not self.identity_ok:
            return

        depth_msg = self.depth_cache.pop(depth_key)
        seg = self.segment_cache.pop(segment_key)
        self.pose_cache.pop(segment_key, None)

        depth_stamp_ns = depth_key[0] * 1_000_000_000 + depth_key[1]
        segment_stamp_ns = segment_key[0] * 1_000_000_000 + segment_key[1]
        delta_ms = abs(depth_stamp_ns - segment_stamp_ns) / 1_000_000
        self.get_logger().info(
            f"Matched depth and segmentation: delta={delta_ms:.3f} ms",
            throttle_duration_sec=1.0,
        )

        try:
            self.build(depth_msg, seg, pose)
        except Exception as e:
            self.get_logger().error(f"Instance reconstruction failed: {e}")

    @staticmethod
    def voxel_downsample(points, voxel_size):
        """점유 칸당 1점 — 그 칸 실측점들의 산술평균.

        격자는 world 원점에 고정된 모눈이고 칸 주소는 floor(p / voxel_size)다.
        칸 기하 중심이 아니라 실측점 평균을 쓰므로 점이 모인 표면 쪽으로 쏠려
        앉는다(sub-voxel 충실도). 점유 칸만 남기므로 M은 가변이다.
        """
        if len(points) == 0:
            return points

        cells = np.floor(points / voxel_size).astype(np.int64)

        # 키를 음수 아닌 값으로 만들기 위한 정수 칸 단위 평행이동. 정수 이동이므로
        # 칸 경계는 world 좌표에서 voxel_size의 배수에 그대로 남고, 어떤 점이 어느
        # 칸에 들어가는지는 바뀌지 않는다.
        local = cells - cells.min(axis=0)
        dims = local.max(axis=0) + 1

        keys = (local[:, 0] * dims[1] + local[:, 1]) * dims[2] + local[:, 2]
        _, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )

        sums = np.empty((len(counts), 3), dtype=np.float64)
        for axis in range(3):
            sums[:, axis] = np.bincount(
                inverse, weights=points[:, axis], minlength=len(counts)
            )
        return sums / counts[:, None]

    @staticmethod
    def erode(mask, iterations):
        """마스크를 안쪽으로 iterations 픽셀 깎는다 (4-이웃 침식).

        이미지 경계는 깎지 않는다. 화면 끝은 물체의 경계가 아니라 시야의 끝이라
        depth가 섞이지 않기 때문이다. 경계에 걸친 물체를 괜히 잘라낼 이유가 없다.
        """
        if iterations <= 0:
            return mask
        for _ in range(iterations):
            eroded = mask.copy()
            eroded[1:, :] &= mask[:-1, :]    # 위 이웃
            eroded[:-1, :] &= mask[1:, :]    # 아래 이웃
            eroded[:, 1:] &= mask[:, :-1]    # 왼쪽 이웃
            eroded[:, :-1] &= mask[:, 1:]    # 오른쪽 이웃
            mask = eroded
        return mask

    def match_depth_to_labels(self, depth, label_shape):
        """depth를 label 격자로 축소하고 intrinsics를 같은 배율로 환산한다.

        segmentor가 proto 공간(예: 192x256) 라벨을 내면 카메라 depth(480x640)와
        크기가 다르다. 축소를 여기서 하는 이유는, 그 배율로 환산한 intrinsics를
        바로 아래 back-projection이 쓰기 때문이다 -- 둘이 떨어져 있으면 한쪽만
        고쳤을 때 조용히 어긋난다.

        반환: (depth, fx, fy, cx, cy) 또는 맞출 수 없으면 None.
        """
        source_h, source_w = depth.shape
        target_h, target_w = label_shape
        scale_y = target_h / source_h
        scale_x = target_w / source_w

        if abs(scale_y - scale_x) > 1e-6:
            self.get_logger().error(
                f"Depth/label 종횡비가 다릅니다: depth {depth.shape} vs "
                f"label {label_shape} (scale y={scale_y:.4f} x={scale_x:.4f}). "
                "라벨에 패딩이 남아 있으면 이렇게 됩니다.",
                throttle_duration_sec=5.0,
            )
            return None
        scale = scale_y

        # 목적지 픽셀 u'의 중심은 원본 좌표 (u'+0.5)/scale - 0.5에 대응한다.
        # 그 최근접 정수 픽셀을 직접 고른다. 라이브러리 resize에 맡기지 않는 이유는
        # 아래 intrinsics 환산이 바로 이 관례를 전제하기 때문이다 -- 관례를 추측하면
        # 반픽셀이 어긋나고, 그건 back-projection에 그대로 실린다.
        #
        # 보간이 아니라 픽셀 선택이라는 점이 중요하다. 평균을 내면 물체 경계에서
        # 전경과 배경 depth가 섞여 실제로는 없는 표면이 생긴다(1.5m 물체와 3m 벽
        # 사이에 2.25m 유령면).
        rows = np.rint((np.arange(target_h) + 0.5) / scale - 0.5).astype(np.int64)
        cols = np.rint((np.arange(target_w) + 0.5) / scale - 0.5).astype(np.int64)
        np.clip(rows, 0, source_h - 1, out=rows)
        np.clip(cols, 0, source_w - 1, out=cols)
        small = depth[rows[:, None], cols[None, :]]

        # 위와 같은 관례이므로 주점에 반픽셀 항이 붙는다. scale=0.4에서 0.3픽셀,
        # 1.5m에서 약 1.8mm -- 2cm 격자에는 미미하지만 공짜로 맞출 수 있다.
        fx = self.fx * scale
        fy = self.fy * scale
        cx = (self.cx + 0.5) * scale - 0.5
        cy = (self.cy + 0.5) * scale - 0.5

        if not self.logged_resize:
            self.logged_resize = True
            suggested = max(1, int(round(self.minimum_points * scale * scale)))
            self.get_logger().info(
                f"depth {source_h}x{source_w} -> label {target_h}x{target_w} "
                f"(scale {scale:.4f}, nearest), intrinsics "
                f"fx {self.fx:.1f}->{fx:.1f}, cy {self.cy:.1f}->{cy:.1f}"
            )
            if suggested != self.minimum_points:
                self.get_logger().warn(
                    f"minimum_points={self.minimum_points}는 이 격자에서 원본 기준 "
                    f"{int(self.minimum_points / (scale * scale))}픽셀을 요구합니다. "
                    f"원래 의도를 유지하려면 {suggested} 정도로 낮추세요: "
                    f"-p minimum_points:={suggested}"
                )
        return small, fx, fy, cx, cy

    def build(self, depth_msg, seg, pose_msg):
        if depth_msg.encoding == "16UC1":
            depth = np.asarray(
                self.bridge.imgmsg_to_cv2(depth_msg, "passthrough"),
                dtype=np.float32,
            ) * self.depth_scale_m
        elif depth_msg.encoding == "32FC1":
            depth = np.asarray(
                self.bridge.imgmsg_to_cv2(depth_msg, "32FC1"),
                dtype=np.float32,
            )
        else:
            self.get_logger().error(
                f"Unsupported depth encoding: {depth_msg.encoding}"
            )
            return
        labels = np.asarray(
            self.bridge.imgmsg_to_cv2(seg, "mono8"),
            dtype=np.uint8,
        )

        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        if depth.shape != labels.shape:
            matched = self.match_depth_to_labels(depth, labels.shape)
            if matched is None:
                return
            depth, fx, fy, cx, cy = matched

        rot = np.eye(3, dtype=np.float64)
        trans = np.zeros(3, dtype=np.float64)
        frame_id = depth_msg.header.frame_id

        if pose_msg is not None:
            # PoseWithCovarianceStamped: pose.pose가 world_T_camera다.
            p = pose_msg.pose.pose
            rot = quat_to_rot(
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            )
            trans = np.array(
                [p.position.x, p.position.y, p.position.z],
                dtype=np.float64,
            )
            frame_id = "world"

        # 자세는 float64로 만들고 점 변환에만 float32 사본을 쓴다. 루프 안에서
        # 매번 astype하지 않도록 여기서 한 번만 내린다.
        rot32 = rot.astype(np.float32)
        trans32 = trans.astype(np.float32)

        # header.stamp이 frame key이고, frame_id는 source camera frame이다.
        # 각 instance_points[k]의 frame_id는 world frame이어야 한다(아래에서 설정).
        out = Instance3DSet()
        out.header = deepcopy(seg.header)
        debug_blocks = []
        qualities = []  # (segment_id, geometry_quality, 점 수)

        # 루프 안에서는 로그를 찍지 않고 여기 모아둔다. 이유는 아래 dropped를
        # 채우는 곳의 주석 참고 -- 억제된 로그도 호출 비용이 든다.
        dropped = []      # (segment_id, 유효점, 마스크면적, quality)
        oversized = []    # (segment_id, 점 수)

        # RViz 전용 출력은 보는 사람이 있을 때만 만든다. 아무도 안 붙어 있으면
        # Marker 54개와 전체 점을 다시 훑는 디버그 클라우드가 통째로 사라진다.
        want_markers = self.centroid_marker_pub.get_subscription_count() > 0
        want_debug = self.debug_cloud_pub.get_subscription_count() > 0

        centroid_markers = MarkerArray()
        if want_markers:
            delete_all = Marker()
            delete_all.action = Marker.DELETEALL
            centroid_markers.markers.append(delete_all)

        # depth만 보고 정해지는 값이라 sid와 무관하다. 루프 안에 두면 인스턴스마다
        # 192x256 배열 비교를 처음부터 다시 한다(54개면 2.76ms, 밖에 두면 0.04ms).
        measurable = (
            np.isfinite(depth)
            & (depth >= self.minimum_depth_m)
            & (depth <= self.maximum_depth_m)
        )

        # ------------------------------------------------------------------
        # 세그먼트별로 전체 이미지를 훑던 것을 라벨맵 한 번씩으로 바꾼다.
        #
        # 예전에는 세그먼트마다 (labels == sid), count_nonzero, erode,
        # np.nonzero를 돌아서 192x256 배열을 8번씩 스캔했다. 세그먼트가 20개면
        # 160번이다. 아래는 전부 라벨맵 전체에 대해 한 번씩만 한다.
        # 결과(픽셀 순서 포함)는 예전과 동일하다 -- stable 정렬이 row-major
        # 순서를 보존하기 때문이다.
        # ------------------------------------------------------------------
        flat_labels = labels.ravel()
        flat_meas = measurable.ravel()

        # 면적과 quality: bincount 두 번이면 전 세그먼트가 한꺼번에 나온다
        areas = np.bincount(flat_labels, minlength=MAX_SEGMENT_ID + 1)
        # weights=를 쓰면 numpy가 float64 누산 경로를 탄다. 불리언 마스크로
        # 걸러서 세면 정수 경로라 같은 값이 더 싸게 나온다 (0.375 -> 0.195ms).
        meas_counts = np.bincount(
            flat_labels[flat_meas], minlength=MAX_SEGMENT_ID + 1
        )

        # 침식도 전 세그먼트 동시에. 어떤 픽셀이 살아남으려면 상하좌우 이웃이
        # "같은 라벨이고 직전 단계에서 살아 있어야" 한다 -- 세그먼트별 boolean
        # 마스크를 깎는 것과 정확히 같은 조건이다.
        alive = flat_labels.reshape(labels.shape) > 0
        for _ in range(max(self.erosion_px, 0)):
            keep = np.ones_like(alive)
            keep[1:, :] &= (labels[1:, :] == labels[:-1, :]) & alive[:-1, :]
            keep[:-1, :] &= (labels[:-1, :] == labels[1:, :]) & alive[1:, :]
            keep[:, 1:] &= (labels[:, 1:] == labels[:, :-1]) & alive[:, :-1]
            keep[:, :-1] &= (labels[:, :-1] == labels[:, 1:]) & alive[:, 1:]
            alive = alive & keep

        # 침식으로 통째로 사라진 세그먼트는 원본 마스크를 쓴다(작은 물체).
        eroded_areas = np.bincount(
            flat_labels[alive.ravel()], minlength=MAX_SEGMENT_ID + 1
        )
        fallback = eroded_areas == 0

        used = alive.ravel() | fallback[flat_labels]
        valid_flat = used & flat_meas & (flat_labels > 0)

        # 여기가 유일한 nonzero다. 세그먼트마다 하던 것을 한 번으로 줄였다.
        flat_idx = np.nonzero(valid_flat)[0]
        idx_labels = flat_labels[flat_idx]
        order = np.argsort(idx_labels, kind="stable")
        flat_idx = flat_idx[order]
        idx_labels = idx_labels[order]

        valid_counts = np.bincount(idx_labels, minlength=MAX_SEGMENT_ID + 1)
        starts = np.concatenate(([0], np.cumsum(valid_counts)))
        width = labels.shape[1]

        # ── 역투영을 세그먼트별이 아니라 전체 유효 픽셀에 한 번만 한다.
        #
        # 점 자체의 산술은 마이크로초 수준인데, 세그먼트마다 numpy를 ~15회
        # 부르면 호출당 고정 오버헤드(~3us)가 쌓여 그게 비용의 전부가 된다.
        # 54세그먼트 기준 2.73ms -> 0.62ms.
        #
        # flat_idx가 이미 라벨 순으로 정렬돼 있으므로, 아래 루프에서는
        # starts[sid]:starts[sid+1] 슬라이스(뷰, 복사 없음)만 꺼내 쓰면 된다.
        # 연산 순서가 세그먼트별로 하던 것과 같아 결과가 비트 단위로 동일하다.
        vv_all = flat_idx // width
        uu_all = flat_idx % width
        z_all = depth[vv_all, uu_all].astype(np.float32)
        x_all = (uu_all.astype(np.float32) - np.float32(cx)) * z_all \
            / np.float32(fx)
        y_all = (vv_all.astype(np.float32) - np.float32(cy)) * z_all \
            / np.float32(fy)
        pts_world_all = np.column_stack((x_all, y_all, z_all)) @ rot32.T \
            + trans32

        # 인스턴스마다 deepcopy(header)를 부르면 54개에 0.66ms다. 내용이 전부
        # 같으므로 하나만 만들어 공유한다 -- 직렬화는 값으로 읽으므로 안전하다.
        shared_header = Header()
        shared_header.stamp = depth_msg.header.stamp
        shared_header.frame_id = frame_id

        present = np.nonzero(areas)[0]
        for sid in present:
            if sid == 0:
                continue

            mask_area = int(areas[sid])

            # geometry_quality는 침식 전 원본 마스크 기준이다. "이 물체를 얼마나
            # 관측했나"는 물체의 실제 크기 대비여야 하고, 침식은 그 뒤에 오는
            # "어느 점을 믿을 것인가"의 문제이기 때문이다.
            geometry_quality = float(meas_counts[sid]) / max(mask_area, 1)

            lo, hi = starts[sid], starts[sid + 1]
            n = hi - lo
            if n < self.minimum_points:
                # ★ 루프 안에서 로그를 찍지 않는다. rclpy의 로거는 throttle
                #   판정 전에 inspect로 스택을 뒤지고 realpath로 경로를 풀어서,
                #   억제된 호출도 비용이 그대로 든다(프레임당 lstat 366회 실측).
                #   여기서는 세기만 하고 루프가 끝난 뒤 한 줄로 낸다.
                dropped.append((int(sid), n, mask_area, geometry_quality))
                continue

            # 위에서 전체를 한 번에 계산해 뒀다. 여기서는 잘라 쓰기만 한다(뷰).
            pts_segment = pts_world_all[lo:hi]

            # geometry_quality / truncated / mask_area / extent는 새 계약에서
            # 삭제됐다. 계산도 같이 지운다 -- 아무도 안 쓰는 값을 매 프레임 만드는
            # 셈이기 때문이다. center만 남기는데, 그것도 wire가 아니라 아래
            # RViz centroid marker 전용이다.
            center = pts_segment.mean(axis=0)

            # voxel_size_m <= 0이면 다운샘플 없이 back-projection 결과를 그대로 낸다.
            if self.voxel_size_m > 0.0:
                pts_world = self.voxel_downsample(
                    pts_segment, self.voxel_size_m
                )
            else:
                pts_world = pts_segment

            if len(pts_world) > self.voxel_warn_cells:
                oversized.append((int(sid), len(pts_world)))

            segment_rgb = segment_id_to_rgb(sid)
            packed_rgb = pack_rgb_float(*segment_rgb)

            if want_markers:
                marker = Marker()
                marker.header = shared_header
                marker.ns = "instance_centroids"
                marker.id = int(sid)
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(center[0])
                marker.pose.position.y = float(center[1])
                marker.pose.position.z = float(center[2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.r = segment_rgb[0] / 255.0
                marker.color.g = segment_rgb[1] / 255.0
                marker.color.b = segment_rgb[2] / 255.0
                marker.color.a = 1.0
                centroid_markers.markers.append(marker)

            # parallel array 4개: index k가 인스턴스 하나를 설명하며 길이가 같아야 한다.
            qualities.append((int(sid), geometry_quality, len(pts_world)))

            out.segment_ids.append(int(sid))
            out.center_world_m.append(
                Point(x=float(center[0]), y=float(center[1]),
                      z=float(center[2]))
            )
            out.geometry_quality.append(float(geometry_quality))
            # (N, 4) float32 버퍼를 한 번만 만들고 두 곳에서 함께 쓴다.
            # astype은 생략한다 -- numpy 대입이 알아서 float32로 낮춘다.
            block = np.empty((len(pts_world), 4), dtype=np.float32)
            block[:, :3] = pts_world
            block[:, 3] = packed_rgb

            out.instance_points.append(pack_cloud_bytes(shared_header, block))

            if want_debug:
                debug_blocks.append(block)

        self.pub.publish(out)

        if want_markers:
            self.centroid_marker_pub.publish(centroid_markers)

        if debug_blocks:
            debug_header = deepcopy(depth_msg.header)
            debug_header.frame_id = frame_id
            self.debug_cloud_pub.publish(
                pack_cloud_bytes(
                    debug_header, np.concatenate(debug_blocks, axis=0)
                )
            )

        # 루프에서 모아둔 것을 여기서 한 번씩만 낸다. 세그먼트가 N개여도
        # 로그 호출은 최대 3회다.
        if dropped:
            head = "  ".join(
                f"#{sid} {n}/{area}px q={q:.2f}"
                for sid, n, area, q in dropped[:5]
            )
            more = f" 외 {len(dropped) - 5}개" if len(dropped) > 5 else ""
            self.get_logger().warn(
                f"{len(dropped)}개 세그먼트 버림 (유효 depth < "
                f"{self.minimum_points}): {head}{more}",
                throttle_duration_sec=2.0,
            )

        if oversized:
            head = "  ".join(
                f"#{sid} {pts}점({pts * 12 / 1024:.0f}KB)"
                for sid, pts in oversized[:5]
            )
            self.get_logger().warn(
                f"{len(oversized)}개 세그먼트가 "
                f"{self.voxel_warn_cells}점을 넘습니다: {head}"
                + ("" if self.voxel_size_m > 0.0 else "  — voxel이 꺼져 있음"),
                throttle_duration_sec=2.0,
            )

        # 정상 프레임마다 찍히는 유일한 로그다. 로거 호출 자체가 비싸므로
        # (스택 조회 + 경로 realpath) 필요할 때만 켜도록 파라미터로 막는다.
        if self.verbose:
            detail = "  ".join(
                f"#{sid} q={q:.2f} {pts}pts" for sid, q, pts in qualities[:8]
            )
            self.get_logger().info(
                f"Published {len(out.segment_ids)} instances   {detail}",
                throttle_duration_sec=1.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = GeobuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
