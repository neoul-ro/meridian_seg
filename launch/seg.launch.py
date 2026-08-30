"""seg_node(+ with_geobuilder:=true 면 geobuilder_node)를 띄운다. 카메라는 별도로 실행한다.

토픽 이름
--------
노드의 기본값은 meridian_msgs 문서의 계약 이름이다:
    /camera/rgb  /camera/depth  /camera/info  /segment_image  /pose  /instance_3d_set

그런데 RealSense 드라이버는 /camera/camera/color/image_raw 같은 자기 이름으로
발행한다. 그래서 이 launch가 실제 드라이버 토픽을 파라미터로 넘겨준다. 드라이버를
계약 이름으로 remap해서 쓰신다면 아래 인자를 비워도 된다.

    ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true \
        enable_sync:=true align_depth.enable:=true enable_rgbd:=false

출력 마스크는 proto 공간에서 letterbox 패딩을 제거한 (192, 256) mono8이다.
geobuilder는 depth를 그 격자로 스스로 축소하고 intrinsics를 같은 배율로
환산한다 -- 그래서 depth 토픽은 카메라 원본을 그대로 주면 된다.

파라미터를 하나하나 설명한 표는 README.md의 "조절할 수 있는 값"에 있다.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# depth가 480x640에서 192x256으로 6.25배(2.5^2) 줄어든다. minimum_points는
# 640x480 기준으로 잡힌 값이라 그대로 두면 원본 기준 625픽셀을 요구하는 셈이
# 되어 작은 물체가 전부 탈락한다. 노드가 시작할 때 권고값을 로그로 알려준다.
DOWNSAMPLED_MINIMUM_POINTS = 16

# RealSense 드라이버가 실제로 쓰는 이름
RS_COLOR = "/camera/camera/color/image_raw"
RS_DEPTH = "/camera/camera/aligned_depth_to_color/image_raw"
RS_INFO = "/camera/camera/aligned_depth_to_color/camera_info"


def generate_launch_description():
    model_path = LaunchConfiguration("model_path")
    conf_th = LaunchConfiguration("conf_th")
    iou_th = LaunchConfiguration("iou_th")
    area_min = LaunchConfiguration("area_min")
    color_topic = LaunchConfiguration("color_topic")
    segment_topic = LaunchConfiguration("segment_topic")
    with_geobuilder = LaunchConfiguration("with_geobuilder")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    pose_topic = LaunchConfiguration("pose_topic")
    minimum_points = LaunchConfiguration("minimum_points")
    postprocess_mode = LaunchConfiguration("postprocess_mode")
    mask_dedup = LaunchConfiguration("mask_dedup")
    lanes = LaunchConfiguration("lanes")

    arguments = [
        DeclareLaunchArgument(
            "model_path",
            default_value="",
            description="TensorRT 엔진 경로. 비우면 MERIDIAN_SEG_ENGINE 환경변수 -> "
                        "share/meridian_seg/weights -> 소스 트리 weights/ 순으로 "
                        "FastSAM-s-1024.engine을 찾는다. 품질 기준이 필요하면 "
                        "-x로 엔진을 따로 빌드해서 그 경로를 준다(6배 느림, "
                        "README의 '큰 모델(-x)이 필요하면' 참고). 어느 쪽이든 "
                        "1024x1024 고정 크기 엔진이어야 한다.",
        ),
        DeclareLaunchArgument(
            "conf_th", default_value="0.5",
            description="이 점수 미만은 물체로 보지 않는다. 노드 기본값은 0.4. "
                        "올리면 확실한 것만 남고, 내리면 물체 수가 늘어난다.",
        ),
        DeclareLaunchArgument(
            "iou_th", default_value="0.7",
            description="상자 NMS 임계값. 노드 기본값은 0.9. mask_dedup이 켜져 "
                        "있으면 진짜 중복 판정은 마스크가 하므로 0.7로 관대하게 "
                        "두는 편이 낫다.",
        ),
        DeclareLaunchArgument(
            "area_min", default_value="64",
            description="proto 격자(192x256)에서 이 픽셀 수 미만인 마스크는 "
                        "버린다. 노드 기본값은 16. 올리면 자잘한 조각이 사라진다.",
        ),
        DeclareLaunchArgument(
            "color_topic", default_value=RS_COLOR,
            description="계약 이름은 /camera/rgb. 기본값은 RealSense 실제 토픽.",
        ),
        DeclareLaunchArgument(
            "depth_topic", default_value=RS_DEPTH,
            description="geobuilder 전용. 계약 이름은 /camera/depth. "
                        "기본값은 RealSense 실제 토픽.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic", default_value=RS_INFO,
            description="geobuilder 전용. 계약 이름은 /camera/info. "
                        "기본값은 RealSense 실제 토픽.",
        ),
        DeclareLaunchArgument(
            "segment_topic", default_value="/segment_image",
            description="seg_node가 내보내고 geobuilder가 받는 라벨 이미지 토픽.",
        ),
        DeclareLaunchArgument(
            "pose_topic", default_value="/pose",
            description="geobuilder 전용. world_T_camera 자세. 안 들어오면 "
                        "카메라 좌표계로 낸다.",
        ),
        DeclareLaunchArgument(
            "minimum_points", default_value=str(DOWNSAMPLED_MINIMUM_POINTS),
            description="geobuilder 전용. 192x256 격자 기준. 640x480이면 100이 "
                        "원래 값이다.",
        ),
        DeclareLaunchArgument(
            "with_geobuilder", default_value="false",
            description="3D geobuilder까지 같이 띄운다.",
        ),
        DeclareLaunchArgument(
            "postprocess_mode", default_value="graph_full",
            description="eager / fixed / graph / graph_full. 결과는 모두 같고 "
                        "속도만 다르다. 실측(실시간 RealSense, 물체 39개): "
                        "eager 9.5ms, graph_full 8.9ms. 문제가 생기면 "
                        "eager로 되돌리면 된다.",
        ),
        DeclareLaunchArgument(
            "mask_dedup", default_value="true",
            description="상자는 안 겹치는데 픽셀은 겹치는 중복 마스크를 "
                        "지운다. iou_th를 관대하게(0.7) 두는 것과 짝이다.",
        ),
        DeclareLaunchArgument(
            "lanes", default_value="56",
            description="NMS 생존 마스크를 담는 고정 슬롯 수. "
                        "'NMS 생존자가 lanes(56)를 채웠습니다' 경고가 뜨면 "
                        "72처럼 올린다. 최대 255.",
        ),
    ]

    seg = Node(
        package="meridian_seg",
        executable="seg_node",
        name="seg_node",
        output="screen",
        parameters=[
            {
                "model_path": model_path,
                "conf_th": conf_th,
                "iou_th": iou_th,
                "area_min": area_min,
                "color_topic": color_topic,
                "segment_topic": segment_topic,
                "postprocess_mode": postprocess_mode,
                "mask_dedup": mask_dedup,
                "lanes": lanes,
            }
        ],
    )

    geobuilder = Node(
        package="meridian_seg",
        executable="geobuilder_node",
        name="geobuilder_node",
        output="screen",
        condition=IfCondition(with_geobuilder),
        # 스레드 수 제한은 launch의 additional_env로는 적용되지 않아
        # (Humble에서 확인) 노드 코드 맨 위에서 직접 건다.
        parameters=[
            {
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
                "segment_topic": segment_topic,
                "pose_topic": pose_topic,
                "minimum_points": minimum_points,
            }
        ],
    )

    return LaunchDescription(arguments + [seg, geobuilder])
