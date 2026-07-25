import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge

from meridian_msgs.msg import RGBDFrame, SegmentImage


class MeridianSeg(Node):
    # Deterministic SAM placeholder: labels each frame with a fixed
    # grid_rows x grid_cols grid of segment ids, identical every frame.

    def __init__(self):
        super().__init__('meridian_seg')

        self.declare_parameter('grid_rows', 2)
        self.declare_parameter('grid_cols', 2)
        self.grid_rows = self.get_parameter('grid_rows').value
        self.grid_cols = self.get_parameter('grid_cols').value

        # segment ids must fit uint8 (1..255); clamp if params exceed that
        if self.grid_rows * self.grid_cols > 255:
            self.get_logger().warn(
                'grid_rows * grid_cols (%d) exceeds 255; segment ids will be clamped to fit uint8'
                % (self.grid_rows * self.grid_cols))

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self.sub = self.create_subscription(RGBDFrame, '/rgbd_frame', self.frame_callback, qos)
        self.pub = self.create_publisher(SegmentImage, '/segment_image', qos)

        self.get_logger().info(
            'meridian_seg started: grid_rows=%d grid_cols=%d' % (self.grid_rows, self.grid_cols))

    def frame_callback(self, msg):
        height = msg.rgb.height
        width = msg.rgb.width

        # cell (r, c) -> segment_id = r * grid_cols + c + 1
        row_idx = (np.arange(height) * self.grid_rows // height).astype(np.int32)
        col_idx = (np.arange(width) * self.grid_cols // width).astype(np.int32)
        labels = row_idx[:, None] * self.grid_cols + col_idx[None, :] + 1
        labels = np.clip(labels, 0, 255).astype(np.uint8)

        labels_msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
        labels_msg.header.stamp = msg.timestamp
        labels_msg.header.frame_id = 'camera'

        seg_msg = SegmentImage()
        seg_msg.timestamp = msg.timestamp
        seg_msg.labels = labels_msg
        self.pub.publish(seg_msg)

        self.get_logger().info(
            'published segment_image %dx%d grid=%dx%d' % (height, width, self.grid_rows, self.grid_cols),
            throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = MeridianSeg()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
