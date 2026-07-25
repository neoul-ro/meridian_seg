# meridian_seg

Deterministic placeholder for SAM segmentation: labels each RGB-D frame with a fixed grid of segment ids.

## I/O

| Topic | Type | Direction |
| --- | --- | --- |
| `/camera/rgb` | `sensor_msgs/Image` (rgb8) | subscribe |
| `/segment_image` | `sensor_msgs/Image` (mono8) | publish |

## Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `grid_rows` | int | 2 | Number of grid rows for placeholder segmentation |
| `grid_cols` | int | 2 | Number of grid columns for placeholder segmentation |

## Run

```
ros2 run meridian_seg seg_node
```
