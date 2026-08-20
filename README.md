# pusher_ghost_filter

[中文](#中文) | [English](#english)

## 中文

面向 ROS 2 Humble 的独立推车人影过滤功能包。它用于人工推着 LiDAR/建图设备行走时，
删除长期跟随传感器的人体拖影，同时通过多位置、多视角静态证据保护墙面、柱体、货架和
门边等真实结构。

算法不依赖 FAST-LIO2 源码，也不读取特定 rosbag 格式。任何前端只要提供时间同步的
“车体/雷达系局部点云 + 固定坐标系里程计”，都可以接入。

### 输入约定

- `cloud_topic`：`sensor_msgs/msg/PointCloud2`，必须位于移动的车体或雷达坐标系；
- `odom_topic`：同步的 `nav_msgs/msg/Odometry`，其 pose 将点云坐标系变换到固定坐标系。

已经变换到全局地图坐标系的点云不能直接输入，否则会被重复应用里程计位姿。只有最终
合并 PCD 也不够，因为算法需要逐帧时序和同步位姿。

### 构建和运行

```bash
mkdir -p ~/ghost_filter_ws/src
git clone https://github.com/luoxinyong/pusher_ghost_filter.git \
  ~/ghost_filter_ws/src/pusher_ghost_filter
cd ~/ghost_filter_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 launch pusher_ghost_filter filter.launch.py
```

回放完成后保存结果：

```bash
ros2 service call /pusher_ghost_filter/save std_srvs/srv/Trigger '{}'
```

会生成三层 PCD：清理后地图、判定删除的人影候选、被多视角静态证据保护的候选。正式
接受地图前，应同时检查 removed 和 static-protected 两层。再次回放前调用：

```bash
ros2 service call /pusher_ghost_filter/reset std_srvs/srv/Trigger '{}'
```

### 主要限制

- 默认参数面向距传感器 3 m 内的步行人体，换车或换安装位置后必须重新检查；
- 细柱、货架腿、门边和贴墙通道需要重点复核；
- 输出不会自动覆盖原始地图；
- 需要完整逐帧数据，不能仅靠最终合并 PCD 恢复动态判断。

所有话题、输出路径和过滤阈值均位于 `config/default.yaml`。

## English

ROS 2 package for removing the trail left by a person who walks with or pushes a
LiDAR mapping platform. It learns short-lived body-frame masks, protects map
voxels seen repeatedly from different viewpoints, and makes the final decision
per frame instead of applying one global exclusion box.

### Inputs

- `cloud_topic`: `sensor_msgs/msg/PointCloud2` expressed in the moving body or
  LiDAR frame.
- `odom_topic`: synchronized `nav_msgs/msg/Odometry`; its pose transforms the
  input cloud into a fixed map/odom frame.

The package works with ROS 2 bags through normal topic playback. It does not
read a particular rosbag storage format directly.

### Build and run

```bash
colcon build --packages-select pusher_ghost_filter
source install/setup.bash
ros2 launch pusher_ghost_filter filter.launch.py
ros2 bag play /path/to/ros2_bag
ros2 service call /pusher_ghost_filter/save std_srvs/srv/Trigger '{}'
```

The save service writes three PCD files: the cleaned map, points classified as
moving-pusher candidates, and candidate points retained by multi-view static
evidence. Use the last two as review layers before accepting a map.

Use `/pusher_ghost_filter/reset` before replaying another bag in the same node.

### Important limitations

- A final merged PCD is not enough: temporal body-frame clouds and synchronized
  poses are required.
- Input clouds already transformed into the fixed map frame are not valid unless
  they are first transformed back into the moving body frame.
- The default parameters target a walking-height object within 3 m of the
  sensor. Review narrow shelves, posts, door edges, and wall-adjacent passages.
- No automatic output should overwrite the original map.

See `config/default.yaml` for all parameters.

## License

MIT License. See `LICENSE`.
