#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pusher_ghost_filter {

// 三维体素的整数索引。这里的同一种 Key 会用于两套坐标系：
// 1. 车体/雷达坐标系体素：寻找长期跟着传感器移动的人体；
// 2. 地图坐标系体素：寻找从多个位置反复看到的静态结构。
// 具体属于哪套坐标系，由生成它的 voxelKey()/mapVoxelKey() 决定。
struct BodyVoxelKey {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;

  bool operator==(const BodyVoxelKey &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct BodyVoxelKeyHash {
  std::size_t operator()(const BodyVoxelKey &key) const;
};

// 算法全部参数。默认值针对“人在近距离推着建图设备行走”的数据，
// 换雷达安装位置、车辆尺寸或建图方式后，应通过 YAML 调整并重新检查结果。
struct BodyFixedGhostFilterConfig {
  // 车体系人体学习使用的体素边长和水平搜索半径。
  double voxel_size = 0.20;
  double min_radius = 0.25;
  double max_radius = 3.0;

  // 每次用 window_size_frames 帧学习人体遮罩，但只向前应用
  // window_stride_frames 帧。窗口重叠可以适应人中途换位置。
  int window_size_frames = 60;
  int window_stride_frames = 30;
  // 一个车体系体素至少要出现多少帧，才算“长期跟车”。实际门槛为：
  // max(min_support_frames, ceil(window_size * min_keyframe_ratio))。
  double min_keyframe_ratio = 0.12;
  int min_support_frames = 8;

  // 将车辆在地图中的 XY 位置离散成格子。候选必须跨多个车辆位置格出现，
  // 避免车辆原地停留时把普通近场物体误认为跟车人体。
  double pose_cell_size = 2.0;
  int min_pose_cells = 4;

  // 航向角被离散成 yaw_bins 个区间；候选至少覆盖 min_yaw_bins 个区间。
  int yaw_bins = 12;
  int min_yaw_bins = 1;

  // 地面排除和人体三维连通域尺寸限制。
  double ground_exclusion_half_height = 0.25;
  int min_component_voxels = 4;
  int max_component_voxels = 3000;
  double min_component_height = 0.50;
  double max_component_height = 2.80;
  double max_component_width = 1.80;

  // 地图系静态保护参数：同一地图体素若被多帧、多个车辆位置、多个
  // 车体相对位置重复观测，就更像墙/柱/货架，应从删除候选中保护下来。
  double static_map_voxel_size = 0.30;
  int static_min_support_frames = 3;
  double static_pose_cell_size = 1.0;
  int static_min_pose_cells = 2;
  int static_min_body_voxels = 3;
};

class BodyFixedGhostFilter {
public:
  using Point = pcl::PointXYZI;
  using Cloud = pcl::PointCloud<Point>;

  struct Frame {
    // cloud 必须仍在移动的车体或雷达坐标系中，不能提前变换到地图系。
    Cloud::ConstPtr cloud;
    // pose 将 cloud 从移动坐标系变换到固定 map/odom 坐标系：
    // p_map = pose * p_body。
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  };

  struct Result {
    using VoxelSet = std::unordered_set<BodyVoxelKey, BodyVoxelKeyHash>;

    // 每帧单独的人体候选遮罩。不能只用一个全局遮罩，因为推车人可能换边。
    std::vector<VoxelSet> frame_masks;
    // 所有窗口中曾被接受的人体尺度车体系体素，主要用于统计和调试。
    VoxelSet body_candidate_voxels;
    // 获得多视角证据的地图系静态体素；命中它的候选点不能删除。
    VoxelSet static_map_voxels;

    // 以下字段均为诊断统计，不直接参与最终删除判断。
    std::size_t near_field_voxels = 0U;
    std::size_t persistent_voxels = 0U;
    std::size_t pose_supported_voxels = 0U;
    std::size_t windows_examined = 0U;
    std::size_t components_examined = 0U;
    std::size_t components_accepted = 0U;
    std::size_t required_support_frames = 0U;
    std::size_t candidate_observations = 0U;
    std::size_t static_protected_observations = 0U;
    std::size_t removed_observations = 0U;
    double estimated_ground_z = std::numeric_limits<double>::quiet_NaN();
  };

  explicit BodyFixedGhostFilter(BodyFixedGhostFilterConfig config);

  // 第一遍分析全部关键帧，生成逐帧人体候选和地图系静态保护集合。
  Result detect(const std::vector<Frame> &frames) const;
  // 第二遍导出地图时调用：判断点是否命中当前帧的人体候选遮罩。
  bool isBodyCandidate(const Point &point, std::size_t frame_index,
                       const Result &result) const;
  // 判断点变换到地图系后，是否命中多视角静态保护体素。
  bool isStaticProtected(const Point &point, const Eigen::Isometry3d &pose,
                         const Result &result) const;
  // 最终删除条件：是人体候选，并且没有静态保护证据。
  bool isMasked(const Point &point, std::size_t frame_index,
                const Eigen::Isometry3d &pose, const Result &result) const;

private:
  bool isNearField(const Point &point) const;
  BodyVoxelKey voxelKey(const Point &point) const;
  BodyVoxelKey mapVoxelKey(const Point &point,
                           const Eigen::Isometry3d &pose) const;

  BodyFixedGhostFilterConfig config_;
};

} // namespace pusher_ghost_filter
