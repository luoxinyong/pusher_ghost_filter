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

struct BodyFixedGhostFilterConfig {
  double voxel_size = 0.20;
  double min_radius = 0.25;
  double max_radius = 3.0;
  int window_size_frames = 60;
  int window_stride_frames = 30;
  double min_keyframe_ratio = 0.12;
  int min_support_frames = 8;
  double pose_cell_size = 2.0;
  int min_pose_cells = 4;
  int yaw_bins = 12;
  int min_yaw_bins = 1;
  double ground_exclusion_half_height = 0.25;
  int min_component_voxels = 4;
  int max_component_voxels = 3000;
  double min_component_height = 0.50;
  double max_component_height = 2.80;
  double max_component_width = 1.80;
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
    Cloud::ConstPtr cloud;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  };

  struct Result {
    using VoxelSet = std::unordered_set<BodyVoxelKey, BodyVoxelKeyHash>;

    std::vector<VoxelSet> frame_masks;
    VoxelSet body_candidate_voxels;
    VoxelSet static_map_voxels;
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

  Result detect(const std::vector<Frame> &frames) const;
  bool isBodyCandidate(const Point &point, std::size_t frame_index,
                       const Result &result) const;
  bool isStaticProtected(const Point &point, const Eigen::Isometry3d &pose,
                         const Result &result) const;
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
