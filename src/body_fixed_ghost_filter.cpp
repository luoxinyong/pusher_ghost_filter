#include "pusher_ghost_filter/body_fixed_ghost_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <pcl/common/point_tests.h>

namespace pusher_ghost_filter {

namespace {
constexpr double kPi = 3.14159265358979323846;

struct VoxelStats {
  std::size_t frame_count = 0U;
  std::uint64_t yaw_mask = 0U;
};

struct MapVoxelStats {
  std::size_t frame_count = 0U;
  std::size_t last_frame = std::numeric_limits<std::size_t>::max();
  std::uint64_t pose_mask = 0U;
  std::uint64_t body_voxel_mask = 0U;
};

std::uint64_t poseCellCode(const Eigen::Isometry3d &pose, double cell_size) {
  const auto x =
      static_cast<std::int32_t>(std::floor(pose.translation().x() / cell_size));
  const auto y =
      static_cast<std::int32_t>(std::floor(pose.translation().y() / cell_size));
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32U) |
         static_cast<std::uint32_t>(y);
}

std::uint64_t yawBit(const Eigen::Isometry3d &pose, int yaw_bins) {
  double yaw = std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0));
  if (yaw < 0.0) {
    yaw += 2.0 * kPi;
  }
  int bin = static_cast<int>(std::floor(yaw * yaw_bins / (2.0 * kPi)));
  bin = std::clamp(bin, 0, yaw_bins - 1);
  return std::uint64_t{1} << static_cast<unsigned int>(bin);
}

std::uint64_t hashedBit(std::size_t value) {
  value ^= value >> 33U;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33U;
  return std::uint64_t{1} << static_cast<unsigned int>(value & 63U);
}

int bitCount(std::uint64_t value) { return __builtin_popcountll(value); }

} // namespace

std::size_t BodyVoxelKeyHash::operator()(const BodyVoxelKey &key) const {
  std::size_t seed = std::hash<std::int32_t>{}(key.x);
  seed ^= std::hash<std::int32_t>{}(key.y) + 0x9e3779b9U + (seed << 6U) +
          (seed >> 2U);
  seed ^= std::hash<std::int32_t>{}(key.z) + 0x9e3779b9U + (seed << 6U) +
          (seed >> 2U);
  return seed;
}

BodyFixedGhostFilter::BodyFixedGhostFilter(BodyFixedGhostFilterConfig config)
    : config_(std::move(config)) {
  if (config_.voxel_size <= 0.0 || config_.min_radius < 0.0 ||
      config_.max_radius <= config_.min_radius ||
      config_.window_size_frames < 1 || config_.window_stride_frames < 1 ||
      config_.min_keyframe_ratio <= 0.0 || config_.min_keyframe_ratio > 1.0 ||
      config_.min_support_frames < 1 || config_.pose_cell_size <= 0.0 ||
      config_.min_pose_cells < 1 || config_.yaw_bins < 1 ||
      config_.yaw_bins > 64 || config_.min_yaw_bins < 1 ||
      config_.min_yaw_bins > config_.yaw_bins ||
      config_.min_component_voxels < 1 ||
      config_.max_component_voxels < config_.min_component_voxels ||
      config_.min_component_height < 0.0 ||
      config_.max_component_height < config_.min_component_height ||
      config_.max_component_width <= 0.0 ||
      config_.static_map_voxel_size <= 0.0 ||
      config_.static_min_support_frames < 1 ||
      config_.static_pose_cell_size <= 0.0 ||
      config_.static_min_pose_cells < 1 || config_.static_min_body_voxels < 1) {
    throw std::invalid_argument(
        "invalid body-fixed ghost filter configuration");
  }
}

bool BodyFixedGhostFilter::isNearField(const Point &point) const {
  if (!pcl::isFinite(point)) {
    return false;
  }
  const double radius = std::hypot(point.x, point.y);
  return radius >= config_.min_radius && radius <= config_.max_radius;
}

BodyVoxelKey BodyFixedGhostFilter::voxelKey(const Point &point) const {
  const double inverse = 1.0 / config_.voxel_size;
  return BodyVoxelKey{static_cast<std::int32_t>(std::floor(point.x * inverse)),
                      static_cast<std::int32_t>(std::floor(point.y * inverse)),
                      static_cast<std::int32_t>(std::floor(point.z * inverse))};
}

BodyVoxelKey
BodyFixedGhostFilter::mapVoxelKey(const Point &point,
                                  const Eigen::Isometry3d &pose) const {
  const Eigen::Vector3d mapped =
      pose * Eigen::Vector3d(point.x, point.y, point.z);
  const double inverse = 1.0 / config_.static_map_voxel_size;
  return BodyVoxelKey{
      static_cast<std::int32_t>(std::floor(mapped.x() * inverse)),
      static_cast<std::int32_t>(std::floor(mapped.y() * inverse)),
      static_cast<std::int32_t>(std::floor(mapped.z() * inverse))};
}

BodyFixedGhostFilter::Result
BodyFixedGhostFilter::detect(const std::vector<Frame> &frames) const {
  Result result;
  result.frame_masks.resize(frames.size());
  if (frames.empty()) {
    return result;
  }

  Result::VoxelSet all_near_field;
  Result::VoxelSet all_persistent;
  Result::VoxelSet all_pose_supported;

  // Estimate the body-frame ground layer once from the whole recording. A
  // short window can be dominated by a person or a nearby wall at one height.
  // Counting occupied near-field voxels over all frames makes the broad ground
  // plane dominate without requiring vehicle rotation.
  std::unordered_map<std::int32_t, std::size_t> global_z_histogram;
  for (const auto &frame : frames) {
    if (!frame.cloud) {
      continue;
    }
    Result::VoxelSet frame_voxels;
    for (const auto &point : frame.cloud->points) {
      if (isNearField(point)) {
        frame_voxels.insert(voxelKey(point));
      }
    }
    for (const auto &key : frame_voxels) {
      ++global_z_histogram[key.z];
      all_near_field.insert(key);
    }
  }
  if (global_z_histogram.empty()) {
    return result;
  }
  const auto global_ground_layer =
      std::max_element(global_z_histogram.begin(), global_z_histogram.end(),
                       [](const auto &left, const auto &right) {
                         return left.second < right.second;
                       });
  const std::int32_t ground_layer_z = global_ground_layer->first;
  result.estimated_ground_z =
      (static_cast<double>(ground_layer_z) + 0.5) * config_.voxel_size;

  const std::size_t stride =
      static_cast<std::size_t>(config_.window_stride_frames);
  const std::size_t requested_window =
      static_cast<std::size_t>(config_.window_size_frames);

  for (std::size_t apply_begin = 0U; apply_begin < frames.size();
       apply_begin += stride) {
    const std::size_t apply_end = std::min(frames.size(), apply_begin + stride);
    const std::size_t window_size = std::min(frames.size(), requested_window);
    const std::size_t apply_center =
        apply_begin + (apply_end - apply_begin) / 2U;
    std::size_t learn_begin =
        apply_center > window_size / 2U ? apply_center - window_size / 2U : 0U;
    if (learn_begin + window_size > frames.size()) {
      learn_begin = frames.size() - window_size;
    }
    const std::size_t learn_end = learn_begin + window_size;
    ++result.windows_examined;

    std::unordered_map<BodyVoxelKey, VoxelStats, BodyVoxelKeyHash> stats;
    for (std::size_t frame_index = learn_begin; frame_index < learn_end;
         ++frame_index) {
      const auto &frame = frames[frame_index];
      if (!frame.cloud) {
        continue;
      }
      Result::VoxelSet frame_voxels;
      frame_voxels.reserve(frame.cloud->size());
      for (const auto &point : frame.cloud->points) {
        if (isNearField(point)) {
          frame_voxels.insert(voxelKey(point));
        }
      }
      const std::uint64_t yaw_bit = yawBit(frame.pose, config_.yaw_bins);
      for (const auto &key : frame_voxels) {
        auto &voxel_stats = stats[key];
        ++voxel_stats.frame_count;
        voxel_stats.yaw_mask |= yaw_bit;
      }
    }

    const std::size_t required_support = std::max<std::size_t>(
        static_cast<std::size_t>(config_.min_support_frames),
        static_cast<std::size_t>(std::ceil(config_.min_keyframe_ratio *
                                           static_cast<double>(window_size))));
    result.required_support_frames =
        std::max(result.required_support_frames, required_support);

    Result::VoxelSet persistent;
    for (const auto &[key, voxel_stats] : stats) {
      if (voxel_stats.frame_count >= required_support &&
          bitCount(voxel_stats.yaw_mask) >= config_.min_yaw_bins) {
        persistent.insert(key);
        all_persistent.insert(key);
      }
    }

    std::unordered_map<BodyVoxelKey, std::unordered_set<std::uint64_t>,
                       BodyVoxelKeyHash>
        pose_cells;
    for (std::size_t frame_index = learn_begin; frame_index < learn_end;
         ++frame_index) {
      const auto &frame = frames[frame_index];
      if (!frame.cloud) {
        continue;
      }
      Result::VoxelSet frame_voxels;
      for (const auto &point : frame.cloud->points) {
        if (!isNearField(point)) {
          continue;
        }
        const BodyVoxelKey key = voxelKey(point);
        if (persistent.count(key) != 0U) {
          frame_voxels.insert(key);
        }
      }
      const std::uint64_t pose_code =
          poseCellCode(frame.pose, config_.pose_cell_size);
      for (const auto &key : frame_voxels) {
        pose_cells[key].insert(pose_code);
      }
    }

    Result::VoxelSet supported;
    for (const auto &key : persistent) {
      const auto found = pose_cells.find(key);
      if (found != pose_cells.end() &&
          static_cast<int>(found->second.size()) >= config_.min_pose_cells) {
        supported.insert(key);
        all_pose_supported.insert(key);
      }
    }
    if (supported.empty()) {
      continue;
    }

    const int ground_half_bins = static_cast<int>(
        std::ceil(config_.ground_exclusion_half_height / config_.voxel_size));

    Result::VoxelSet unseen;
    for (const auto &key : supported) {
      if (std::abs(key.z - ground_layer_z) > ground_half_bins) {
        unseen.insert(key);
      }
    }

    Result::VoxelSet accepted_mask;
    while (!unseen.empty()) {
      std::vector<BodyVoxelKey> stack{*unseen.begin()};
      unseen.erase(stack.back());
      std::vector<BodyVoxelKey> component;
      BodyVoxelKey minimum = stack.back();
      BodyVoxelKey maximum = stack.back();

      while (!stack.empty()) {
        const BodyVoxelKey key = stack.back();
        stack.pop_back();
        component.push_back(key);
        minimum.x = std::min(minimum.x, key.x);
        minimum.y = std::min(minimum.y, key.y);
        minimum.z = std::min(minimum.z, key.z);
        maximum.x = std::max(maximum.x, key.x);
        maximum.y = std::max(maximum.y, key.y);
        maximum.z = std::max(maximum.z, key.z);

        for (int dx = -1; dx <= 1; ++dx) {
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
              if (dx == 0 && dy == 0 && dz == 0) {
                continue;
              }
              const BodyVoxelKey neighbor{key.x + dx, key.y + dy, key.z + dz};
              const auto found = unseen.find(neighbor);
              if (found != unseen.end()) {
                stack.push_back(*found);
                unseen.erase(found);
              }
            }
          }
        }
      }

      ++result.components_examined;
      const double width_x = (maximum.x - minimum.x + 1) * config_.voxel_size;
      const double width_y = (maximum.y - minimum.y + 1) * config_.voxel_size;
      const double height = (maximum.z - minimum.z + 1) * config_.voxel_size;
      if (static_cast<int>(component.size()) < config_.min_component_voxels ||
          static_cast<int>(component.size()) > config_.max_component_voxels ||
          height < config_.min_component_height ||
          height > config_.max_component_height ||
          width_x > config_.max_component_width ||
          width_y > config_.max_component_width) {
        continue;
      }

      ++result.components_accepted;
      accepted_mask.insert(component.begin(), component.end());
      result.body_candidate_voxels.insert(component.begin(), component.end());
    }

    for (std::size_t frame_index = apply_begin; frame_index < apply_end;
         ++frame_index) {
      result.frame_masks[frame_index].insert(accepted_mask.begin(),
                                             accepted_mask.end());
    }
  }

  result.near_field_voxels = all_near_field.size();
  result.persistent_voxels = all_persistent.size();
  result.pose_supported_voxels = all_pose_supported.size();
  std::unordered_map<BodyVoxelKey, MapVoxelStats, BodyVoxelKeyHash> map_stats;
  for (std::size_t frame_index = 0U; frame_index < frames.size();
       ++frame_index) {
    const auto &frame = frames[frame_index];
    if (!frame.cloud) {
      continue;
    }
    const std::uint64_t pose_bit = hashedBit(std::hash<std::uint64_t>{}(
        poseCellCode(frame.pose, config_.static_pose_cell_size)));
    for (const auto &point : frame.cloud->points) {
      if (!pcl::isFinite(point)) {
        continue;
      }
      auto &map_stat = map_stats[mapVoxelKey(point, frame.pose)];
      if (map_stat.last_frame != frame_index) {
        ++map_stat.frame_count;
        map_stat.last_frame = frame_index;
      }
      map_stat.pose_mask |= pose_bit;
      map_stat.body_voxel_mask |=
          hashedBit(BodyVoxelKeyHash{}(voxelKey(point)));
    }
  }

  for (const auto &[key, map_stat] : map_stats) {
    if (static_cast<int>(map_stat.frame_count) >=
            config_.static_min_support_frames &&
        bitCount(map_stat.pose_mask) >= config_.static_min_pose_cells &&
        bitCount(map_stat.body_voxel_mask) >= config_.static_min_body_voxels) {
      result.static_map_voxels.insert(key);
    }
  }

  for (std::size_t frame_index = 0U; frame_index < frames.size();
       ++frame_index) {
    const auto &frame = frames[frame_index];
    if (!frame.cloud) {
      continue;
    }
    for (const auto &point : frame.cloud->points) {
      if (!isNearField(point) ||
          result.frame_masks[frame_index].count(voxelKey(point)) == 0U) {
        continue;
      }
      ++result.candidate_observations;
      if (result.static_map_voxels.count(mapVoxelKey(point, frame.pose)) !=
          0U) {
        ++result.static_protected_observations;
      } else {
        ++result.removed_observations;
      }
    }
  }
  return result;
}

bool BodyFixedGhostFilter::isBodyCandidate(const Point &point,
                                           std::size_t frame_index,
                                           const Result &result) const {
  if (!isNearField(point) || frame_index >= result.frame_masks.size()) {
    return false;
  }
  return result.frame_masks[frame_index].count(voxelKey(point)) != 0U;
}

bool BodyFixedGhostFilter::isStaticProtected(const Point &point,
                                             const Eigen::Isometry3d &pose,
                                             const Result &result) const {
  return pcl::isFinite(point) &&
         result.static_map_voxels.count(mapVoxelKey(point, pose)) != 0U;
}

bool BodyFixedGhostFilter::isMasked(const Point &point, std::size_t frame_index,
                                    const Eigen::Isometry3d &pose,
                                    const Result &result) const {
  return isBodyCandidate(point, frame_index, result) &&
         !isStaticProtected(point, pose, result);
}

} // namespace pusher_ghost_filter
