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

// 一个车体系体素在当前学习窗口里的时序证据。
struct VoxelStats {
  // 该体素被多少个不同关键帧占用；同一帧有多少个点都只计一次。
  std::size_t frame_count = 0U;
  // 车辆在哪些航向区间看到过它；第 n 位对应第 n 个 yaw bin。
  std::uint64_t yaw_mask = 0U;
};

// 一个地图系体素的静态证据。
struct MapVoxelStats {
  std::size_t frame_count = 0U;
  // 防止同一帧中的多个点把 frame_count 重复累加。
  std::size_t last_frame = std::numeric_limits<std::size_t>::max();
  // 车辆从哪些地图位置格看到过这个地图体素。
  std::uint64_t pose_mask = 0U;
  // 这个地图体素曾对应过哪些车体系相对位置。
  std::uint64_t body_voxel_mask = 0U;
};

// 将车辆地图 XY 位置离散成格子，再把两个有符号 32 位格子编号装进 uint64。
// 它用于回答“车辆是否真的从多个不同位置观察过候选”。
std::uint64_t poseCellCode(const Eigen::Isometry3d &pose, double cell_size) {
  const auto x =
      static_cast<std::int32_t>(std::floor(pose.translation().x() / cell_size));
  const auto y =
      static_cast<std::int32_t>(std::floor(pose.translation().y() / cell_size));
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32U) |
         static_cast<std::uint32_t>(y);
}

// 将车辆航向离散为不超过 64 个区间，并返回只有对应位为 1 的位掩码。
std::uint64_t yawBit(const Eigen::Isometry3d &pose, int yaw_bins) {
  double yaw = std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0));
  if (yaw < 0.0) {
    yaw += 2.0 * kPi;
  }
  int bin = static_cast<int>(std::floor(yaw * yaw_bins / (2.0 * kPi)));
  bin = std::clamp(bin, 0, yaw_bins - 1);
  return std::uint64_t{1} << static_cast<unsigned int>(bin);
}

// 把任意编号压成 64 位掩码中的一位。地图静态统计使用它节省内存；
// 不同编号理论上可能落到同一位，因此这里记录的是紧凑近似支持数。
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
  // 只按 XY 水平距离限制近场，不限制 Z；人体从脚到头都应进入候选范围。
  const double radius = std::hypot(point.x, point.y);
  return radius >= config_.min_radius && radius <= config_.max_radius;
}

BodyVoxelKey BodyFixedGhostFilter::voxelKey(const Point &point) const {
  // 直接对原始点坐标取整，所以这是“跟着车辆移动”的车体系体素。
  const double inverse = 1.0 / config_.voxel_size;
  return BodyVoxelKey{static_cast<std::int32_t>(std::floor(point.x * inverse)),
                      static_cast<std::int32_t>(std::floor(point.y * inverse)),
                      static_cast<std::int32_t>(std::floor(point.z * inverse))};
}

BodyVoxelKey
BodyFixedGhostFilter::mapVoxelKey(const Point &point,
                                  const Eigen::Isometry3d &pose) const {
  // 同一个点先由 pose 变到固定地图系，再使用独立的静态地图体素尺寸。
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
  // 每帧必须有独立遮罩：人在建图中途换边时，不能用一个全局硬框删除全程。
  result.frame_masks.resize(frames.size());
  if (frames.empty()) {
    return result;
  }

  Result::VoxelSet all_near_field;
  Result::VoxelSet all_persistent;
  Result::VoxelSet all_pose_supported;

  // 阶段 1：用整段记录估计车体系地面 Z 层。
  // 不能只在单个短窗口里估计，因为短窗口可能被人体或贴近的墙面支配。
  // 对每帧先去重体素，再按 Z 层累计“被多少帧占用”；宽广地面通常成为主峰。
  std::unordered_map<std::int32_t, std::size_t> global_z_histogram;
  for (const auto &frame : frames) {
    if (!frame.cloud) {
      continue;
    }
    // set 保证同一帧同一体素无论包含多少原始点，都只给直方图贡献一次。
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
  // 出现帧数最多的 Z 体素层作为地面层，保存的是体素中心高度。
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

  // 阶段 2：滑动窗口学习车体系人体遮罩。
  // “学习窗口”比“应用区间”更宽：用周围上下文学习，只把结果应用到中间
  // 对应的一小段帧，随后窗口向前移动。这样人换位置后能重新学习。
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

    // 2.1 统计当前学习窗口内，每个近场车体系体素出现的帧数和航向支持。
    std::unordered_map<BodyVoxelKey, VoxelStats, BodyVoxelKeyHash> stats;
    for (std::size_t frame_index = learn_begin; frame_index < learn_end;
         ++frame_index) {
      const auto &frame = frames[frame_index];
      if (!frame.cloud) {
        continue;
      }
      // 仍然先做帧内去重，避免高反射/高密度表面被重复计数。
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

    // 2.2 持久性门槛同时受绝对帧数和窗口占比约束。
    // 默认 60 帧窗口时：max(8, ceil(60*0.12)) = 8 帧。
    const std::size_t required_support = std::max<std::size_t>(
        static_cast<std::size_t>(config_.min_support_frames),
        static_cast<std::size_t>(std::ceil(config_.min_keyframe_ratio *
                                           static_cast<double>(window_size))));
    result.required_support_frames =
        std::max(result.required_support_frames, required_support);

    // persistent 表示“在车体附近相同位置长期存在”的体素。
    // 这时它可能是推车人，也可能是车辆短时间贴着走的墙，尚不能删除。
    Result::VoxelSet persistent;
    for (const auto &[key, voxel_stats] : stats) {
      if (voxel_stats.frame_count >= required_support &&
          bitCount(voxel_stats.yaw_mask) >= config_.min_yaw_bins) {
        persistent.insert(key);
        all_persistent.insert(key);
      }
    }

    // 2.3 统计持久体素跨过多少个车辆地图位置格。
    // 车辆原地停留时出现的物体不能仅凭帧数成为跟车候选。
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

    // supported = 同时满足时序持久性、航向和车辆位置支持的车体系体素。
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

    // 2.4 把以米表示的地面排除半高换算成整数体素层数。
    const int ground_half_bins = static_cast<int>(
        std::ceil(config_.ground_exclusion_half_height / config_.voxel_size));

    // 地面自身也会长期固定在车体系里，必须在连通域分析前排除。
    Result::VoxelSet unseen;
    for (const auto &key : supported) {
      if (std::abs(key.z - ground_layer_z) > ground_half_bins) {
        unseen.insert(key);
      }
    }

    // 2.5 对剩余体素做 26 邻域三维连通域搜索，并按人体尺寸筛选。
    Result::VoxelSet accepted_mask;
    while (!unseen.empty()) {
      std::vector<BodyVoxelKey> stack{*unseen.begin()};
      unseen.erase(stack.back());
      std::vector<BodyVoxelKey> component;
      BodyVoxelKey minimum = stack.back();
      BodyVoxelKey maximum = stack.back();

      // 使用显式栈做深度优先搜索；minimum/maximum 同时记录包围盒。
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

      // 把体素包围盒恢复成米。过矮、过高、过宽或规模异常的组件均拒绝。
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

      // 通过尺寸检查后，它才成为当前时间段的人体遮罩。
      ++result.components_accepted;
      accepted_mask.insert(component.begin(), component.end());
      result.body_candidate_voxels.insert(component.begin(), component.end());
    }

    // 学习结果只应用到对应的 stride 帧，不把一个局部人体位置套到整段 bag。
    for (std::size_t frame_index = apply_begin; frame_index < apply_end;
         ++frame_index) {
      result.frame_masks[frame_index].insert(accepted_mask.begin(),
                                             accepted_mask.end());
    }
  }

  result.near_field_voxels = all_near_field.size();
  result.persistent_voxels = all_persistent.size();
  result.pose_supported_voxels = all_pose_supported.size();
  // 阶段 3：建立地图坐标系的多视角静态保护。
  // 墙/柱/货架在车体系里不断移动，但乘 pose 后应重复落到同一地图体素。
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
      // 同一地图位置若从多个车辆位置、多个车体相对方位被看见，
      // 说明它更像固定环境，而不是始终贴着雷达的推车人。
      map_stat.pose_mask |= pose_bit;
      map_stat.body_voxel_mask |=
          hashedBit(BodyVoxelKeyHash{}(voxelKey(point)));
    }
  }

  // 同时通过“帧数、车辆位置数、车体相对位置数”三道门槛才进入保护集合。
  for (const auto &[key, map_stat] : map_stats) {
    if (static_cast<int>(map_stat.frame_count) >=
            config_.static_min_support_frames &&
        bitCount(map_stat.pose_mask) >= config_.static_min_pose_cells &&
        bitCount(map_stat.body_voxel_mask) >= config_.static_min_body_voxels) {
      result.static_map_voxels.insert(key);
    }
  }

  // 阶段 4：统计最终决策。真正导出点云在节点 save() 中完成；这里先统计：
  //   候选且命中 static_map_voxels -> 保护
  //   候选且未命中 static_map_voxels -> 删除
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
  // 必须查询“这一帧自己的遮罩”，而不是全局 body_candidate_voxels。
  return result.frame_masks[frame_index].count(voxelKey(point)) != 0U;
}

bool BodyFixedGhostFilter::isStaticProtected(const Point &point,
                                             const Eigen::Isometry3d &pose,
                                             const Result &result) const {
  // 先变换为地图体素，再查询是否具有多视角静态证据。
  return pcl::isFinite(point) &&
         result.static_map_voxels.count(mapVoxelKey(point, pose)) != 0U;
}

bool BodyFixedGhostFilter::isMasked(const Point &point, std::size_t frame_index,
                                    const Eigen::Isometry3d &pose,
                                    const Result &result) const {
  // 整个算法最终可以归结为这一行：
  // 删除 = 车体系人体候选 AND NOT 地图系静态结构。
  return isBodyCandidate(point, frame_index, result) &&
         !isStaticProtected(point, pose, result);
}

} // namespace pusher_ghost_filter
