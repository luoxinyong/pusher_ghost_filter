#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Geometry>

#include "pusher_ghost_filter/body_fixed_ghost_filter.hpp"

namespace pusher_ghost_filter {
namespace {

BodyFixedGhostFilterConfig testConfig() {
  BodyFixedGhostFilterConfig config;
  config.voxel_size = 0.20;
  config.window_size_frames = 20;
  config.window_stride_frames = 10;
  config.min_keyframe_ratio = 0.25;
  config.min_support_frames = 5;
  config.pose_cell_size = 0.5;
  config.min_pose_cells = 4;
  config.yaw_bins = 8;
  config.min_yaw_bins = 1;
  config.ground_exclusion_half_height = 0.20;
  config.min_component_voxels = 3;
  config.min_component_height = 0.40;
  config.max_component_width = 1.80;
  config.static_map_voxel_size = 0.30;
  config.static_min_support_frames = 3;
  config.static_pose_cell_size = 0.25;
  config.static_min_pose_cells = 2;
  config.static_min_body_voxels = 2;
  return config;
}

Eigen::Isometry3d makePose(double x) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation().x() = x;
  return pose;
}

void addGround(BodyFixedGhostFilter::Cloud &cloud) {
  for (double x = -2.0; x <= 2.0; x += 0.4) {
    for (double y = -2.0; y <= 2.0; y += 0.4) {
      cloud.push_back(pcl::PointXYZI{static_cast<float>(x),
                                     static_cast<float>(y), -1.0F, 1.0F});
    }
  }
}

void addHuman(BodyFixedGhostFilter::Cloud &cloud, float x, float y) {
  for (float z = -0.6F; z <= 0.8F; z += 0.2F) {
    cloud.push_back(pcl::PointXYZI{x, y, z, 10.0F});
    cloud.push_back(pcl::PointXYZI{x, y + 0.2F, z, 10.0F});
  }
}

TEST(BodyFixedGhostFilter, RemovesBodyFixedHumanButRejectsGround) {
  const BodyFixedGhostFilterConfig config = testConfig();
  BodyFixedGhostFilter filter(config);

  std::vector<BodyFixedGhostFilter::Frame> frames;
  for (int frame_index = 0; frame_index < 40; ++frame_index) {
    auto cloud = std::make_shared<BodyFixedGhostFilter::Cloud>();
    addGround(*cloud);
    addHuman(*cloud, -1.0F, 0.0F);
    frames.push_back(
        BodyFixedGhostFilter::Frame{cloud, makePose(0.5 * frame_index)});
  }

  const auto result = filter.detect(frames);
  ASSERT_EQ(result.frame_masks.size(), frames.size());
  EXPECT_GT(result.body_candidate_voxels.size(), 0U);
  EXPECT_GE(result.components_accepted, 1U);
  EXPECT_TRUE(filter.isMasked(pcl::PointXYZI{-1.0F, 0.0F, 0.4F, 0.0F}, 10U,
                              frames[10].pose, result));
  EXPECT_FALSE(filter.isMasked(pcl::PointXYZI{0.0F, 0.0F, -1.0F, 0.0F}, 10U,
                               frames[10].pose, result));
}

TEST(BodyFixedGhostFilter, ProtectsParallelWallWithMultiViewSupport) {
  const BodyFixedGhostFilterConfig config = testConfig();
  BodyFixedGhostFilter filter(config);

  std::vector<BodyFixedGhostFilter::Frame> frames;
  for (int frame_index = 0; frame_index < 30; ++frame_index) {
    const double pose_x = 0.2 * frame_index;
    auto cloud = std::make_shared<BodyFixedGhostFilter::Cloud>();
    addGround(*cloud);
    addHuman(*cloud, -1.0F, 0.0F);

    for (double map_x = -1.0; map_x <= 7.0; map_x += 0.2) {
      if (std::abs(map_x - pose_x) > 0.61) {
        continue;
      }
      for (float z = -0.6F; z <= 0.8F; z += 0.2F) {
        cloud->push_back(
            pcl::PointXYZI{static_cast<float>(map_x - pose_x), 1.5F, z, 20.0F});
      }
    }
    frames.push_back(BodyFixedGhostFilter::Frame{cloud, makePose(pose_x)});
  }

  const auto result = filter.detect(frames);
  const pcl::PointXYZI human{-1.0F, 0.0F, 0.4F, 0.0F};
  const pcl::PointXYZI wall{0.0F, 1.5F, 0.4F, 0.0F};
  EXPECT_TRUE(filter.isMasked(human, 10U, frames[10].pose, result));
  EXPECT_FALSE(filter.isMasked(wall, 10U, frames[10].pose, result));
  EXPECT_GT(result.static_map_voxels.size(), 0U);
}

TEST(BodyFixedGhostFilter, TracksHumanPositionChangesByTimeWindow) {
  const BodyFixedGhostFilterConfig config = testConfig();
  BodyFixedGhostFilter filter(config);

  std::vector<BodyFixedGhostFilter::Frame> frames;
  for (int frame_index = 0; frame_index < 40; ++frame_index) {
    auto cloud = std::make_shared<BodyFixedGhostFilter::Cloud>();
    addGround(*cloud);
    if (frame_index < 20) {
      addHuman(*cloud, -1.0F, 0.0F);
    } else {
      addHuman(*cloud, 1.0F, 0.0F);
    }
    frames.push_back(
        BodyFixedGhostFilter::Frame{cloud, makePose(0.5 * frame_index)});
  }

  const auto result = filter.detect(frames);
  const pcl::PointXYZI early_human{-1.0F, 0.0F, 0.4F, 0.0F};
  const pcl::PointXYZI late_human{1.0F, 0.0F, 0.4F, 0.0F};
  EXPECT_TRUE(filter.isMasked(early_human, 5U, frames[5].pose, result));
  EXPECT_TRUE(filter.isMasked(late_human, 35U, frames[35].pose, result));
  EXPECT_FALSE(filter.isMasked(early_human, 35U, frames[35].pose, result));
}

TEST(BodyFixedGhostFilter, EmptyFramesProduceEmptyMasks) {
  BodyFixedGhostFilter filter(BodyFixedGhostFilterConfig{});
  const auto result = filter.detect({});
  EXPECT_TRUE(result.frame_masks.empty());
  EXPECT_TRUE(result.body_candidate_voxels.empty());
  EXPECT_TRUE(result.static_map_voxels.empty());
}

} // namespace
} // namespace pusher_ghost_filter
