#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "pusher_ghost_filter/body_fixed_ghost_filter.hpp"

namespace pusher_ghost_filter {

namespace {

Eigen::Isometry3d poseFromMessage(const nav_msgs::msg::Odometry &message) {
  const auto &position = message.pose.pose.position;
  const auto &orientation = message.pose.pose.orientation;
  Eigen::Quaterniond quaternion(orientation.w, orientation.x, orientation.y,
                                orientation.z);
  quaternion.normalize();

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = quaternion.toRotationMatrix();
  pose.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return pose;
}

double rotationAngle(const Eigen::Matrix3d &rotation) {
  return Eigen::AngleAxisd(rotation).angle();
}

struct VoxelKey {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const VoxelKey &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey &key) const {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
    return seed;
  }
};

struct VoxelAccumulator {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double intensity = 0.0;
  std::size_t count = 0U;
};

} // namespace

class PusherGhostFilterNode : public rclcpp::Node {
public:
  using Point = BodyFixedGhostFilter::Point;
  using Cloud = BodyFixedGhostFilter::Cloud;
  using Frame = BodyFixedGhostFilter::Frame;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;

  PusherGhostFilterNode()
      : Node("pusher_ghost_filter"), filter_(loadFilterConfig()) {
    cloud_topic_ =
        declare_parameter<std::string>("cloud_topic", "/cloud_registered_body");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/Odometry");
    keyframe_translation_ =
        declare_parameter<double>("keyframe_translation", 0.5);
    keyframe_rotation_ =
        declare_parameter<double>("keyframe_rotation_deg", 10.0) *
        3.14159265358979323846 / 180.0;
    keyframe_voxel_size_ =
        declare_parameter<double>("keyframe_voxel_size", 0.20);
    map_voxel_size_ = declare_parameter<double>("map_voxel_size", 0.08);
    map_output_path_ = declare_parameter<std::string>(
        "map_output_path", "/tmp/pusher_ghost_filter/cleaned_map.pcd");
    removed_output_path_ = declare_parameter<std::string>(
        "removed_output_path",
        "/tmp/pusher_ghost_filter/removed_candidates.pcd");
    protected_output_path_ = declare_parameter<std::string>(
        "protected_output_path",
        "/tmp/pusher_ghost_filter/static_protected_candidates.pcd");

    const auto input_qos =
        rclcpp::QoS(rclcpp::KeepLast(100)).reliable().get_rmw_qos_profile();
    cloud_subscriber_.subscribe(this, cloud_topic_, input_qos);
    odom_subscriber_.subscribe(this, odom_topic_, input_qos);
    synchronizer_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(100), cloud_subscriber_, odom_subscriber_);
    synchronizer_->registerCallback(
        std::bind(&PusherGhostFilterNode::measurementCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    save_service_ = create_service<std_srvs::srv::Trigger>(
        "~/save", std::bind(&PusherGhostFilterNode::save, this,
                            std::placeholders::_1, std::placeholders::_2));
    reset_service_ = create_service<std_srvs::srv::Trigger>(
        "~/reset", std::bind(&PusherGhostFilterNode::reset, this,
                             std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(),
                "Standalone pusher ghost filter started: cloud=%s odom=%s",
                cloud_topic_.c_str(), odom_topic_.c_str());
  }

private:
  BodyFixedGhostFilterConfig loadFilterConfig() {
    BodyFixedGhostFilterConfig config;
    config.voxel_size = declare_parameter<double>("filter.voxel_size", 0.20);
    config.min_radius = declare_parameter<double>("filter.min_radius", 0.25);
    config.max_radius = declare_parameter<double>("filter.max_radius", 3.0);
    config.window_size_frames =
        declare_parameter<int>("filter.window_size_frames", 60);
    config.window_stride_frames =
        declare_parameter<int>("filter.window_stride_frames", 30);
    config.min_keyframe_ratio =
        declare_parameter<double>("filter.min_keyframe_ratio", 0.12);
    config.min_support_frames =
        declare_parameter<int>("filter.min_support_frames", 8);
    config.pose_cell_size =
        declare_parameter<double>("filter.pose_cell_size", 2.0);
    config.min_pose_cells = declare_parameter<int>("filter.min_pose_cells", 4);
    config.yaw_bins = declare_parameter<int>("filter.yaw_bins", 12);
    config.min_yaw_bins = declare_parameter<int>("filter.min_yaw_bins", 1);
    config.ground_exclusion_half_height =
        declare_parameter<double>("filter.ground_exclusion_half_height", 0.25);
    config.min_component_voxels =
        declare_parameter<int>("filter.min_component_voxels", 4);
    config.max_component_voxels =
        declare_parameter<int>("filter.max_component_voxels", 3000);
    config.min_component_height =
        declare_parameter<double>("filter.min_component_height", 0.50);
    config.max_component_height =
        declare_parameter<double>("filter.max_component_height", 2.80);
    config.max_component_width =
        declare_parameter<double>("filter.max_component_width", 1.80);
    config.static_map_voxel_size =
        declare_parameter<double>("filter.static_map_voxel_size", 0.30);
    config.static_min_support_frames =
        declare_parameter<int>("filter.static_min_support_frames", 3);
    config.static_pose_cell_size =
        declare_parameter<double>("filter.static_pose_cell_size", 1.0);
    config.static_min_pose_cells =
        declare_parameter<int>("filter.static_min_pose_cells", 2);
    config.static_min_body_voxels =
        declare_parameter<int>("filter.static_min_body_voxels", 3);
    return config;
  }

  bool isKeyframe(const Eigen::Isometry3d &pose) const {
    if (frames_.empty()) {
      return true;
    }
    const Eigen::Isometry3d delta = frames_.back().pose.inverse() * pose;
    return delta.translation().norm() >= keyframe_translation_ ||
           rotationAngle(delta.rotation()) >= keyframe_rotation_;
  }

  Cloud::Ptr downsample(const Cloud::Ptr &input, double leaf_size) const {
    Cloud::Ptr finite(new Cloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*input, *finite, indices);
    if (leaf_size <= 0.0) {
      return finite;
    }
    Cloud::Ptr output(new Cloud);
    pcl::VoxelGrid<Point> voxel_filter;
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.setInputCloud(finite);
    voxel_filter.filter(*output);
    return output;
  }

  Cloud::Ptr globalDownsample(const Cloud::Ptr &input, double leaf_size) const {
    if (leaf_size <= 0.0) {
      return input;
    }
    std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxels;
    voxels.reserve(input->size());
    const double inverse = 1.0 / leaf_size;
    for (const auto &point : input->points) {
      if (!pcl::isFinite(point)) {
        continue;
      }
      const VoxelKey key{
          static_cast<std::int64_t>(std::floor(point.x * inverse)),
          static_cast<std::int64_t>(std::floor(point.y * inverse)),
          static_cast<std::int64_t>(std::floor(point.z * inverse))};
      auto &accumulator = voxels[key];
      accumulator.x += point.x;
      accumulator.y += point.y;
      accumulator.z += point.z;
      accumulator.intensity += point.intensity;
      ++accumulator.count;
    }

    Cloud::Ptr output(new Cloud);
    output->reserve(voxels.size());
    for (const auto &[key, accumulator] : voxels) {
      (void)key;
      const double scale = 1.0 / static_cast<double>(accumulator.count);
      output->push_back(
          Point{static_cast<float>(accumulator.x * scale),
                static_cast<float>(accumulator.y * scale),
                static_cast<float>(accumulator.z * scale),
                static_cast<float>(accumulator.intensity * scale)});
    }
    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1U;
    output->is_dense = true;
    return output;
  }

  void measurementCallback(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_message,
      const nav_msgs::msg::Odometry::ConstSharedPtr &odometry) {
    const Eigen::Isometry3d pose = poseFromMessage(*odometry);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!isKeyframe(pose)) {
      return;
    }

    Cloud::Ptr raw_cloud(new Cloud);
    pcl::fromROSMsg(*cloud_message, *raw_cloud);
    Cloud::Ptr cloud = downsample(raw_cloud, keyframe_voxel_size_);
    if (cloud->empty()) {
      return;
    }
    frames_.push_back(Frame{cloud, pose});
    RCLCPP_DEBUG(get_logger(), "Stored keyframe %zu (%zu points)",
                 frames_.size() - 1U, cloud->size());
  }

  static bool prepareOutputPath(const std::string &path) {
    if (path.empty()) {
      return false;
    }
    const std::filesystem::path output(path);
    if (output.has_parent_path()) {
      std::filesystem::create_directories(output.parent_path());
    }
    return true;
  }

  void save(const std_srvs::srv::Trigger::Request::ConstSharedPtr,
            std_srvs::srv::Trigger::Response::SharedPtr response) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frames_.empty()) {
      response->success = false;
      response->message = "No synchronized keyframes available";
      return;
    }

    const BodyFixedGhostFilter::Result detection = filter_.detect(frames_);
    Cloud::Ptr kept(new Cloud);
    Cloud::Ptr removed(new Cloud);
    Cloud::Ptr protected_points(new Cloud);
    for (std::size_t frame_index = 0U; frame_index < frames_.size();
         ++frame_index) {
      const auto &frame = frames_[frame_index];
      Cloud kept_body;
      Cloud removed_body;
      Cloud protected_body;
      for (const auto &point : frame.cloud->points) {
        const bool candidate =
            filter_.isBodyCandidate(point, frame_index, detection);
        const bool static_protected =
            candidate &&
            filter_.isStaticProtected(point, frame.pose, detection);
        if (candidate && !static_protected) {
          removed_body.push_back(point);
        } else {
          kept_body.push_back(point);
          if (static_protected) {
            protected_body.push_back(point);
          }
        }
      }
      // PCL 1.12 divides by the input cloud width while assigning an empty
      // transform result. Avoid passing empty candidate branches to it.
      if (!kept_body.empty()) {
        Cloud transformed;
        pcl::transformPointCloud(kept_body, transformed,
                                 frame.pose.matrix().cast<float>());
        *kept += transformed;
      }
      if (!removed_body.empty()) {
        Cloud transformed;
        pcl::transformPointCloud(removed_body, transformed,
                                 frame.pose.matrix().cast<float>());
        *removed += transformed;
      }
      if (!protected_body.empty()) {
        Cloud transformed;
        pcl::transformPointCloud(protected_body, transformed,
                                 frame.pose.matrix().cast<float>());
        *protected_points += transformed;
      }
    }

    kept = globalDownsample(kept, map_voxel_size_);
    removed = globalDownsample(removed, map_voxel_size_);
    protected_points = globalDownsample(protected_points, map_voxel_size_);
    if (!prepareOutputPath(map_output_path_) ||
        !prepareOutputPath(removed_output_path_) ||
        !prepareOutputPath(protected_output_path_)) {
      response->success = false;
      response->message = "One or more output paths are empty";
      return;
    }

    const int map_result = pcl::io::savePCDFileBinary(map_output_path_, *kept);
    const int removed_result =
        pcl::io::savePCDFileBinary(removed_output_path_, *removed);
    const int protected_result =
        pcl::io::savePCDFileBinary(protected_output_path_, *protected_points);
    response->success =
        map_result == 0 && removed_result == 0 && protected_result == 0;
    response->message =
        "frames=" + std::to_string(frames_.size()) +
        " cleaned=" + std::to_string(kept->size()) +
        " removed=" + std::to_string(removed->size()) +
        " protected=" + std::to_string(protected_points->size()) +
        " ground_z=" + std::to_string(detection.estimated_ground_z);
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
  }

  void reset(const std_srvs::srv::Trigger::Request::ConstSharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t count = frames_.size();
    frames_.clear();
    response->success = true;
    response->message = "Cleared " + std::to_string(count) + " keyframes";
  }

  std::string cloud_topic_;
  std::string odom_topic_;
  std::string map_output_path_;
  std::string removed_output_path_;
  std::string protected_output_path_;
  double keyframe_translation_ = 0.5;
  double keyframe_rotation_ = 10.0 * 3.14159265358979323846 / 180.0;
  double keyframe_voxel_size_ = 0.20;
  double map_voxel_size_ = 0.08;

  BodyFixedGhostFilter filter_;
  std::vector<Frame> frames_;
  std::mutex mutex_;

  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_subscriber_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_subscriber_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
};

} // namespace pusher_ghost_filter

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pusher_ghost_filter::PusherGhostFilterNode>());
  rclcpp::shutdown();
  return 0;
}
