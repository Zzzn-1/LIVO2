/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <unistd.h>
#include <Eigen/Geometry>
#include <fast_livo/Keyframe.h>
#include <fast_livo/LoopConstraint.h>

namespace
{
class FilteringCoutBuf : public std::streambuf
{
public:
  FilteringCoutBuf(std::streambuf *dest, bool keep_loop_event_only)
      : dest_(dest), keep_loop_event_only_(keep_loop_event_only)
  {
  }

protected:
  int overflow(int c) override
  {
    if (c == EOF)
    {
      return !EOF;
    }
    line_buffer_.push_back(static_cast<char>(c));
    if (c == '\n')
    {
      flushLine();
    }
    return c;
  }

  int sync() override
  {
    flushLine();
    return 0;
  }

private:
  void flushLine()
  {
    if (line_buffer_.empty())
    {
      return;
    }
    if (!keep_loop_event_only_ ||
        line_buffer_.find("[SemanticLoop] LOOP_EVENT") != std::string::npos)
    {
      dest_->sputn(line_buffer_.data(), line_buffer_.size());
      dest_->pubsync();
    }
    line_buffer_.clear();
  }

  std::streambuf *dest_ = nullptr;
  bool keep_loop_event_only_ = true;
  std::string line_buffer_;
};

std::streambuf *g_original_cout_buf = nullptr;
std::unique_ptr<FilteringCoutBuf> g_filtering_cout_buf;

class ConsoleFdFilter
{
public:
  void start(bool keep_loop_event_only)
  {
    if (started_)
    {
      return;
    }

    keep_loop_event_only_ = keep_loop_event_only;
    fflush(stdout);
    fflush(stderr);

    saved_stdout_fd_ = dup(STDOUT_FILENO);
    saved_stderr_fd_ = dup(STDERR_FILENO);
    if (saved_stdout_fd_ < 0 || saved_stderr_fd_ < 0)
    {
      return;
    }

    if (pipe(pipe_fd_) != 0)
    {
      return;
    }

    dup2(pipe_fd_[1], STDOUT_FILENO);
    dup2(pipe_fd_[1], STDERR_FILENO);
    close(pipe_fd_[1]);
    pipe_fd_[1] = -1;
    started_ = true;

    worker_ = std::thread(&ConsoleFdFilter::readLoop, this);
    worker_.detach();
  }

private:
  void readLoop()
  {
    char buffer[512];
    while (true)
    {
      const ssize_t n = read(pipe_fd_[0], buffer, sizeof(buffer));
      if (n <= 0)
      {
        break;
      }
      line_buffer_.append(buffer, static_cast<size_t>(n));
      flushCompleteLines();
    }
  }

  void flushCompleteLines()
  {
    size_t newline_pos = std::string::npos;
    while ((newline_pos = line_buffer_.find('\n')) != std::string::npos)
    {
      const std::string line = line_buffer_.substr(0, newline_pos + 1);
      line_buffer_.erase(0, newline_pos + 1);
      if (!keep_loop_event_only_ ||
          line.find("[SemanticLoop] LOOP_EVENT") != std::string::npos)
      {
        const ssize_t unused = write(saved_stdout_fd_, line.data(), line.size());
        (void)unused;
      }
    }
  }

  bool started_ = false;
  bool keep_loop_event_only_ = true;
  int pipe_fd_[2] = {-1, -1};
  int saved_stdout_fd_ = -1;
  int saved_stderr_fd_ = -1;
  std::thread worker_;
  std::string line_buffer_;
};

ConsoleFdFilter g_console_fd_filter;

void configureConsoleOutputFilter(bool verbose_console_log, bool keep_loop_event_when_quiet)
{
  if (g_original_cout_buf == nullptr)
  {
    g_original_cout_buf = std::cout.rdbuf();
  }

  if (verbose_console_log)
  {
    if (std::cout.rdbuf() != g_original_cout_buf)
    {
      std::cout.rdbuf(g_original_cout_buf);
    }
    g_filtering_cout_buf.reset();
    return;
  }

  g_console_fd_filter.start(keep_loop_event_when_quiet);
  g_filtering_cout_buf.reset(new FilteringCoutBuf(g_original_cout_buf, keep_loop_event_when_quiet));
  std::cout.rdbuf(g_filtering_cout_buf.get());
}
} // namespace

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),//平移
      extR(M3D::Identity())//旋转
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);
  
  // 初始化SETIC相关变量
  last_timestamp_img_setic = -1.0;
  setic_processing = false;
  setic_skip_count = 0;
  max_setic_skip = 5;
  setic_timeout_ms = 50;
  last_successful_setic_time = 0.0;
  setic_use_semantic_if_valid = true;
  setic_min_nonzero_pixels = 200;
  setic_semantic_valid_timeout = 0.5;
  setic_fallback_mode_ = false;
  setic_invalid_frame_count_ = 0;
  setic_valid_frame_count_ = 0;
  last_valid_semantic_time_ = -1.0;

  // RGB缓存大小（回环检索使用）
  max_buffer_size_ = 20;

  p_pre.reset(new Preprocess()); // LiDAR预处理
  p_imu.reset(new ImuProcess()); // IMU数据

  readParameters(nh);
  configureConsoleOutputFilter(verbose_console_log_, keep_loop_event_log_when_quiet_);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  setic_manager.reset(new seticmanager()); // 初始化SETIC管理器
  setic_voxelmap_manager.reset(new VoxelMapManager(voxel_config, setic_voxel_map)); // 初始化SETIC体素地图管理器
  root_dir = ROOT_DIR;
  initializeFiles();//初始化文件
  initializeComponents();//初始化组成
  if (semantic_loop_enable_)
  {
    SemanticLoopConfig config;
    config.top_k = semantic_top_k_;
    config.score_threshold = semantic_score_threshold_;
    config.max_queue_size = semantic_max_queue_size_;
    config.min_candidate_time_gap = semantic_candidate_min_time_gap_;
    config.min_candidate_path_distance = semantic_candidate_min_path_distance_;
    config.min_candidate_euclidean_distance = semantic_candidate_min_euclidean_distance_;
    config.max_candidate_euclidean_distance = semantic_candidate_max_euclidean_distance_;
    config.topology_verify_enable = semantic_topology_verify_enable_;
    config.topology_verify_top_n = semantic_topology_verify_top_n_;
    config.topology_min_nodes = semantic_topology_min_nodes_;
    config.topology_min_edges = semantic_topology_min_edges_;
    config.topology_max_edge_distance = semantic_topology_max_edge_distance_;
    config.topology_edge_tolerance = semantic_topology_edge_tolerance_;
    config.topology_score_threshold = semantic_topology_score_threshold_;
    config.topology_fallback_enable = semantic_topology_fallback_enable_;
    config.topology_fallback_voxel_size = semantic_topology_fallback_voxel_size_;
    config.topology_fallback_inlier_ratio_threshold =
        semantic_topology_fallback_inlier_ratio_threshold_;
    config.topology_fallback_min_inliers = semantic_topology_fallback_min_inliers_;
    config.icp_verify_enable = semantic_icp_verify_enable_;
    config.icp_verify_top_n = semantic_icp_verify_top_n_;
    config.icp_min_inliers = semantic_icp_min_inliers_;
    config.icp_downsample_leaf_size = semantic_icp_downsample_leaf_size_;
    config.icp_max_correspondence_distance = semantic_icp_max_correspondence_distance_;
    config.icp_max_iterations = semantic_icp_max_iterations_;
    config.icp_fitness_threshold = semantic_icp_fitness_threshold_;
    config.icp_use_local_cloud_for_relative_pose = semantic_icp_use_local_cloud_for_relative_pose_;
    config.verbose_debug_log = semantic_verbose_debug_log_;
    config.print_top_candidates = semantic_print_top_candidates_;
    config.print_icp_detail = semantic_print_icp_detail_;
    config.save_loop_event_pcd = semantic_save_loop_event_pcd_;
    config.max_geometric_accepts_per_query = semantic_max_geometric_accepts_per_query_;
    config.accept_cooldown_time_gap = semantic_accept_cooldown_time_gap_;
    config.accept_cooldown_path_gap = semantic_accept_cooldown_path_gap_;
    config.global_loop_cooldown_time_gap = semantic_global_loop_cooldown_time_gap_;
    config.global_loop_cooldown_path_gap = semantic_global_loop_cooldown_path_gap_;
    config.global_loop_cooldown_match_id_window = semantic_global_loop_cooldown_match_id_window_;
    config.max_relative_rotation_deg = semantic_max_relative_rotation_deg_;
    semantic_loop_manager_.reset(new SemanticLoopManager(config));
    semantic_loop_manager_->setLoopEventCallback([this](const SemanticLoopManager::LoopEvent &event) {
      std_msgs::String msg;
      std::ostringstream ss;
      ss << "query=" << event.query_id
         << ",match=" << event.match_id
         << ",rank=" << event.rank
         << ",score=" << std::fixed << std::setprecision(3) << event.score
         << ",dt=" << std::setprecision(2) << event.dt
         << ",path=" << event.path_dist
         << ",euclid=" << event.euclidean_dist
         << ",topology_edges=" << event.geo_inliers
         << ",topology_score=" << std::setprecision(3) << event.geo_ratio
         << ",icp_fitness=" << event.icp_fitness
         << ",query_time=" << std::setprecision(6) << event.query_time
         << ",query_travel=" << std::setprecision(3) << event.query_travel
         << ",query_pos=(" << event.query_pos_x << "," << event.query_pos_y << "," << event.query_pos_z << ")"
         << ",match_time=" << std::setprecision(6) << event.match_time
         << ",match_travel=" << std::setprecision(3) << event.match_travel
         << ",match_pos=(" << event.match_pos_x << "," << event.match_pos_y << "," << event.match_pos_z << ")"
         << ",rel_source=" << (event.rel_pose_from_icp ? "icp_lite" : "odom")
         << ",rel_pos=(" << event.rel_pos_x << "," << event.rel_pos_y << "," << event.rel_pos_z << ")"
         << ",rel_dist=" << event.rel_dist;
      msg.data = ss.str();
      if (pubSemanticLoopEvent)
      {
        pubSemanticLoopEvent.publish(msg);
      }
      if (pubSemanticPgLoopConstraint)
      {
        if (event.match_id < 0 || event.query_id < 0)
        {
          if (semantic_verbose_debug_log_)
          {
            std::cout << "[SemanticLoop] skip loop publish: invalid semantic keyframe id. "
                      << "query=" << event.query_id << " match=" << event.match_id << std::endl;
          }
          publishSemanticLoopMarkers(event);
          return;
        }
        fast_livo::LoopConstraint lc_msg;
        lc_msg.header.stamp = ros::Time::now();
        lc_msg.header.frame_id = semantic_pg_frame_id_;
        lc_msg.from_id = static_cast<uint32_t>(event.match_id);
        lc_msg.to_id = static_cast<uint32_t>(event.query_id);
        lc_msg.from_time = event.match_time;
        lc_msg.to_time = event.query_time;
        lc_msg.rel_translation.x = event.rel_pos_x;
        lc_msg.rel_translation.y = event.rel_pos_y;
        lc_msg.rel_translation.z = event.rel_pos_z;
        Eigen::Quaterniond q_rel(event.rel_quat_w, event.rel_quat_x, event.rel_quat_y, event.rel_quat_z);
        if (q_rel.norm() < 1e-9)
        {
          q_rel = Eigen::Quaterniond::Identity();
        }
        else
        {
          q_rel.normalize();
        }
        lc_msg.rel_rotation.x = q_rel.x();
        lc_msg.rel_rotation.y = q_rel.y();
        lc_msg.rel_rotation.z = q_rel.z();
        lc_msg.rel_rotation.w = q_rel.w();
        const auto clamp01 = [](double v) {
          return std::max(0.0, std::min(1.0, v));
        };
        const double icp_quality = event.icp_fitness < 0.0 ? 0.0 : 1.0 / (1.0 + event.icp_fitness);
        const double topology_quality =
            clamp01(event.geo_ratio / std::max(semantic_loop_info_geo_ratio_ref_, 1e-9));
        const double inlier_quality =
            clamp01(static_cast<double>(std::max(event.geo_inliers, 0)) /
                    std::max(semantic_loop_info_inlier_ref_, 1.0));
        const double pose_source_quality =
            event.rel_pose_from_icp ? 1.0 : clamp01(semantic_loop_info_non_icp_scale_);
        const double loop_quality = clamp01(event.score) *
                                    topology_quality *
                                    inlier_quality *
                                    clamp01(icp_quality / 0.8) *
                                    pose_source_quality;
        const double sigma_t = 1.0 - loop_quality * (1.0 - 0.25);
        const double sigma_z = 2.0 * sigma_t;
        constexpr double kDegToRad = 0.017453292519943295;
        const double sigma_yaw = (20.0 - loop_quality * (20.0 - 5.0)) * kDegToRad;
        const double sigma_roll_pitch = 2.0 * sigma_yaw;

        lc_msg.info_x = 1.0 / (sigma_t * sigma_t);
        lc_msg.info_y = 1.0 / (sigma_t * sigma_t);
        lc_msg.info_z = 1.0 / (sigma_z * sigma_z);
        lc_msg.info_roll = 1.0 / (sigma_roll_pitch * sigma_roll_pitch);
        lc_msg.info_pitch = 1.0 / (sigma_roll_pitch * sigma_roll_pitch);
        lc_msg.info_yaw = 1.0 / (sigma_yaw * sigma_yaw);
        lc_msg.score = event.score;
        lc_msg.geo_ratio = event.geo_ratio;
        lc_msg.geo_inliers = event.geo_inliers > 0 ? static_cast<uint32_t>(event.geo_inliers) : 0U;
        lc_msg.icp_fitness = event.icp_fitness;
        lc_msg.edge_type = "semantic_loop";
        lc_msg.source = event.rel_pose_from_icp ? "online_livmapper_icp_lite" : "online_livmapper_odom";
        pubSemanticPgLoopConstraint.publish(lc_msg);
      }
      publishSemanticLoopMarkers(event);
    });
    semantic_loop_manager_->start();
    std::cout << "[SemanticLoop] manager started, topic=" << static_semantic_topic_
              << ", top_k=" << semantic_top_k_
              << ", score_th=" << semantic_score_threshold_
              << ", dt_gap>=" << semantic_candidate_min_time_gap_
              << "s, path_gap>=" << semantic_candidate_min_path_distance_ << "m"
              << ", euclid_gap>=" << semantic_candidate_min_euclidean_distance_ << "m"
              << ", euclid_max<=" << semantic_candidate_max_euclidean_distance_ << "m"
              << ", topology_enable=" << semantic_topology_verify_enable_
              << ", topology_top_n=" << semantic_topology_verify_top_n_
              << ", topology_cluster_tol=" << semantic_topology_cluster_tolerance_
              << "m, topology_fragment_merge=" << semantic_topology_fragment_merge_distance_
              << "m, topology_clusters_per_class=" << semantic_topology_max_clusters_per_class_
              << ", topology_min_nodes=" << semantic_topology_min_nodes_
              << ", topology_min_edges=" << semantic_topology_min_edges_
              << ", topology_edge_max=" << semantic_topology_max_edge_distance_
              << "m, topology_edge_tol=" << semantic_topology_edge_tolerance_
              << "m, topology_score_th=" << semantic_topology_score_threshold_
              << ", topology_fallback=" << semantic_topology_fallback_enable_
              << ", fallback_voxel=" << semantic_topology_fallback_voxel_size_
              << "m, fallback_ratio_th=" << semantic_topology_fallback_inlier_ratio_threshold_
              << ", fallback_min_inliers=" << semantic_topology_fallback_min_inliers_
              << ", icp_enable=" << semantic_icp_verify_enable_
              << ", icp_top_n=" << semantic_icp_verify_top_n_
              << ", icp_min_inliers=" << semantic_icp_min_inliers_
              << ", icp_leaf=" << semantic_icp_downsample_leaf_size_
              << "m, icp_corr=" << semantic_icp_max_correspondence_distance_
              << "m, icp_iter=" << semantic_icp_max_iterations_
              << ", icp_fit_th=" << semantic_icp_fitness_threshold_
              << ", verbose_debug=" << semantic_verbose_debug_log_
              << ", print_top=" << semantic_print_top_candidates_
              << ", print_icp_detail=" << semantic_print_icp_detail_
              << ", max_geo_accepts=" << semantic_max_geometric_accepts_per_query_
              << ", accept_cooldown_dt=" << semantic_accept_cooldown_time_gap_ << "s"
              << ", accept_cooldown_path=" << semantic_accept_cooldown_path_gap_ << "m"
              << ", global_loop_cooldown_dt=" << semantic_global_loop_cooldown_time_gap_ << "s"
              << ", global_loop_cooldown_path=" << semantic_global_loop_cooldown_path_gap_ << "m"
              << ", global_loop_cooldown_match_window=" << semantic_global_loop_cooldown_match_id_window_
              << ", rot_gate<=" << semantic_max_relative_rotation_deg_ << "deg"
              << std::endl;
  }
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {
  if (semantic_loop_manager_)
  {
    semantic_loop_manager_->stop();
  }
}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");//LiDAR 数据的 ROS 话题名称，默认值为 /livox/lidar
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");//IMU 数据的 ROS 话题名称，默认值为 /livox/imu
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<bool>("common/verbose_console_log", verbose_console_log_, true);
  nh.param<bool>("common/keep_loop_event_log_when_quiet", keep_loop_event_log_when_quiet_, true);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1); // 是否启用 LiDAR，默认值为 1（启用）

  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");//图像数据的 ROS 话题名称，默认值为 /left_camera/image

  nh.param<string>("common/setic_topic", setic_topic, "/setic/image_raw");//语义部分订阅话题
  nh.param<int>("common/setic_en", setic_en, 1);

  // 🔧 新增：SETIC性能参数配置
  nh.param<double>("setic_performance/max_processing_time", setic_max_processing_time, 0.050); // 50ms
  nh.param<double>("setic_performance/warning_threshold", setic_warning_threshold, 0.030);     // 30ms
  nh.param<int>("setic_performance/adaptive_downsample_threshold", setic_adaptive_threshold, 5000); // 点云数量
  nh.param<bool>("setic_performance/enable_adaptive_processing", setic_adaptive_enable, true);
  nh.param<bool>("setic_performance/enable_performance_monitoring", setic_monitoring_enable, true);
  nh.param<int>("setic_performance/max_queue_size", setic_max_queue_size, 5);
  nh.param<int>("setic_performance/skip_visualization_interval", setic_viz_skip_interval, 5);
  nh.param<int>("setic_performance/skip_patch_update_interval", setic_patch_skip_interval, 3);
  nh.param<double>("setic_performance/adaptive_image_scale_threshold", setic_image_scale_threshold, 0.8);
  nh.param<bool>("setic/enable_heavy_voxel_update", setic_heavy_voxel_update_en_, false);
  nh.param<bool>("setic/semantic_loop_en", setic_semantic_loop_en_, false);
  nh.param<bool>("setic/use_semantic_if_valid", setic_use_semantic_if_valid, true);
  nh.param<int>("setic/min_nonzero_pixels", setic_min_nonzero_pixels, 200);
  nh.param<double>("setic/semantic_valid_timeout", setic_semantic_valid_timeout, 0.5);
  nh.param<bool>("setic/semantic_loop_en", setic_semantic_loop_en_, false);
  nh.param<bool>("setic/lio_dynamic_filter_en", lio_dynamic_filter_en_, true);
  nh.param<double>("setic/lio_semantic_sync_threshold", lio_semantic_sync_threshold_, 0.10);
  nh.param<bool>("semantic_loop/enable", semantic_loop_enable_, false);
  nh.param<string>("semantic_loop/static_semantic_topic", static_semantic_topic_, "/semantic/static_mask");
  nh.param<string>("semantic_loop/pg_frame_id", semantic_pg_frame_id_, "camera_init");
  nh.param<string>("semantic_loop/pg_keyframe_topic", semantic_pg_keyframe_topic_, "/semantic_pg/keyframe");
  nh.param<string>("semantic_loop/pg_loop_topic", semantic_pg_loop_topic_, "/semantic_pg/loop_constraint");
  nh.param<double>("semantic_loop/semantic_frame_min_translation", semantic_frame_min_translation_, 1.0);
  nh.param<double>("semantic_loop/semantic_frame_min_rotation_deg", semantic_frame_min_rotation_deg_, 10.0);
  nh.param<double>("semantic_loop/semantic_frame_min_interval", semantic_frame_min_interval_, 1.0);
  // Backward compatibility with old key names.
  nh.param<double>("semantic_loop/keyframe_min_translation", semantic_frame_min_translation_, semantic_frame_min_translation_);
  nh.param<double>("semantic_loop/keyframe_min_rotation_deg", semantic_frame_min_rotation_deg_, semantic_frame_min_rotation_deg_);
  nh.param<double>("semantic_loop/keyframe_min_interval", semantic_frame_min_interval_, semantic_frame_min_interval_);
  nh.param<int>("semantic_loop/min_static_pixels", min_static_pixels_, 300);
  nh.param<int>("semantic_loop/min_static_points", min_static_points_, 200);
  nh.param<int>("semantic_loop/semantic_top_k", semantic_top_k_, 5);
  nh.param<double>("semantic_loop/semantic_score_threshold", semantic_score_threshold_, 0.55);
  nh.param<int>("semantic_loop/max_queue_size", semantic_max_queue_size_, 3);
  nh.param<double>("semantic_loop/candidate_min_time_gap", semantic_candidate_min_time_gap_, 10.0);
  nh.param<double>("semantic_loop/candidate_min_path_distance", semantic_candidate_min_path_distance_, 20.0);
  nh.param<double>("semantic_loop/candidate_min_euclidean_distance", semantic_candidate_min_euclidean_distance_, 0.0);
  nh.param<double>("semantic_loop/candidate_max_euclidean_distance", semantic_candidate_max_euclidean_distance_, 0.0);
  // Backward compatibility: if old key exists, reuse it as path-distance threshold.
  nh.param<double>("semantic_loop/candidate_min_distance",semantic_candidate_min_path_distance_, semantic_candidate_min_path_distance_);
  nh.param<bool>("semantic_loop/topology_verify_enable", semantic_topology_verify_enable_, true);
  nh.param<int>("semantic_loop/topology_verify_top_n", semantic_topology_verify_top_n_, 3);
  nh.param<int>("semantic_loop/topology_min_component_pixels", semantic_topology_min_component_pixels_, 80);
  nh.param<int>("semantic_loop/topology_min_component_points", semantic_topology_min_component_points_, 8);
  nh.param<double>("semantic_loop/topology_cluster_tolerance", semantic_topology_cluster_tolerance_, 0.45);
  // Keep the old key as a fallback for existing configuration files.
  nh.param<int>("semantic_loop/topology_max_clusters_per_component",
                semantic_topology_max_clusters_per_class_, 3);
  nh.param<double>("semantic_loop/topology_fragment_merge_distance",
                   semantic_topology_fragment_merge_distance_, 0.60);
  nh.param<int>("semantic_loop/topology_max_clusters_per_class",
                semantic_topology_max_clusters_per_class_,
                semantic_topology_max_clusters_per_class_);
  nh.param<int>("semantic_loop/topology_min_nodes", semantic_topology_min_nodes_, 3);
  nh.param<int>("semantic_loop/topology_min_edges", semantic_topology_min_edges_, 2);
  nh.param<double>("semantic_loop/topology_max_edge_distance", semantic_topology_max_edge_distance_, 8.0);
  nh.param<double>("semantic_loop/topology_edge_tolerance", semantic_topology_edge_tolerance_, 0.5);
  nh.param<double>("semantic_loop/topology_score_threshold", semantic_topology_score_threshold_, 0.55);
  nh.param<bool>("semantic_loop/topology_fallback_enable",
                 semantic_topology_fallback_enable_, true);
  nh.param<double>("semantic_loop/topology_fallback_voxel_size",
                   semantic_topology_fallback_voxel_size_, 0.8);
  nh.param<double>("semantic_loop/topology_fallback_inlier_ratio_threshold",
                   semantic_topology_fallback_inlier_ratio_threshold_, 0.12);
  nh.param<int>("semantic_loop/topology_fallback_min_inliers",
                semantic_topology_fallback_min_inliers_, 80);
  // Backward compatibility with the original geometric verification keys.
  nh.param<double>("semantic_loop/geometric_voxel_size",
                   semantic_topology_fallback_voxel_size_,
                   semantic_topology_fallback_voxel_size_);
  nh.param<double>("semantic_loop/geometric_inlier_ratio_threshold",
                   semantic_topology_fallback_inlier_ratio_threshold_,
                   semantic_topology_fallback_inlier_ratio_threshold_);
  nh.param<int>("semantic_loop/geometric_min_inliers",
                semantic_topology_fallback_min_inliers_,
                semantic_topology_fallback_min_inliers_);
  nh.param<bool>("semantic_loop/icp_verify_enable", semantic_icp_verify_enable_, true);
  nh.param<int>("semantic_loop/icp_verify_top_n", semantic_icp_verify_top_n_, 3);
  nh.param<int>("semantic_loop/icp_min_inliers", semantic_icp_min_inliers_, 80);
  nh.param<double>("semantic_loop/icp_downsample_leaf_size", semantic_icp_downsample_leaf_size_, 0.5);
  nh.param<double>("semantic_loop/icp_max_correspondence_distance", semantic_icp_max_correspondence_distance_, 1.5);
  nh.param<int>("semantic_loop/icp_max_iterations", semantic_icp_max_iterations_, 30);
  nh.param<double>("semantic_loop/icp_fitness_threshold", semantic_icp_fitness_threshold_, 0.35);
  nh.param<bool>("semantic_loop/icp_use_local_cloud_for_relative_pose", semantic_icp_use_local_cloud_for_relative_pose_, true);
  nh.param<double>("semantic_loop/information_geo_ratio_ref", semantic_loop_info_geo_ratio_ref_, 0.8);
  nh.param<double>("semantic_loop/information_inlier_ref", semantic_loop_info_inlier_ref_, 2000.0);
  nh.param<double>("semantic_loop/information_non_icp_scale", semantic_loop_info_non_icp_scale_, 0.5);
  nh.param<bool>("semantic_loop/verbose_debug_log", semantic_verbose_debug_log_, false);
  nh.param<bool>("semantic_loop/print_top_candidates", semantic_print_top_candidates_, true);
  nh.param<bool>("semantic_loop/print_icp_detail", semantic_print_icp_detail_, false);
  nh.param<bool>("semantic_loop/save_loop_event_pcd", semantic_save_loop_event_pcd_, false);
  nh.param<int>("semantic_loop/max_geometric_accepts_per_query", semantic_max_geometric_accepts_per_query_, 1);
  nh.param<double>("semantic_loop/accept_cooldown_time_gap", semantic_accept_cooldown_time_gap_, 20.0);
  nh.param<double>("semantic_loop/accept_cooldown_path_gap", semantic_accept_cooldown_path_gap_, 8.0);
  nh.param<double>("semantic_loop/global_loop_cooldown_time_gap", semantic_global_loop_cooldown_time_gap_, 8.0);
  nh.param<double>("semantic_loop/global_loop_cooldown_path_gap", semantic_global_loop_cooldown_path_gap_, 5.0);
  nh.param<int>("semantic_loop/global_loop_cooldown_match_id_window", semantic_global_loop_cooldown_match_id_window_, 2);
  nh.param<double>("semantic_loop/max_relative_rotation_deg", semantic_max_relative_rotation_deg_, 0.0);

  nh.param<bool>("vio/normal_en", normal_en, true);//是否启用法向量计算，默认值为 true
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);//是否启用逆合成，默认值为 false
  nh.param<int>("vio/max_iterations", max_iterations, 5);//最大迭代次数，默认值为 5
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);//图像点的协方差，默认值为 100
  nh.param<bool>("vio/raycast_en", raycast_en, false);//是否启用射线投射，默认值为 false
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);//是否启用曝光估计，默认值为 true
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);//逆曝光协方差，默认值为 0.2
  nh.param<int>("vio/grid_size", grid_size, 5);//图像网格大小，默认值为 5
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);//图像网格高度，默认值为 17
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);//图像金字塔层数，默认值为 3
  nh.param<int>("vio/patch_size", patch_size, 8);//图像 patch 大小，默认值为 8
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);//异常值阈值，默认值为 1000

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);//初始曝光时间，默认值为 0.0
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);//图像时间戳的偏移量，默认值为 0.0
  nh.param<double>("time_offset/img_time_offset_setic", img_time_offset_setic, 0.0);//图像时间戳的偏移量，默认值为 0.0
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);//IMU 时间戳的偏移量，默认值为 0.0
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);//是否启用重力对齐，默认值为 false

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);//陀螺仪的协方差，默认值为 1.0
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);//加速度计的协方差，默认值为 1.0
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);//IMU 数据插值的帧数，默认值为 3
  nh.param<bool>("imu/imu_en", imu_en, false);//是否启用 IMU，默认值为 false
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);
/*
点云预处理
*/
  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);//盲区距离，默认值为 0.01
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);//点云降采样的最小体素大小，默认值为 0.5
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);//是否启用特征提取，默认值为 false

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);//点云保存的间隔，默认值为 -1（不保存）
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);//是否启用点云保存，默认值为 false
  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);//是否启用 COLMAP 输出，默认值为 false
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);//点云保存时的体素大小，默认值为 0.5
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());//LiDAR 到 IMU 的外部平移向量
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());//LiDAR 到 IMU 的外部旋转矩阵
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());//相机到 IMU 的外部平移向量
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());//相机到 IMU 的外部旋转矩阵

  /*
  调试与发布
  */
  nh.param<double>("debug/plot_time", plot_time, -10);//绘图时间  case LO:，默认值为 -10
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);//帧计数器，默认值为 6
/*
发布配置
*/
  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);//盲区 RGB 点的阈值，默认值为 0.01
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);//发布的点云帧数，默认值为 1
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);//是否发布有效点云，默认值为 false
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);//是否启用稠密地图，默认值为 false

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}
/*
体素降采样：通过downSizeFilterSurf减少LiDAR点数
VIO管理器：加载相机模型、设置算法参数（网格尺寸、补丁大小）、配置传感器外参
IMU处理器：设置噪声参数、启用/禁用估计功能（偏置、重力）
*/
void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);//点云降采样滤波器初始化 设置点云降采样滤波器的体素大小，用于减少点云数据量
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);//将 LiDAR 到 IMU 的外部平移 (extrinT) 和旋转 (extrinR) 参数赋值给 extT 和 extR

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);//同步设置体素地图管理器 (voxelmap_manager) 的外部参数，确保地图构建时坐标转换正确

  // 设置SETIC体素地图管理器的外参
  setic_voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  setic_voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);
/*
从 ROS 参数服务器的 laserMapping 命名空间加载相机模型参数到 VIO 管理器
*/
  if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam)) throw std::runtime_error("Camera model not correctly specified.");

  vio_manager->grid_size = grid_size;//图像网格大小，用于特征提取
  vio_manager->patch_size = patch_size;//图像 patch 大小，用于光流跟踪
  vio_manager->outlier_threshold = outlier_threshold;//异常值阈值，过滤错误匹配
  vio_manager->setImuToLidarExtrinsic(extT, extR);//设置 IMU 到 LiDAR 的外参
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);//设置 LiDAR 到相机的外参
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;//优化最大迭代次数
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;//是否启用曝光时间估计
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->initializeVIO();

  // 🔧 新增：初始化SETIC管理器的性能参数
  if (setic_manager) {
    setic_manager->cam = vio_manager->cam;
    setic_manager->pinhole_cam = vio_manager->pinhole_cam;
    setic_manager->state = &_state;
    setic_manager->state_propagat = &state_propagat;
    setic_manager->Rli = vio_manager->Rli;
    setic_manager->Rcl = vio_manager->Rcl;
    setic_manager->Rci = vio_manager->Rci;
    setic_manager->Pli = vio_manager->Pli;
    setic_manager->Pcl = vio_manager->Pcl;
    setic_manager->Pci = vio_manager->Pci;
    setic_manager->grid_size = vio_manager->grid_size;
    setic_manager->grid_n_height = vio_manager->grid_n_height;
    setic_manager->patch_size = vio_manager->patch_size;
    setic_manager->patch_pyrimid_level = vio_manager->patch_pyrimid_level;
    setic_manager->semantic_loop_en = setic_semantic_loop_en_;
    setic_manager->initializeVIO_setic();

    setic_manager->max_allowed_processing_time = setic_max_processing_time;
    setic_manager->warning_processing_time = setic_warning_threshold;
    
    std::cout << "[SETIC] Performance parameters configured:" << std::endl;
    std::cout << "  Max processing time: " << setic_max_processing_time * 1000 << " ms" << std::endl;
    std::cout << "  Warning threshold: " << setic_warning_threshold * 1000 << " ms" << std::endl;
    std::cout << "  Adaptive processing: " << (setic_adaptive_enable ? "enabled" : "disabled") << std::endl;
    std::cout << "  Performance monitoring: " << (setic_monitoring_enable ? "enabled" : "disabled") << std::endl;
  }

  p_imu->set_extrinsic(extT, extR);//IMU 到 LiDAR 的平移和旋转
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));//陀螺仪 (gyr_cov) 和加速度计 (acc_cov) 的协方差缩放因子
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));//陀螺仪和加速度计偏差的协方差（硬编码为 0.0001）
  p_imu->set_imu_init_frame_num(imu_int_frame);//imu_int_frame 控制 IMU 初始化所需的帧数

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();


/*
若同时启用图像和 LiDAR (img_en && lidar_en)，选择 LIVO（LiDAR-Inertial-Visual Odometry）
否则，若启用 IMU (imu_en)，选择 ONLY_LIO（仅 LiDAR-惯性里程计）。
否则，选择 ONLY_LO（仅 LiDAR 里程计）
*/
  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;//
}

void LIVMapper::initializeFiles() //初始化文件
{
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";//日志输出
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_interval > 0) fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);
}
/*
主函数首先运行的函数
*/
void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it) //初始化ros订阅部分
{
  sub_pcl = p_pre->lidar_type == AVIA ? //根据 LiDAR 类型选择不同的回调函数
            nh.subscribe(lid_topic, 800, &LIVMapper::livox_pcl_cbk, this) :
            nh.subscribe(lid_topic, 800, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 800, &LIVMapper::imu_cbk, this);//订阅 IMU 和图像数据
  sub_img = nh.subscribe(img_topic, 800, &LIVMapper::img_cbk, this);
  sub_setic = nh.subscribe(setic_topic, 800, &LIVMapper::setic_cbk, this); // 订阅SETIC语义图像
  if (semantic_loop_enable_)
  {
    sub_static_semantic = nh.subscribe(static_semantic_topic_, 100, &LIVMapper::static_semantic_cbk, this);
    std::cout << "[STATIC-SEM] subscribe topic: " << static_semantic_topic_ << std::endl;
    pubSemanticLoopEvent = nh.advertise<std_msgs::String>("/semantic_loop/event", 100);
    pubSemanticLoopMarkers = nh.advertise<visualization_msgs::MarkerArray>("/semantic_loop/markers", 10);
    pubSemanticPgKeyframe = nh.advertise<fast_livo::Keyframe>(semantic_pg_keyframe_topic_, 200);
    pubSemanticPgLoopConstraint = nh.advertise<fast_livo::LoopConstraint>(semantic_pg_loop_topic_, 200);
  }

  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);//发布配准后的点云（去畸变、降采样）
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);//发布有效特征点云（用于状态估计）
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);//发布全局地图点云
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);//发布优化后的里程计信息
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);//发布历史路径

  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);//发布平面法线（用于可视化平面特征）
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);//发布体素结构（调试体素地图）

  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);//兼容 MAVROS 的位姿信息

  pubImage = it.advertise("/rgb_img", 1);//发布处理后的 RGB 图像
  pubImage_setic = it.advertise("/setic_img", 1);//发布处理后的语义图像

  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);//发布 IMU 传播的中间里程计
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
  
  // 添加SETIC体素地图发布器
  setic_voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/setic_planes", 10000);
  pubSeticVoxelMap = nh.advertise<sensor_msgs::PointCloud2>("/setic_voxel_map", 100);
  
  if (setic_en) {
    std::cout << "[SETIC] Async worker removed; running mask-only mode for LIO filtering." << std::endl;
  }
}

void LIVMapper::publishSemanticLoopMarkers(const SemanticLoopManager::LoopEvent &event)
{
  if (!pubSemanticLoopMarkers)
  {
    return;
  }

  geometry_msgs::Point query_pt;
  query_pt.x = event.query_pos_x;
  query_pt.y = event.query_pos_y;
  query_pt.z = event.query_pos_z;

  geometry_msgs::Point match_pt;
  match_pt.x = event.match_pos_x;
  match_pt.y = event.match_pos_y;
  match_pt.z = event.match_pos_z;

  visualization_msgs::MarkerArray marker_array;
  {
    std::lock_guard<std::mutex> lock(semantic_loop_marker_mutex_);
    semantic_loop_marker_line_points_.push_back(query_pt);
    semantic_loop_marker_line_points_.push_back(match_pt);
    semantic_loop_marker_node_points_.push_back(query_pt);
    semantic_loop_marker_node_points_.push_back(match_pt);

    visualization_msgs::Marker lines;
    lines.header.stamp = ros::Time::now();
    lines.header.frame_id = "camera_init";
    lines.ns = "semantic_loop_edges";
    lines.id = 0;
    lines.type = visualization_msgs::Marker::LINE_LIST;
    lines.action = visualization_msgs::Marker::ADD;
    lines.pose.orientation.w = 1.0;
    lines.scale.x = 0.06;
    lines.color.r = 1.0;
    lines.color.g = 0.1;
    lines.color.b = 0.1;
    lines.color.a = 0.95;
    lines.points = semantic_loop_marker_line_points_;
    marker_array.markers.push_back(lines);

    visualization_msgs::Marker nodes;
    nodes.header = lines.header;
    nodes.ns = "semantic_loop_nodes";
    nodes.id = 1;
    nodes.type = visualization_msgs::Marker::SPHERE_LIST;
    nodes.action = visualization_msgs::Marker::ADD;
    nodes.pose.orientation.w = 1.0;
    nodes.scale.x = 0.12;
    nodes.scale.y = 0.12;
    nodes.scale.z = 0.12;
    nodes.color.r = 1.0;
    nodes.color.g = 0.6;
    nodes.color.b = 0.1;
    nodes.color.a = 0.9;
    nodes.points = semantic_loop_marker_node_points_;
    marker_array.markers.push_back(nodes);
  }

  pubSemanticLoopMarkers.publish(marker_array);
}

void LIVMapper::handleFirstFrame() //设置时间零点
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time; //_first_lidar_time 是整个系统的时间零点
        p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() //重力初始化及相关参数设置 processImu函数中引用
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() //imu进程
{

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();//引用上面的重力初始化

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

}

void LIVMapper::stateEstimationAndMapping() //状态估计 run函数中引用
{//过程：LIO->VIO->LIO->VIO =ESKF
  switch (LidarMeasures.lio_vio_flg) //lio-vio完成了第一次测量时执行
  {
    case VIO://VIO调用这个函数
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();//LIO调用这个函数
      break;
  }
}

void LIVMapper::handleVIO() //vio执行
{
  euler_cur = RotMtoEuler(_state.rot_end);// 从旋转矩阵提取欧拉角
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) //确保待处理的点云数据有效，避免空指针或空点云导致后续崩溃
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;//输出当前帧的原始特征点数量，用于调试或监控特征提取性能

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) //根据时间差动态启用/禁用可视化绘图功能
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }
  //开启vio.cpp中的进程
  const MeasureGroup &cur_meas = LidarMeasures.measures.back();
  vio_manager->setSemanticMask(cur_meas.img_setic, cur_meas.setic_time, cur_meas.vio_time);
  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);

  if (imu_prop_enable) //若启用IMU传播，更新扩展卡尔曼滤波（EKF）状态
  {
    ekf_finish_once = true;//标记EKF完成
    latest_ekf_state = _state;// 更新最新状态
    latest_ekf_time = LidarMeasures.last_lio_update_time;// 记录时间戳
    state_update_flg = true;// 触发状态更新标志
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++) 
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);//// 发布全局点云
  publish_img_rgb(pubImage, vio_manager);// 发布RGB图像
}//点云与图像发布：将处理后的点云和图像发送到ROS话题


/*
LIO使用时调用的代码
lio_vio_flg=LO时使用
*/
void LIVMapper::handleLIO()
{    
  euler_cur = RotMtoEuler(_state.rot_end);//将旋转矩阵转换为欧拉角（通常为Z-Y-X顺序）
           
  if (feats_undistort->empty() || (feats_undistort == nullptr)) //确保去畸变后的点云数据有效，避免空数据导致后续崩溃
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);//通过体素滤波减少点云密度，提升后续处理效率
  downSizeFilterSurf.filter(*feats_down_body);//filter_size_surf_min 控制下采样体素尺寸（例如0.5米）
  if (lidar_map_inited)
  {
    PointCloudXYZI::Ptr static_feats_down_body(new PointCloudXYZI());
    int removed_dynamic_points_before_lio = 0;
    filterDynamicCloudBeforeLio(
        feats_down_body,
        LidarMeasures.last_lio_update_time,
        _state,
        static_feats_down_body,
        removed_dynamic_points_before_lio);
    *feats_down_body = *static_feats_down_body;
    if (removed_dynamic_points_before_lio > 0)
    {
      std::cout << "[ LIO ] Dynamic points removed before state estimation: "
                << removed_dynamic_points_before_lio << " / "
                << (feats_down_body->size() + removed_dynamic_points_before_lio) << std::endl;
    }
  }

  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);//将降采样后的点云从LiDAR坐标系转换到世界坐标系
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) //首次运行时构建体素地图，存储环境结构信息。依赖条件：lidar_map_inited 标志确保仅初始化一次
  {
    lidar_map_inited = true;
    /*
    在voxel_map.cpp文件中
    */
    voxelmap_manager->BuildVoxelMap();//构建新的体素地图
  }

  double t1 = omp_get_wtime();
/*
具体在voxel_map函数中 调用状态估计函数进行下一部分调用
在体素地图构造中引用
*/
  voxelmap_manager->StateEstimation(state_propagat);//通过点云与体素地图匹配（如ICP算法），优化LiDAR位姿
  _state = voxelmap_manager->state_;
  {
    std::lock_guard<std::mutex> pv_lock(pv_list_mutex_);
    _pv_list = voxelmap_manager->pv_list_;
  }

  double t2 = omp_get_wtime();

  if (imu_prop_enable) //若启用，更新扩展卡尔曼滤波（EKF）状态，用于高频位姿输出
  {
    ekf_finish_once = true;//ekf_finish_once 表示EKF已完成一次更新，可用于后续IMU插值
    latest_ekf_state = _state;//将优化后的状态 _state 保存到 latest_ekf_state，供IMU传播线程使用
    latest_ekf_time = LidarMeasures.last_lio_update_time;//latest_ekf_time 确保IMU数据与LiDAR时间戳对齐
    state_update_flg = true;
  }

  if (pose_output_en)//将优化后的位姿（位置+四元数）写入 seq_name.txt，用于后续评测（如EVO工具）
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);//欧拉角转四元数：使用ROS的tf库生成四元数消息
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);//将位姿、速度、协方差封装为nav_msgs/Odometry消息，通过pubOdomAftMapped话题发布

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  //将当前帧的点云从LiDAR坐标系转换到世界坐标系
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);// 点云从LiDAR系到世界系的变换
  for (size_t i = 0; i < world_lidar->points.size(); i++) //协方差传播：考虑LiDAR-IMU外参 extR 和状态协方差
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    //协方差传播公式
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;//更新点的协方差
  }
  /*
  重要的步骤 维护地图
  雷达  ->  计算出与世界点云的协方差  ->  体素地图更准 
  voxel_map.cpp函数中调用的更新体素地图的函数 进行插帧操作
  */
  std::vector<pointWithVar> static_points_for_map;
  int removed_dynamic_points = 0;
  filterDynamicPointsBeforeMapUpdate(
      voxelmap_manager->pv_list_,
      LidarMeasures.last_lio_update_time,
      _state,
      static_points_for_map,
      removed_dynamic_points);

  voxelmap_manager->UpdateVoxelMap(static_points_for_map);// 插入当前帧静态点云到地图
  if (removed_dynamic_points > 0)
  {
    std::cout << "[ LIO ] Dynamic points removed before map update: " << removed_dynamic_points
              << " / " << voxelmap_manager->pv_list_.size() << std::endl;
  }
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  PointCloudXYZI::Ptr static_world_cloud(new PointCloudXYZI());
  static_world_cloud->reserve(static_points_for_map.size());
  for (const auto &pv : static_points_for_map)
  {
    PointType point;
    point.x = pv.point_w[0];
    point.y = pv.point_w[1];
    point.z = pv.point_w[2];
    point.intensity = 0.0f;
    static_world_cloud->push_back(point);
  }
  tryCreateSemanticKeyframe(static_world_cloud, LidarMeasures.last_lio_update_time, _state);
  {
    std::lock_guard<std::mutex> pv_lock(pv_list_mutex_);
    _pv_list = static_points_for_map;
  }
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    /*
    调用voxel_map检查是否超出
    如果没超出则更新地图 超出就删除
    判断当前位置 position_last_ 是否移动足够远，如果是，则更新地图
    */
    voxelmap_manager->mapSliding();// 移除超出滑动窗口的区域
  }
  
  // 发布链路使用地图更新前过滤后的静态世界点云，保证RViz/PCD可见效果一致
  *pcl_w_wait_pub = *static_world_cloud; // 缓存待发布点云

  if (!img_en) publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();//pubVoxelMap为发布的平面数据
  publish_path(pubPath, LidarMeasures.last_lio_update_time);// 发布路径（使用LIO时间轴）
  publish_mavros(mavros_pose_publisher);// 发送到MAVROS

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;//lio平均处理时间

  // printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  // printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  // printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  // printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  // printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  // printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);//点云降采样耗时
  // printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);//点云匹配优化耗时
  // printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);//体素地图更新耗时
  // printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  // printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  // printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  // printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

}
/*
保存存储的各种状态
*/
void LIVMapper::savePCD() 
{
  // if (pcd_save_en && pcl_wait_save->points.size() > 0 && pcd_save_interval < 0) 
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";

    pcl::PCDWriter pcd_writer;

      if (img_en)
      {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(pcl_wait_save);
        voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
        voxel_filter.filter(*downsampled_cloud);
        pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); // Save the raw point cloud data
        std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                  << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
        pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the downsampled point cloud data
        std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                  << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;
        if(colmap_output_en)
        {
          fout_points << "# 3D point list with one line of data per point\n";
          fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
          for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
          {
              const auto& point = downsampled_cloud->points[i];

              fout_points << i << " "
                          << std::fixed << std::setprecision(6)
                          << point.x << " " << point.y << " " << point.z << " "
                          << static_cast<int>(point.r) << " "
                          << static_cast<int>(point.g) << " "
                          << static_cast<int>(point.b) << " "
                          << 0 << std::endl;
      }
    }
  }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
     }
  }
}


/*
主循环，负责驱动整个SLAM系统的处理流程
*/
void LIVMapper::run() 
{
  ros::Rate rate(5000);// 设置循环频率为5000Hz
  while (ros::ok()) 
  {
    ros::spinOnce();  // 处理回调队列中的消息，即让 ROS 执行 callback
    if (!sync_packages(LidarMeasures)) // 同步LiDAR和IMU数据包
    {
      rate.sleep();// 无数据时休眠
      continue;
    }   
    handleFirstFrame();// 处理首帧初始化

    processImu();// 处理IMU预积分

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();// 执行状态估计与建图
  }
  savePCD();// 退出时保存点云地图
}


void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)//基于IMU测量值进行一次状态传播（姿态、位置、速度）
{
  // 1. 加速度归一化与偏差补偿
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;//mean_acc_norm：加速度计测量值的归一化因子，用于消除重力影响
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;//bias_a 和 bias_g：加速度计和陀螺仪偏置，从状态估计中获取
  angvel_avr -= imu_prop_state.bias_g;
// 2. 姿态更新（旋转矩阵指数映射）
  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;
// 3. 加速度转换到世界坐标系并计算位置和速度
  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

/*
定时器回调，处理IMU数据的高频状态传播与里程计发布
*/
void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();// 加锁保护共享数据
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    // 情况1：有新的EKF状态更新
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;// 从EKF获取最新状态
      // drop all useless imu pkg
      // 丢弃过时的IMU数据
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      // 遍历剩余IMU数据包进行传播
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);// 提取加速度
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);// 提取角速度
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;// 清除更新标志
    }
    else
    // 情况2：无新状态，仅用最新IMU数据单步传播
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);// 提取加速度
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);// 提取角速度
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    // 构造并发布里程计消息
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();// 释放锁
}

/*
坐标系变换函数组
是将点云从LiDAR坐标系转换到世界坐标系
*/
void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}
void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;// 若LiDAR未启用则直接返回
  mtx_buffer.lock();// 加锁保护共享资源
  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
   // 1. 时间戳检查（防止旧数据干扰）
  // if (msg->header.stamp.toSec() < last_timestamp_lidar)
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();// 时间回退时清空缓冲区
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  // 2. 点云预处理
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);// 降采样/去畸变/特征提取（假设）
  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  // 3. 数据入队
  // lidar buffer cap
  const size_t MAX_LIDAR_BUF = 200;
  while (lid_raw_data_buffer.size() >= MAX_LIDAR_BUF)
  {
    lid_raw_data_buffer.pop_front();
    if (!lid_header_time_buffer.empty())
      lid_header_time_buffer.pop_front();
  }

  lid_raw_data_buffer.push_back(ptr);

  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();// 解锁
  sig_buffer.notify_all();// 通知处理线程
}
/*
雷达回调函数  处理Livox LiDAR数据，进行预处理、时间同步检查及数据入队
*/
void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;// LiDAR未启用则直接返回
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));

  // ---- 时间同步检查 ----
  // 检测IMU与LiDAR时间戳差异过大（仅当IMU数据存在时）
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  // 时间回退检查（如传感器重启）
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  // ---- 数据预处理与入队 ----
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);// 去畸变、降采样等

  // lidar buffer cap
  const size_t MAX_LIDAR_BUF = 200;
  while (lid_raw_data_buffer.size() >= MAX_LIDAR_BUF)
  {
    lid_raw_data_buffer.pop_front();
    if (!lid_header_time_buffer.empty())
      lid_header_time_buffer.pop_front();
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();// 唤醒主处理线程
}
/*
处理IMU数据，进行时间偏移修正、同步检查及数据入队
*/
void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;// IMU未启用则直接返回

  if (last_timestamp_lidar < 0.0) return; // 等待LiDAR初始化
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  // ---- 时间偏移修正 ----
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();
  // 检测LiDAR与IMU时间同步状态
  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }
  // 强制时间对齐（若启用）
  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();
// 时间回退检查
  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  last_timestamp_imu = timestamp;

  // imu buffer cap
  const size_t MAX_IMU_BUF = 4000;
  while (imu_buffer.size() >= MAX_IMU_BUF)
  {
    imu_buffer.pop_front();
  }

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  // ---- IMU传播处理 ----
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) 
    { prop_imu_buffer.push_back(*msg); }// 入队传播缓冲区

    newest_imu = *msg;// 更新最新IMU数据
    new_imu = true;// 标记新数据到达
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)//获取图像
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

// static int i = 0;
void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en) return;//有图像
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));

  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  
  std::cout << "[RGB] Received RGB image at timestamp: " << std::fixed << std::setprecision(6) 
            << msg_header_time << std::endl;
            
  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img)
  {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time;

  if (img_time_correct - last_timestamp_img < 0.02)//时间对称
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);

  // 同时将图像添加到原有缓存和新的同步缓存
  // image buffer cap
  const size_t MAX_IMG_BUF = 60;
  while (img_buffer.size() >= MAX_IMG_BUF)
  {
    img_buffer.pop_front();
    if (!img_time_buffer.empty())
      img_time_buffer.pop_front();
  }

  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);
  
  // 添加到帧同步系统
  bool add_result = addRGBFrame(img_cur.clone(), img_time_correct);
  
  std::cout << "[RGB] Added RGB frame to sync buffer: " << (add_result ? "success" : "failed") 
            << ", timestamp: " << img_time_correct << std::endl;

  last_timestamp_img = img_time_correct;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}
void LIVMapper::setic_cbk(const sensor_msgs::ImageConstPtr &msg_in_setic)
{
  if (!setic_en) return; // 只检查是否启用SETIC，不检查副线程状态

  // 取语义图时间戳，语义图时间戳沿用 RGB 原图时间戳，这对同步很重要
  double msg_header_time_setic = msg_in_setic->header.stamp.toSec() + img_time_offset_setic;
  
  std::cout << "[SETIC] Received semantic image at timestamp: " << std::fixed << std::setprecision(6) 
            << msg_header_time_setic << std::endl;

  // 防止时间倒退
  if (msg_header_time_setic <= last_timestamp_img_setic)
  {
    ROS_DEBUG("[SETIC] Timestamp regression detected, skipping frame");
    return;
  }
  
  try {
    // 获取语义图像数据
    cv::Mat img_cur_setic = getImageFromMsg_setic(msg_in_setic);
    if (img_cur_setic.empty()) {
      ROS_WARN("[SETIC] Received empty semantic image");
      return;
    }
    if (setic_use_semantic_if_valid) {
      int nonzero_pixels = 0;
      double max_class_value = 0.0;
      if (!isSemanticImageValid(img_cur_setic, &nonzero_pixels, &max_class_value)) {
        setic_invalid_frame_count_++;
        if (setic_invalid_frame_count_ % 20 == 1) {
          ROS_WARN("[SETIC] Invalid semantic frame detected (nonzero=%d, max=%.1f). Continue FAST-LIVO2 baseline flow.",
                   nonzero_pixels, max_class_value);
        }
        return;
      }
      setic_valid_frame_count_++;
      setic_invalid_frame_count_ = 0;
      last_valid_semantic_time_ = msg_header_time_setic;
      if (setic_fallback_mode_) {
        setic_fallback_mode_ = false;
        ROS_INFO("[SETIC] Valid semantic data recovered, semantic fusion resumed");
      }
    }

    {
      std::lock_guard<std::mutex> lock(semantic_cache_mutex_);
      latest_semantic_mask_id_ = img_cur_setic.clone();
      latest_semantic_mask_time_ = msg_header_time_setic;
    }

    last_timestamp_img_setic = msg_header_time_setic;

  } catch (const std::exception& e) {
    ROS_ERROR("[SETIC] Exception in setic_cbk: %s", e.what());
    std::cout << "[SETIC] Skipping this semantic frame due to error" << std::endl;
    return;
  } catch (...) {
    ROS_ERROR("[SETIC] Unknown exception in setic_cbk");
    std::cout << "[SETIC] Skipping this semantic frame due to unknown error" << std::endl;
    return;
  }
}

void LIVMapper::static_semantic_cbk(const sensor_msgs::ImageConstPtr& msg)
{
  if (!semantic_loop_enable_) return;

  try
  {
    cv::Mat static_mask;
    if (msg->encoding == "mono8" || msg->encoding == "8UC1")
    {
      static_mask = cv_bridge::toCvCopy(msg, "mono8")->image;
    }
    else
    {
      static_mask = cv_bridge::toCvCopy(msg, "mono8")->image;
    }

    if (static_mask.empty())
    {
      return;
    }

    const double msg_time = msg->header.stamp.toSec();
    const int nonzero = cv::countNonZero(static_mask);

    {
      std::lock_guard<std::mutex> lock(static_semantic_mutex_);
      latest_static_semantic_mask_ = static_mask.clone();
      latest_static_semantic_time_ = msg_time;
    }

    std::cout << "[STATIC-SEM] received mask, nonzero=" << nonzero
              << ", t=" << std::fixed << std::setprecision(6) << msg_time << std::endl;
  }
  catch (const std::exception& e)
  {
    ROS_WARN("[STATIC-SEM] callback exception: %s", e.what());
  }
}

bool LIVMapper::publishSemanticGraphKeyframe(const SemanticKeyframeData& keyframe)
{
  if (!pubSemanticPgKeyframe || keyframe.id < 0) return false;

  fast_livo::Keyframe kf_msg;
  kf_msg.header.stamp = ros::Time(keyframe.timestamp);
  kf_msg.header.frame_id = semantic_pg_frame_id_;
  kf_msg.id = static_cast<uint32_t>(keyframe.id);
  kf_msg.time = keyframe.timestamp;
  kf_msg.position.x = keyframe.position_x;
  kf_msg.position.y = keyframe.position_y;
  kf_msg.position.z = keyframe.position_z;
  kf_msg.orientation.x = keyframe.quat_x;
  kf_msg.orientation.y = keyframe.quat_y;
  kf_msg.orientation.z = keyframe.quat_z;
  kf_msg.orientation.w = keyframe.quat_w;
  kf_msg.travel = keyframe.travel_distance;
  kf_msg.source = "online_livmapper_semantic_keyframe";
  pubSemanticPgKeyframe.publish(kf_msg);
  return true;
}

bool LIVMapper::getLatestStaticSemanticMask(double query_time, cv::Mat& static_mask) const
{
  std::lock_guard<std::mutex> lock(static_semantic_mutex_);
  if (latest_static_semantic_mask_.empty()) return false;
  if (latest_static_semantic_time_ < 0.0) return false;
  if (query_time >= 0.0 && std::abs(query_time - latest_static_semantic_time_) > lio_semantic_sync_threshold_) return false;

  static_mask = latest_static_semantic_mask_.clone();
  if (static_mask.type() != CV_8UC1) return false;

  if (vio_manager && vio_manager->cam && vio_manager->width > 0 && vio_manager->height > 0 &&
      (static_mask.cols != vio_manager->width || static_mask.rows != vio_manager->height))
  {
    cv::resize(static_mask, static_mask, cv::Size(vio_manager->width, vio_manager->height), 0, 0, cv::INTER_NEAREST);
  }
  return true;
}

std::vector<SemanticTopologyNode> LIVMapper::buildSemanticTopology(
    const cv::Mat& static_mask,
    const PointCloudXYZI::Ptr& static_world_cloud,
    const StatesGroup& state) const
{
  std::vector<SemanticTopologyNode> nodes;
  if (static_mask.empty() || static_mask.type() != CV_8UC1 ||
      !static_world_cloud || static_world_cloud->empty() ||
      !vio_manager || !vio_manager->cam)
  {
    return nodes;
  }

  struct ClassPoints
  {
    int class_id = -1;
    int pixel_count = 0;
    std::vector<V3D> points_local;
    std::vector<double> depths;
  };

  cv::Mat class_map(static_mask.size(), CV_32SC1, cv::Scalar(-1));
  std::vector<ClassPoints> class_points;
  const int static_class_ids[] = {10, 11, 12, 13, 14, 15, 16};
  for (int class_id : static_class_ids)
  {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const cv::Mat binary = (static_mask == class_id);
    const int count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    const int class_index = static_cast<int>(class_points.size());
    ClassPoints points;
    points.class_id = class_id;
    std::vector<uint8_t> valid_labels(count, 0);
    for (int label = 1; label < count; ++label)
    {
      const int pixels = stats.at<int>(label, cv::CC_STAT_AREA);
      if (pixels < std::max(1, semantic_topology_min_component_pixels_))
      {
        continue;
      }
      valid_labels[label] = 1;
      points.pixel_count += pixels;
    }

    for (int y = 0; y < labels.rows; ++y)
    {
      const int* src = labels.ptr<int>(y);
      int* dst = class_map.ptr<int>(y);
      for (int x = 0; x < labels.cols; ++x)
      {
        const int label = src[x];
        if (label > 0 && valid_labels[label])
        {
          dst[x] = class_index;
        }
      }
    }
    if (points.pixel_count > 0)
    {
      class_points.push_back(std::move(points));
    }
  }

  const M3D Rwi(state.rot_end);
  const V3D Pwi(state.pos_end);
  const M3D Rcw = vio_manager->Rci * Rwi.transpose();
  const V3D Pcw = -vio_manager->Rci * Rwi.transpose() * Pwi + vio_manager->Pci;
  const M3D Rbw = Rwi.transpose();
  for (const auto& point : static_world_cloud->points)
  {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
    {
      continue;
    }
    const V3D point_w(point.x, point.y, point.z);
    const V3D point_c = Rcw * point_w + Pcw;
    if (point_c.z() <= 0.0)
    {
      continue;
    }
    const V2D pixel = vio_manager->cam->world2cam(point_c);
    const int u = static_cast<int>(std::lround(pixel.x()));
    const int v = static_cast<int>(std::lround(pixel.y()));
    if (u < 0 || v < 0 || u >= class_map.cols || v >= class_map.rows)
    {
      continue;
    }
    const int class_index = class_map.at<int>(v, u);
    if (class_index < 0 || class_index >= static_cast<int>(class_points.size()))
    {
      continue;
    }
    class_points[class_index].points_local.push_back(Rbw * (point_w - Pwi));
    class_points[class_index].depths.push_back(point_c.z());
  }

  for (auto& points : class_points)
  {
    const int min_component_points = std::max(1, semantic_topology_min_component_points_);
    if (static_cast<int>(points.points_local.size()) < min_component_points)
    {
      continue;
    }

    struct VoxelKey
    {
      int x;
      int y;
      int z;
      bool operator==(const VoxelKey& other) const
      {
        return x == other.x && y == other.y && z == other.z;
      }
    };
    struct VoxelKeyHash
    {
      size_t operator()(const VoxelKey& key) const
      {
        size_t seed = std::hash<int>()(key.x);
        seed ^= std::hash<int>()(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>()(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
      }
    };

    const double tolerance = std::max(0.05, semantic_topology_cluster_tolerance_);
    const double tolerance_sq = tolerance * tolerance;
    const auto voxelKey = [tolerance](const V3D& point) {
      return VoxelKey{
          static_cast<int>(std::floor(point.x() / tolerance)),
          static_cast<int>(std::floor(point.y() / tolerance)),
          static_cast<int>(std::floor(point.z() / tolerance))};
    };

    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> voxel_points;
    voxel_points.reserve(points.points_local.size());
    for (size_t i = 0; i < points.points_local.size(); ++i)
    {
      voxel_points[voxelKey(points.points_local[i])].push_back(static_cast<int>(i));
    }

    struct Cluster
    {
      std::vector<int> point_indices;
      double median_depth = 0.0;
      V3D min_point = V3D::Zero();
      V3D max_point = V3D::Zero();
    };

    std::vector<uint8_t> visited(points.points_local.size(), 0);
    std::vector<Cluster> clusters;
    for (size_t seed = 0; seed < points.points_local.size(); ++seed)
    {
      if (visited[seed]) continue;
      visited[seed] = 1;
      std::queue<int> pending;
      pending.push(static_cast<int>(seed));
      std::vector<int> cluster;

      while (!pending.empty())
      {
        const int current = pending.front();
        pending.pop();
        cluster.push_back(current);
        const VoxelKey center = voxelKey(points.points_local[current]);
        for (int dx = -1; dx <= 1; ++dx)
        {
          for (int dy = -1; dy <= 1; ++dy)
          {
            for (int dz = -1; dz <= 1; ++dz)
            {
              const auto bucket = voxel_points.find({center.x + dx, center.y + dy, center.z + dz});
              if (bucket == voxel_points.end()) continue;
              for (int neighbor : bucket->second)
              {
                if (visited[neighbor]) continue;
                if ((points.points_local[current] - points.points_local[neighbor]).squaredNorm() >
                    tolerance_sq)
                {
                  continue;
                }
                visited[neighbor] = 1;
                pending.push(neighbor);
              }
            }
          }
        }
      }

      if (static_cast<int>(cluster.size()) >= min_component_points)
      {
        std::vector<double> depths;
        depths.reserve(cluster.size());
        V3D min_point = V3D::Constant(std::numeric_limits<double>::max());
        V3D max_point = V3D::Constant(std::numeric_limits<double>::lowest());
        for (int index : cluster)
        {
          depths.push_back(points.depths[index]);
          min_point = min_point.cwiseMin(points.points_local[index]);
          max_point = max_point.cwiseMax(points.points_local[index]);
        }
        const size_t middle = depths.size() / 2;
        std::nth_element(depths.begin(), depths.begin() + middle, depths.end());
        clusters.push_back({std::move(cluster), depths[middle], min_point, max_point});
      }
    }

    const auto updateCluster = [&points](Cluster& cluster) {
      std::vector<double> depths;
      depths.reserve(cluster.point_indices.size());
      cluster.min_point = V3D::Constant(std::numeric_limits<double>::max());
      cluster.max_point = V3D::Constant(std::numeric_limits<double>::lowest());
      for (int index : cluster.point_indices)
      {
        depths.push_back(points.depths[index]);
        cluster.min_point = cluster.min_point.cwiseMin(points.points_local[index]);
        cluster.max_point = cluster.max_point.cwiseMax(points.points_local[index]);
      }
      const size_t middle = depths.size() / 2;
      std::nth_element(depths.begin(), depths.begin() + middle, depths.end());
      cluster.median_depth = depths[middle];
    };

    // Continuous structures may be split by a foreground dynamic mask. Merge only
    // stable classes; movable repeated objects such as chairs remain independent.
    const bool allow_fragment_merge =
        points.class_id == 10 || points.class_id == 12 ||
        points.class_id == 14 || points.class_id == 16;
    const double merge_distance = std::max(0.0, semantic_topology_fragment_merge_distance_);
    const double max_depth_gap = std::max(0.75, 1.5 * merge_distance);
    if (allow_fragment_merge && merge_distance > 0.0)
    {
      bool merged = true;
      while (merged)
      {
        merged = false;
        for (size_t i = 0; i < clusters.size() && !merged; ++i)
        {
          for (size_t j = i + 1; j < clusters.size(); ++j)
          {
            const V3D gap =
                (clusters[i].min_point - clusters[j].max_point)
                    .cwiseMax(clusters[j].min_point - clusters[i].max_point)
                    .cwiseMax(V3D::Zero());
            if (gap.norm() > merge_distance ||
                std::abs(clusters[i].median_depth - clusters[j].median_depth) > max_depth_gap)
            {
              continue;
            }
            clusters[i].point_indices.insert(clusters[i].point_indices.end(),
                                               clusters[j].point_indices.begin(),
                                               clusters[j].point_indices.end());
            updateCluster(clusters[i]);
            clusters.erase(clusters.begin() + static_cast<long>(j));
            merged = true;
            break;
          }
        }
      }
    }

    std::sort(clusters.begin(), clusters.end(), [](const Cluster& lhs, const Cluster& rhs) {
      if (lhs.median_depth != rhs.median_depth) return lhs.median_depth < rhs.median_depth;
      return lhs.point_indices.size() > rhs.point_indices.size();
    });
    const size_t max_clusters =
        static_cast<size_t>(std::max(1, semantic_topology_max_clusters_per_class_));
    if (clusters.size() > max_clusters) clusters.resize(max_clusters);

    for (const auto& cluster : clusters)
    {
      V3D sum = V3D::Zero();
      V3D min_point = V3D::Constant(std::numeric_limits<double>::max());
      V3D max_point = V3D::Constant(std::numeric_limits<double>::lowest());
      for (int index : cluster.point_indices)
      {
        const V3D& point = points.points_local[index];
        sum += point;
        min_point = min_point.cwiseMin(point);
        max_point = max_point.cwiseMax(point);
      }

      SemanticTopologyNode node;
      node.class_id = points.class_id;
      const V3D centroid = sum / static_cast<double>(cluster.point_indices.size());
      const V3D size = max_point - min_point;
      node.centroid_x = centroid.x();
      node.centroid_y = centroid.y();
      node.centroid_z = centroid.z();
      node.size_x = size.x();
      node.size_y = size.y();
      node.size_z = size.z();
      node.pixel_count = points.pixel_count;
      node.point_count = static_cast<int>(cluster.point_indices.size());
      nodes.push_back(node);
    }
  }
  return nodes;
}

bool LIVMapper::getNearestRgbFrame(double query_time, cv::Mat& rgb_image, double& time_diff)
{
  std::lock_guard<std::mutex> lock(rgb_buffer_mutex_);
  if (rgb_frame_buffer_.empty())
  {
    return false;
  }
  size_t best_idx = 0;
  double best_dt = std::abs(rgb_frame_buffer_[0].second - query_time);
  for (size_t i = 1; i < rgb_frame_buffer_.size(); ++i)
  {
    const double dt = std::abs(rgb_frame_buffer_[i].second - query_time);
    if (dt < best_dt)
    {
      best_dt = dt;
      best_idx = i;
    }
  }
  rgb_image = rgb_frame_buffer_[best_idx].first.clone();
  time_diff = best_dt;
  return !rgb_image.empty();
}

void LIVMapper::tryCreateSemanticKeyframe(const PointCloudXYZI::Ptr& static_world_cloud,
                                          double timestamp,
                                          const StatesGroup& state)
{
  if (!semantic_loop_enable_ || !semantic_loop_manager_) return;
  if (!static_world_cloud) return;

  cv::Mat static_mask;
  if (!getLatestStaticSemanticMask(timestamp, static_mask))
  {
    return;
  }

  const int static_pixels = cv::countNonZero(static_mask);
  const int cloud_points = static_cast<int>(static_world_cloud->size());
  if (static_pixels < min_static_pixels_) return;
  if (cloud_points < min_static_points_) return;

  if (has_last_semantic_keyframe_pose_)
  {
    const double dt = timestamp - last_semantic_keyframe_time_;
    if (dt < semantic_frame_min_interval_) return;

    const double translation = (state.pos_end - last_semantic_keyframe_pos_).norm();
    const M3D dR = last_semantic_keyframe_rot_.transpose() * state.rot_end;
    double cos_theta = (dR.trace() - 1.0) * 0.5;
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    constexpr double kRadToDeg = 57.29577951308232;
    const double rotation_deg = std::acos(cos_theta) * kRadToDeg;
    if (translation < semantic_frame_min_translation_ && rotation_deg < semantic_frame_min_rotation_deg_)
    {
      return;
    }
  }

  double segment_length = 0.0;
  if (has_last_semantic_keyframe_pose_)
  {
    segment_length = (state.pos_end - last_semantic_keyframe_pos_).norm();
  }

  SemanticKeyframeData keyframe;
  keyframe.id = semantic_keyframe_next_id_++;
  keyframe.timestamp = timestamp;
  keyframe.travel_distance = semantic_cumulative_path_length_ + segment_length;
  keyframe.position_x = state.pos_end[0];
  keyframe.position_y = state.pos_end[1];
  keyframe.position_z = state.pos_end[2];
  Eigen::Quaterniond q_world_body(state.rot_end);
  q_world_body.normalize();
  keyframe.quat_x = q_world_body.x();
  keyframe.quat_y = q_world_body.y();
  keyframe.quat_z = q_world_body.z();
  keyframe.quat_w = q_world_body.w();
  keyframe.static_mask = static_mask.clone();
  keyframe.static_cloud.reset(new PointCloudXYZI(*static_world_cloud));
  keyframe.static_cloud_local.reset(new PointCloudXYZI());
  keyframe.static_cloud_local->reserve(static_world_cloud->size());
  const M3D R_w_b = state.rot_end;
  const M3D R_b_w = R_w_b.transpose();
  for (const auto &pt_w : static_world_cloud->points)
  {
    if (!std::isfinite(pt_w.x) || !std::isfinite(pt_w.y) || !std::isfinite(pt_w.z))
    {
      continue;
    }
    const V3D pw(pt_w.x, pt_w.y, pt_w.z);
    const V3D pb = R_b_w * (pw - state.pos_end);
    PointType pt_local = pt_w;
    pt_local.x = pb.x();
    pt_local.y = pb.y();
    pt_local.z = pb.z();
    keyframe.static_cloud_local->push_back(pt_local);
  }
  cv::Mat nearest_rgb;
  double nearest_rgb_dt = 1e9;
  if (getNearestRgbFrame(timestamp, nearest_rgb, nearest_rgb_dt))
  {
    if (nearest_rgb_dt <= lio_semantic_sync_threshold_)
    {
      keyframe.rgb_image = nearest_rgb;
    }
    else if (semantic_verbose_debug_log_)
    {
      std::cout << "[SemanticLoop] skip rgb save for keyframe id=" << keyframe.id
                << ", dt=" << nearest_rgb_dt << "s > sync threshold "
                << lio_semantic_sync_threshold_ << "s" << std::endl;
    }
  }
  keyframe.static_pixels = static_pixels;
  keyframe.cloud_points = cloud_points;
  keyframe.topology_nodes = buildSemanticTopology(static_mask, static_world_cloud, state);

  if (!publishSemanticGraphKeyframe(keyframe))
  {
    if (semantic_verbose_debug_log_)
    {
      std::cout << "[SemanticLoop] skip semantic keyframe id=" << keyframe.id
                << ": failed to publish semantic graph keyframe" << std::endl;
    }
    return;
  }

  semantic_loop_manager_->enqueueKeyframe(keyframe);

  semantic_cumulative_path_length_ = keyframe.travel_distance;
  last_semantic_keyframe_time_ = timestamp;
  last_semantic_keyframe_pos_ = state.pos_end;
  last_semantic_keyframe_rot_ = state.rot_end;
  has_last_semantic_keyframe_pose_ = true;

  if (semantic_verbose_debug_log_)
  {
    std::cout << "[SemanticLoop] create keyframe id=" << keyframe.id
              << ", static_pixels=" << static_pixels
              << ", cloud_points=" << cloud_points
              << ", topology_nodes=" << keyframe.topology_nodes.size()
              << ", travel=" << keyframe.travel_distance << "m" << std::endl;
  }
}

cv::Mat LIVMapper::getImageFromMsg_setic(const sensor_msgs::ImageConstPtr &img_msg_setic)//获取语义图像
{
    std::cout << "[SETIC] getImageFromMsg_setic started" << std::endl;
    
    // 检查输入消息是否有效
    if (!img_msg_setic) {
        std::cout << "[SETIC] ERROR: Null image message pointer" << std::endl;
        throw std::runtime_error("Received null image message pointer");
    }
    
    std::cout << "[SETIC] Image message info - encoding: " << img_msg_setic->encoding 
              << ", width: " << img_msg_setic->width 
              << ", height: " << img_msg_setic->height 
              << ", step: " << img_msg_setic->step 
              << ", data size: " << img_msg_setic->data.size() << std::endl;
    
    auto semanticIdFromLegacyColor = [](const cv::Vec3b &color) -> uint8_t {
        if (color == cv::Vec3b(240, 230, 140)) return 1; // person
        if (color == cv::Vec3b(255, 215, 0)) return 2;   // car
        if (color == cv::Vec3b(34, 139, 34)) return 3;   // bicycle
        if (color == cv::Vec3b(130, 0, 75)) return 4;    // motorcycle
        if (color == cv::Vec3b(0, 69, 255)) return 5;    // bus
        return 0;                                         // background/others
    };

    cv::Mat img_setic;
    
    try {
        // 统一接收单通道类别ID图：0背景，1~5动态类
        if (img_msg_setic->encoding == "mono8" || img_msg_setic->encoding == "8UC1") {
            img_setic = cv_bridge::toCvCopy(img_msg_setic, "mono8")->image;
        }
        // 兼容旧彩色语义输入：按已知BGR颜色映射到类别ID
        else if (img_msg_setic->encoding == "bgr8" || img_msg_setic->encoding == "rgb8")
        {
            cv::Mat semantic_bgr;
            if (img_msg_setic->encoding == "bgr8") {
                semantic_bgr = cv_bridge::toCvCopy(img_msg_setic, "bgr8")->image;
            } else {
                cv::Mat semantic_rgb = cv_bridge::toCvCopy(img_msg_setic, "rgb8")->image;
                cv::cvtColor(semantic_rgb, semantic_bgr, cv::COLOR_RGB2BGR);
            }

            img_setic = cv::Mat::zeros(semantic_bgr.rows, semantic_bgr.cols, CV_8UC1);
            for (int y = 0; y < semantic_bgr.rows; ++y) {
                const cv::Vec3b *src_ptr = semantic_bgr.ptr<cv::Vec3b>(y);
                uint8_t *dst_ptr = img_setic.ptr<uint8_t>(y);
                for (int x = 0; x < semantic_bgr.cols; ++x) {
                    dst_ptr[x] = semanticIdFromLegacyColor(src_ptr[x]);
                }
            }
        }
        // 其他格式尝试转换到 mono8
        else {
            std::cout << "[SETIC] WARNING: Unexpected encoding " << img_msg_setic->encoding
                      << ", attempting conversion to mono8 class-id mask" << std::endl;
            img_setic = cv_bridge::toCvCopy(img_msg_setic, "mono8")->image;
        }
        
        std::cout << "[SETIC] Image conversion successful, size: " << img_setic.rows 
                  << "x" << img_setic.cols << ", channels: " << img_setic.channels() << std::endl;
    }
    catch (const cv_bridge::Exception& e) {
        std::cout << "[SETIC] ERROR: cv_bridge exception: " << e.what() << std::endl;
        throw std::runtime_error("cv_bridge conversion failed: " + std::string(e.what()));
    }
    catch (const cv::Exception& e) {
        std::cout << "[SETIC] ERROR: OpenCV exception: " << e.what() << std::endl;
        throw std::runtime_error("OpenCV conversion failed: " + std::string(e.what()));
    }
    catch (const std::exception& e) {
        std::cout << "[SETIC] ERROR: Standard exception: " << e.what() << std::endl;
        throw std::runtime_error("Image conversion failed: " + std::string(e.what()));
    }
    
    if (img_setic.empty()) {
        std::cout << "[SETIC] ERROR: Converted image is empty" << std::endl;
        throw std::runtime_error("Converted image is empty - check input image format");
    }
    
    // 检查图像尺寸是否合理
    if (img_setic.rows <= 0 || img_setic.cols <= 0) {
        std::cout << "[SETIC] ERROR: Invalid image dimensions: " << img_setic.rows 
                  << "x" << img_setic.cols << std::endl;
        throw std::runtime_error("Invalid image dimensions");
    }
    
    std::cout << "[SETIC] getImageFromMsg_setic completed successfully" << std::endl;
    return img_setic;
}



/*
LIVMapper类中用于多传感器（LiDAR、IMU、相机）数据同步的核心函数
其目标是根据SLAM模式（ONLY_LIO或LIVO）将不同传感器的数据按时间对齐，确保后续算法处理时数据一致性
*/
bool LIVMapper::sync_packages(LidarMeasureGroup &meas)//在run中引用
{
  // 检查各传感器缓冲区是否为空（若对应传感器使能）
  if (lid_raw_data_buffer.empty() && lidar_en) return false;//雷达
  if (img_buffer.empty() && img_en) return false;//图像
  if (imu_buffer.empty() && imu_en) return false;//imu

  switch (slam_mode_)//根据SLAM模式（ONLY_LIO或LIVO）组织数据包，确保时间戳对齐，为后续的状态估计与建图提供输入 只有两种模式
  {
    /*
    两个模式：
    ONLY_LIO模式：仅同步LiDAR和IMU数据。
    LIVO模式：同步LiDAR、IMU和相机数据，支持多模态融合（里面加了相机）
    */
  case ONLY_LIO://纯LiDAR-IMU模式
  {
    /*
    同步一帧LiDAR数据及其对应的IMU数据
    时间计算：假设LiDAR点云中curvature字段存储相对于帧头的时间偏移（单位ms），需确认数据格式与实际硬件一致。
    IMU覆盖检查：若IMU最新时间戳小于LiDAR结束时间，返回false等待更多IMU数据，等待 IMU 时间覆盖到 LiDAR 结束时间
    线程安全：通过mtx_buffer保护共享缓冲区，但需确保所有访问缓冲区的代码均正确加锁
    */
    // 初始化LiDAR处理时间
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)// 首次处理LiDAR帧
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic// 取出最早的LiDAR数据

      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();    // LiDAR起始时间                                            // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time 计算LiDAR扫描结束时间
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;              //处理第一帧的标志位                                                                         // flag
    }
    // 等待IMU数据覆盖LiDAR时间段
    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;// IMU数据不足，等待
    }
    // 收集IMU数据
    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())//清空缓冲数据
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    // 弹出已处理的LiDAR数据
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    // 标记为LIO处理并返回
    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
/*
先进行lio补齐 再进行vio补齐
*/

    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO://LiDAR-IMU-相机融合模式
  /*
  LIVO模式通过状态机管理处理流程，交替处理LIO（LiDAR-IMU）和VIO（视觉-IMU）数据
  方式分别是：VIO -》 LIO -》 VIO -》 LIO -》 VIO （这是其中线程）
  1. 等待IMU数据覆盖LiDAR时间段，确保IMU数据足够  
  */
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    //LIVO模式通过状态机管理处理流程，交替处理LIO（LiDAR-IMU）和VIO（视觉-IMU）数据

    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg) // 根据上一轮处理状态，决定这一轮凑 LIO 包还是 VIO 包
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO://第一次vio
      /*相当于if (last_lio_vio_flg == WAIT || last_lio_vio_flg == VIO)
          {
          ...
          }*/
      // 根据图像捕获时间切割LiDAR点云，准备LIO处理
      {
        // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);

        // 计算图像捕获时间（LiDAR起始时间 + 曝光补偿）
        double img_capture_time = img_time_buffer.front() + exposure_time_init;
        /*** has img topic, but img topic timestamp larger than lidar end time,
         * process lidar topic. After LIO update, the meas.lidar_frame_end_time
         * will be refresh. ***/

        if (meas.last_lio_update_time < 0.0)
          meas.last_lio_update_time = lid_header_time_buffer.front();
        // printf("[ Data Cut ] wait \n");
        // printf("[ Data Cut ] last_lio_update_time: %lf \n",
        // meas.last_lio_update_time);

        double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
        double imu_newest_time = imu_buffer.back()->header.stamp.toSec();
        // 检查图像时间是否合理
        if (img_capture_time < meas.last_lio_update_time + 0.00001)
        {
          img_buffer.pop_front(); // 丢弃过期图像
          img_time_buffer.pop_front();
          ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
          return false;
        }

        if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
        {
          // ROS_ERROR("lost first camera frame");
          // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
          // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
          return false;
        }

        // 收集IMU数据到LiDAR结束时间
        struct MeasureGroup m;

        // printf("[ Data Cut ] LIO \n");
        // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
        m.imu.clear();                 // 清除m中imu的数据
        m.lio_time = img_capture_time; // 相机时间戳
        mtx_buffer.lock();             // 暂时锁一下
        while (!imu_buffer.empty())    // imu存在数据
        {
          if (imu_buffer.front()->header.stamp.toSec() > m.lio_time)
            break;

          if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time)
            m.imu.push_back(imu_buffer.front());

          imu_buffer.pop_front();
          // printf("[ Data Cut ] imu time: %lf \n",
          // imu_buffer.front()->header.stamp.toSec());
        }
        mtx_buffer.unlock();
        sig_buffer.notify_all();

        *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
        PointCloudXYZI().swap(*meas.pcl_proc_next);

        int lid_frame_num = lid_raw_data_buffer.size();
        int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
        meas.pcl_proc_cur->reserve(max_size);
        meas.pcl_proc_next->reserve(max_size);
        // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;
        // 分割LiDAR点云到当前帧和下一帧
        while (!lid_raw_data_buffer.empty())
        {
          if (lid_header_time_buffer.front() > img_capture_time)
            break;
          auto pcl(lid_raw_data_buffer.front()->points);
          double frame_header_time(lid_header_time_buffer.front());
          float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

          for (int i = 0; i < pcl.size(); i++)
          {
            auto pt = pcl[i];
            if (pcl[i].curvature < max_offs_time_ms)
            {
              pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
              meas.pcl_proc_cur->points.push_back(pt); // 当前处理帧
            }
            else
            {
              pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
              meas.pcl_proc_next->points.push_back(pt); // 下一帧
            }
          }
          lid_raw_data_buffer.pop_front();
          lid_header_time_buffer.pop_front();
        }

        meas.measures.push_back(m);
        meas.lio_vio_flg = LIO; // 标记为LIO处理在下面的case中调用了
        // meas.last_lio_update_time = m.lio_time;
        // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
        // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
        // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
        // omp_get_wtime() - t0);
        return true;
      }

    
    case LIO://将图像数据与时间对齐的IMU数据打包 
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;//改成vio补齐
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();
      double imu_time = imu_buffer.front()->header.stamp.toSec();

      // 取出图像数据
      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;//统计时间
      m.img = img_buffer.front();//
      m.setic_time = -1.0;

      {
        std::lock_guard<std::mutex> lock(semantic_cache_mutex_);
        if (!latest_semantic_mask_id_.empty() &&
            latest_semantic_mask_time_ >= 0.0 &&
            std::abs(latest_semantic_mask_time_ - img_capture_time) <= lio_semantic_sync_threshold_)
        {
          m.img_setic = latest_semantic_mask_id_.clone();
          m.setic_time = latest_semantic_mask_time_;
        }
      }

      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = imu_buffer.front()->header.stamp.toSec();
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   imu_buffer.front()->header.stamp.toSec());
      // }
      // 弹出已处理图像
      img_buffer.pop_front();
      img_time_buffer.pop_front();//返回第一个元素
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      // 标记为VIO处理
      meas.measures.push_back(m);
      lidar_pushed = false; // 处理vio的第一帧
      //after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }
  case ONLY_LO://单激光
  {
    if (!lidar_pushed) //如果不存在激光数据
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }
  default://如果识别错误
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}


/*
后面全是发布相关函数
将VIO管理器中的RGB图像转换为ROS消息并发布
*/
void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)//发布img——rgb图像
{
  /*
  使用ros::Time::now()，需确保与传感器数据时间同步
  BGR8符合OpenCV默认格式，但ROS中某些节点可能期望RGB8，需确认下游节点兼容性
  out_msg.header.frame_id被注释，若需要坐标系信息需取消注释
  */
  cv::Mat img_rgb = vio_manager->img_cp;//vio_manager获取拷贝的RGB图像img_cp
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;//使用cv_bridge封装图像数据，设置时间戳和BGR8编码格式
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());//通过image_transport::Publisher发布图像消息
}
/*
发布带RGB颜色的世界坐标系点云
融合点云与图像颜色信息，发布彩色点云，并可选保存为PCD文件
*/
void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en)//静态变量pub_num控制发布间隔，累积pub_scan_num次点云后处理
  {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num == pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();//通过pcl_wait_pub临时存储多帧点云数据
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)//遍历点云，将每个世界坐标系点p_w转换到相机坐标系pf和像素坐标pc
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);//使用getInterpolatedPixel进行双线性插值获取颜色，确保像素在图像范围内
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];//颜色
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
          // if (pointRGB.r > 255) pointRGB.r = 255;
          // else if (pointRGB.r < 0) pointRGB.r = 0;
          // if (pointRGB.g > 255) pointRGB.g = 255;
          // else if (pointRGB.g < 0) pointRGB.g = 0;
          // if (pointRGB.b > 255) pointRGB.b = 255;
          // else if (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);//过滤近距离点（blind_rgb_points阈值）
        }
      }
    }
    else
    {
      pub_num++;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (img_en)//根据img_en标志选择发布彩色或原始点云
  {
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);//rgb值
  }
  else 
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); //原始点云数据
  }
  laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";//设置时间戳和camera_init坐标系
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/

  if (pcd_save_en)//点云地图保存模块
  {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;
    // *pcl_wait_save += *laserCloudWorldRGB;
    if (img_en)//如果出现图像
    {
      *pcl_wait_save += *laserCloudWorldRGB;
    }
    else
    {
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
    }
    scan_wait_num++;
    //按间隔保存点云到指定路径，记录位姿信息
    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en)
      {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;//
        if (img_en)
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
          PointCloudXYZRGB().swap(*pcl_wait_save);
        }
        else
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }  
        // pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
        // PointCloudXYZRGB().swap(*pcl_wait_save);
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
                     << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub); 
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}
/*
将有效的点云数据（点到平面关联的点）转换为 ROS 消息，
并通过指定的 ROS 发布器发布，以便在 ROS 中进行可视化或进一步处理
*/
void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)//发布有效点云数据 (publish_effect_world)
{
  int effect_feat_num = ptpl_list.size();//从输入参数ptpl_list（点-平面关联列表）获取有效点数量
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)//创建PointCloudXYZI类型的点云对象，并遍历ptpl_list填充点云数据
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  //将点云转换为ROS消息，设置时间戳和坐标系（camera_init），通过指定发布器pubLaserCloudEffect发布
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}
/*
通用模板函数，用于设置不同ROS消息的位姿字段
*/
template <typename T> void LIVMapper::set_posestamp(T &out)//设置位姿信息的模板函数 (set_posestamp)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);//将内部状态_state.pos_end的位置赋值给目标消息的position
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;//将四元数geoQuat赋值给目标消息的orientation
}
/*
发布处理后的里程计数据和对应的TF变换
*/
void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)//发布里程计
{
  odomAftMapped.header.frame_id = "camera_init";//设置里程计消息odomAftMapped的帧ID和时间戳
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);//调用set_posestamp填充位姿信息

  static tf::TransformBroadcaster br;//使用tf::TransformBroadcaster发布从camera_init到aft_mapped的坐标变换
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );


  pubOdomAftMapped.publish(odomAftMapped);//发布里程计消息到指定发布器pubOdomAftMapped
}
/*
发布MAVROS位姿 (publish_mavros) 向MAVROS系统发布当前机体位姿
*/
void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();//设置消息头的时间戳和帧ID
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);//使用set_posestamp填充位姿，通过mavros_pose_publisher发布
}
/*
发布路径信息 (publish_path) 更新并发布机器人的运动路径
*/
void LIVMapper::publish_path(const ros::Publisher pubPath, double pose_time_sec)
{
  set_posestamp(msg_body_pose.pose);
  if (pose_time_sec > 0.0)
  {
    msg_body_pose.header.stamp = ros::Time(pose_time_sec);
  }
  else
  {
    msg_body_pose.header.stamp = ros::Time::now();
  }
  msg_body_pose.header.frame_id = "camera_init";//更新msg_body_pose的位姿和时间戳
  path.header.stamp = msg_body_pose.header.stamp;
  path.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);//将当前位姿添加到path的轨迹数组中
  pubPath.publish(path);//  发布更新后的路径到指定发布器pubPath
}

bool LIVMapper::isSemanticImageValid(const cv::Mat& semantic_img, int* nonzero_pixels, double* max_class_value) const
{
  if (semantic_img.empty()) return false;
  
  cv::Mat class_mask;
  if (semantic_img.channels() == 3) {
    cv::cvtColor(semantic_img, class_mask, cv::COLOR_BGR2GRAY);
  } else if (semantic_img.channels() == 1) {
    class_mask = semantic_img;
  } else {
    return false;
  }
  
  const int nz = cv::countNonZero(class_mask);
  double min_val = 0.0;
  double max_val = 0.0;
  cv::minMaxLoc(class_mask, &min_val, &max_val);
  
  if (nonzero_pixels) *nonzero_pixels = nz;
  if (max_class_value) *max_class_value = max_val;
  
  return nz >= setic_min_nonzero_pixels && max_val > 0.0;
}

bool LIVMapper::getLatestSemanticMaskForLio(double lio_time, cv::Mat& semantic_mask) const
{
  if (!lio_dynamic_filter_en_) return false;

  std::lock_guard<std::mutex> lock(semantic_cache_mutex_);
  if (latest_semantic_mask_id_.empty()) return false;
  if (latest_semantic_mask_time_ < 0.0) return false;
  if (std::abs(lio_time - latest_semantic_mask_time_) > lio_semantic_sync_threshold_) return false;

  semantic_mask = latest_semantic_mask_id_.clone();

  if (semantic_mask.type() != CV_8UC1) return false;

  if (vio_manager && vio_manager->cam && vio_manager->width > 0 && vio_manager->height > 0 &&
      (semantic_mask.cols != vio_manager->width || semantic_mask.rows != vio_manager->height))
  {
    cv::resize(semantic_mask, semantic_mask, cv::Size(vio_manager->width, vio_manager->height), 0, 0, cv::INTER_NEAREST);
  }
  return true;
}

bool LIVMapper::isDynamicSemanticPointForLio(const V3D& point_w,
                                             const M3D& Rcw,
                                             const V3D& Pcw,
                                             const cv::Mat& semantic_mask) const
{
  if (!vio_manager || !vio_manager->cam) return false;
  if (semantic_mask.empty() || semantic_mask.type() != CV_8UC1) return false;

  V3D pt_c = Rcw * point_w + Pcw;
  if (pt_c[2] <= 0) return false;

  V2D px = vio_manager->cam->world2cam(pt_c);
  const int u = static_cast<int>(px[0]);
  const int v = static_cast<int>(px[1]);
  if (u < 0 || v < 0 || u >= semantic_mask.cols || v >= semantic_mask.rows) return false;

  const uint8_t semantic_id = semantic_mask.at<uint8_t>(v, u);
  return semantic_id == VIOManager::SEM_PERSON ||
         semantic_id == VIOManager::SEM_CAR ||
         semantic_id == VIOManager::SEM_BICYCLE ||
         semantic_id == VIOManager::SEM_MOTORCYCLE ||
         semantic_id == VIOManager::SEM_BUS;
}

void LIVMapper::filterDynamicCloudBeforeLio(const PointCloudXYZI::Ptr &input_cloud,
                                            double lio_time,
                                            const StatesGroup &state,
                                            PointCloudXYZI::Ptr &output_cloud,
                                            int &removed_points) const
{
  removed_points = 0;
  if (!output_cloud)
    output_cloud.reset(new PointCloudXYZI());
  PointCloudXYZI().swap(*output_cloud);

  if (!input_cloud)
  {
    return;
  }

  cv::Mat semantic_mask;
  if (!getLatestSemanticMaskForLio(lio_time, semantic_mask) || !vio_manager || !vio_manager->cam)
  {
    *output_cloud = *input_cloud;
    return;
  }

  const M3D Rwi(state.rot_end);
  const V3D Pwi(state.pos_end);
  const M3D Rcw = vio_manager->Rci * Rwi.transpose();
  const V3D Pcw = -vio_manager->Rci * Rwi.transpose() * Pwi + vio_manager->Pci;

  output_cloud->reserve(input_cloud->size());
  for (const auto &point : input_cloud->points)
  {
    const V3D point_l(point.x, point.y, point.z);
    const V3D point_w = Rwi * (extR * point_l + extT) + Pwi;
    if (isDynamicSemanticPointForLio(point_w, Rcw, Pcw, semantic_mask))
    {
      ++removed_points;
      continue;
    }
    output_cloud->push_back(point);
  }

  if (output_cloud->empty())
  {
    // 预测位姿阶段投影可能不够准，全部被删时回退，避免 LIO 无约束。
    *output_cloud = *input_cloud;
    removed_points = 0;
    return;
  }

  if (input_cloud->size() > 0 && output_cloud->size() < std::max<size_t>(30, input_cloud->size() / 10))
  {
    // LIO 前端需要足够的几何约束；过滤过猛时优先保证系统连续运行。
    *output_cloud = *input_cloud;
    removed_points = 0;
  }
}

void LIVMapper::filterDynamicPointsBeforeMapUpdate(const std::vector<pointWithVar>& input_points,
                                                   double lio_time,
                                                   const StatesGroup& state,
                                                   std::vector<pointWithVar>& output_points,
                                                   int& removed_points) const
{
  removed_points = 0;
  output_points.clear();

  cv::Mat semantic_mask;
  if (!getLatestSemanticMaskForLio(lio_time, semantic_mask) || !vio_manager || !vio_manager->cam)
  {
    output_points = input_points;
    return;
  }

  // 世界坐标点到相机坐标系：p_c = Rcw * p_w + Pcw
  const M3D Rwi(state.rot_end);
  const V3D Pwi(state.pos_end);
  const M3D Rcw = vio_manager->Rci * Rwi.transpose();
  const V3D Pcw = -vio_manager->Rci * Rwi.transpose() * Pwi + vio_manager->Pci;

  output_points.reserve(input_points.size());
  for (const auto& pv : input_points)
  {
    if (isDynamicSemanticPointForLio(pv.point_w, Rcw, Pcw, semantic_mask))
    {
      ++removed_points;
      continue;
    }
    output_points.push_back(pv);
  }

  if (output_points.empty())
  {
    // 全部被删时回退原始流程，避免地图更新完全停摆
    output_points = input_points;
    removed_points = 0;
  }
}

// 发布语义图像
void LIVMapper::publish_img_rgb_setic(const image_transport::Publisher &pubImage_setic, seticmanagerPtr setic_manager)
{
  if (!setic_manager || setic_manager->img_rgb_setic.empty()) {
    return;
  }
  
  try {
    cv::Mat img_setic = setic_manager->img_rgb_setic; // 获取处理后的语义图像
    cv::Mat vis_bgr;

    if (img_setic.type() == CV_8UC1) {
      vis_bgr = cv::Mat::zeros(img_setic.rows, img_setic.cols, CV_8UC3);
      for (int y = 0; y < img_setic.rows; ++y) {
        const uint8_t *id_ptr = img_setic.ptr<uint8_t>(y);
        cv::Vec3b *dst_ptr = vis_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img_setic.cols; ++x) {
          switch (id_ptr[x]) {
            case 1: dst_ptr[x] = cv::Vec3b(240, 230, 140); break; // person
            case 2: dst_ptr[x] = cv::Vec3b(255, 215, 0); break;   // car
            case 3: dst_ptr[x] = cv::Vec3b(34, 139, 34); break;   // bicycle
            case 4: dst_ptr[x] = cv::Vec3b(130, 0, 75); break;    // motorcycle
            case 5: dst_ptr[x] = cv::Vec3b(0, 69, 255); break;    // bus
            default: dst_ptr[x] = cv::Vec3b(0, 0, 0); break;
          }
        }
      }
    } else if (img_setic.type() == CV_8UC3) {
      vis_bgr = img_setic;
    } else {
      ROS_WARN("[SETIC] Unsupported semantic image type for visualization: %d", img_setic.type());
      return;
    }

    cv_bridge::CvImage out_msg;
    out_msg.header.stamp = ros::Time::now();
    out_msg.header.frame_id = "camera_init";
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg.image = vis_bgr;
    
    pubImage_setic.publish(out_msg.toImageMsg());
    
    ROS_DEBUG("[SETIC] Published semantic image with size %dx%d", img_setic.rows, img_setic.cols);
    
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR("[SETIC] cv_bridge exception: %s", e.what());
  } catch (const std::exception& e) {
    ROS_ERROR("[SETIC] Publishing error: %s", e.what());
  }
}

// 添加RGB图像帧到缓存（用于回环检索最近RGB）
bool LIVMapper::addRGBFrame(const cv::Mat& img, double timestamp) 
{
  std::lock_guard<std::mutex> lock(rgb_buffer_mutex_);
  
  // 缓存大小受限，超出时丢弃最旧帧
  while (rgb_frame_buffer_.size() >= max_buffer_size_) {
    rgb_frame_buffer_.pop_front();
  }
  
  rgb_frame_buffer_.emplace_back(img.clone(), timestamp);
  
  ROS_DEBUG("[RGB-BUFFER] Added RGB frame at %.6f, buffer size: %zu", 
            timestamp, rgb_frame_buffer_.size());
  
  return true;
}

/*
发布SETIC体素地图点云 - 将语义图像与点云融合
*/
void LIVMapper::publish_setic_voxel_map(const ros::Publisher &pubSeticVoxelMap, const std::vector<pointWithVar> &setic_pv_list)
{
  if (setic_pv_list.empty()) {
    ROS_DEBUG("[SETIC] Empty point list, skipping voxel map publication");
    return;
  }
  
  // 检查SETIC管理器和语义图像是否可用
  if (!setic_manager || setic_manager->img_rgb_setic.empty()) {
    ROS_DEBUG("[SETIC] No semantic image available, publishing points without semantic info");
    
    // 如果没有语义图像，发布普通点云
    PointCloudXYZI::Ptr setic_voxel_cloud(new PointCloudXYZI());
    setic_voxel_cloud->reserve(setic_pv_list.size());
    
    for (const auto& pv : setic_pv_list) {
      PointType point;
      point.x = pv.point_w[0];
      point.y = pv.point_w[1];
      point.z = pv.point_w[2];
      point.intensity = 1.0;
      setic_voxel_cloud->push_back(point);
    }
    
    sensor_msgs::PointCloud2 setic_voxel_msg;
    pcl::toROSMsg(*setic_voxel_cloud, setic_voxel_msg);
    setic_voxel_msg.header.stamp = ros::Time::now();
    setic_voxel_msg.header.frame_id = "camera_init";
    
    pubSeticVoxelMap.publish(setic_voxel_msg);
    return;
  }
  
  try {
    // 创建带语义信息的彩色点云
    PointCloudXYZRGB::Ptr setic_semantic_cloud(new PointCloudXYZRGB());
    setic_semantic_cloud->reserve(setic_pv_list.size());
    
    cv::Mat img_semantic = setic_manager->img_rgb_setic; // 获取语义图像
    
    // 检查VIO管理器是否可用（用于坐标转换）
    if (!vio_manager || !vio_manager->new_frame_) {
      ROS_WARN("[SETIC] VIO manager not available for coordinate transformation");
      return;
    }
    
    int valid_points = 0;
    int total_points = setic_pv_list.size();
    
    for (const auto& pv : setic_pv_list) {
      // 创建语义点云点
      PointTypeRGB semantic_point;
      semantic_point.x = pv.point_w[0];
      semantic_point.y = pv.point_w[1];
      semantic_point.z = pv.point_w[2];
      
      // 将世界坐标点转换到相机坐标系
      V3D p_w(pv.point_w[0], pv.point_w[1], pv.point_w[2]);
      V3D pf(vio_manager->new_frame_->w2f(p_w)); // 世界到相机坐标
      
      // 检查点是否在相机前方
      if (pf[2] < 0) {
        // 点在相机后方，设置默认颜色
        semantic_point.r = 128;
        semantic_point.g = 128;
        semantic_point.b = 128;
        setic_semantic_cloud->push_back(semantic_point);
        continue;
      }
      
      V2D pc(vio_manager->new_frame_->w2c(p_w)); // 世界到像素坐标
      
      // 检查像素是否在图像范围内
      if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) {
        // 获取语义图像中对应像素的颜色值
        V3F semantic_pixel = vio_manager->getInterpolatedPixel(img_semantic, pc);
        
        // 设置语义颜色
        semantic_point.r = static_cast<uint8_t>(semantic_pixel[2]); // B
        semantic_point.g = static_cast<uint8_t>(semantic_pixel[1]); // G
        semantic_point.b = static_cast<uint8_t>(semantic_pixel[0]); // R
        
        valid_points++;
      } else {
        // 像素超出图像范围，设置默认颜色
        semantic_point.r = 64;
        semantic_point.g = 64;
        semantic_point.b = 64;
      }
      
      // 过滤距离过近的点
      if (pf.norm() > blind_rgb_points) {
        setic_semantic_cloud->push_back(semantic_point);
      }
    }
    
    // 发布语义点云
    if (!setic_semantic_cloud->empty()) {
      sensor_msgs::PointCloud2 setic_semantic_msg;
      pcl::toROSMsg(*setic_semantic_cloud, setic_semantic_msg);
      setic_semantic_msg.header.stamp = ros::Time::now();
      setic_semantic_msg.header.frame_id = "camera_init";
      
      pubSeticVoxelMap.publish(setic_semantic_msg);
      
      ROS_DEBUG("[SETIC] Published semantic voxel map: %zu total points, %d valid projections (%.1f%%)", 
                setic_semantic_cloud->size(), valid_points, 
                total_points > 0 ? (100.0 * valid_points / total_points) : 0.0);
    } else {
      ROS_WARN("[SETIC] No valid semantic points to publish");
    }
    
  } catch (const cv::Exception& e) {
    ROS_ERROR("[SETIC] OpenCV error in semantic voxel map: %s", e.what());
  } catch (const std::exception& e) {
    ROS_ERROR("[SETIC] Error publishing semantic voxel map: %s", e.what());
  }
}
