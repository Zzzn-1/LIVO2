/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "vio.h"
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/String.h>
#include <visualization_msgs/MarkerArray.h>
#include <vikit/camera_loader.h>
#include "setic.h"
#include "semantic_loop_manager.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <queue>
#include <unordered_map>
#include <condition_variable>
#include <memory>

class LIVMapper
{
public:
  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it);
  void initializeComponents();
  void initializeFiles();
  void run();
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  void processImu();
  
  void publish_img_rgb_setic(const image_transport::Publisher &pubImage_setic, seticmanagerPtr setic_manager);
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);
  void setic_cbk(const sensor_msgs::ImageConstPtr &msg_in_setic);
  void static_semantic_cbk(const sensor_msgs::ImageConstPtr& msg);

  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
  cv::Mat getImageFromMsg_setic(const sensor_msgs::ImageConstPtr &img_msg_setic);

  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  template <typename T> void set_posestamp(T &out);
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);
  void publish_path(const ros::Publisher pubPath, double pose_time_sec);
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_setic_voxel_map(const ros::Publisher &pubSeticVoxelMap, const std::vector<pointWithVar> &setic_pv_list);

private:
  void readParameters(ros::NodeHandle &nh);

public:
  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false;
  int pcd_save_interval = -1, pcd_index = 0;
  int pub_scan_num = 1;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;
  bool gravity_align_en = false, gravity_align_finished = false;
  bool sync_jump_flag = false;
  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  bool verbose_console_log_ = true;
  bool keep_loop_event_log_when_quiet_ = true;

  int img_en = 1, imu_int_frame = 3, setic_en = 1;

  bool normal_en = true;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  
  // SETIC异步处理控制变量
  bool setic_processing;
  int setic_skip_count;
  int max_setic_skip;
  int setic_timeout_ms;
  double last_successful_setic_time;
  
  // 语义处理VIO集成控制变量
  bool setic_vio_en;                    // 是否在VIO中启用语义处理
  int setic_process_freq;               // 语义处理频率
  double setic_max_process_time;        // 语义处理最大时间限制(ms)
  bool setic_skip_when_slow;            // 当语义处理过慢时是否跳过
  int vio_frame_count;                  // VIO帧计数器
  bool has_semantic_data;               // 是否有可用的语义数据
  
  double last_timestamp_img_setic;             // 最新语义图像时间戳
  
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  double outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  double img_time_offset_setic = 0.0;

  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  
  deque<cv::Mat> img_buffer;
  deque<cv::Mat> img_buffer_setic;

  deque<double> img_time_buffer;
  deque<double> img_time_buffer_setic;
  vector<pointWithVar> _pv_list;
  std::mutex pv_list_mutex_;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  double IMG_POINT_COV;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pcd_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;

  VIOManagerPtr vio_manager;
  seticmanagerPtr setic_manager;
  
  // 新增：SETIC体素地图管理器和相关变量
  VoxelMapManagerPtr setic_voxelmap_manager;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> setic_voxel_map;
  bool setic_lidar_map_inited = false;

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Publisher semantic_voxel_pub; // 语义体素地图发布器
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_img;
  ros::Subscriber sub_setic;

  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  ros::Publisher pubSemanticLoopEvent;
  ros::Publisher pubSemanticLoopMarkers;
  ros::Publisher pubSemanticPgKeyframe;
  ros::Publisher pubSemanticPgLoopConstraint;
  ros::Publisher pubSeticVoxelMap; // SETIC体素地图点云发布器
  image_transport::Publisher pubImage;
  image_transport::Publisher pubImage_setic;
  ros::Publisher mavros_pose_publisher;
  ros::Timer imu_prop_timer;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;
  
  // 缺失的重要成员变量
  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;
  
  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name, img_topic, setic_topic;
  V3D extT;
  M3D extR;
  
  int feats_down_size = 0, max_iterations = 0;
  
  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;
  
  // 相机内参和语义投影相关
  Eigen::Matrix3d camera_intrinsic_matrix_;
  bool camera_intrinsic_initialized_ = false;
  double fx_, fy_, cx_, cy_;  // 相机内参
  int semantic_projection_count_ = 0;
  double last_projection_time_ = 0.0;

  // 帧缓存管理（用于回环检索最近RGB）
  std::deque<std::pair<cv::Mat, double>> rgb_frame_buffer_;
  std::mutex rgb_buffer_mutex_;
  size_t max_buffer_size_;

  // 帧缓存函数
  bool addRGBFrame(const cv::Mat& img, double timestamp);
  bool isSemanticImageValid(const cv::Mat& semantic_img, int* nonzero_pixels = nullptr, double* max_class_value = nullptr) const;
  bool getLatestSemanticMaskForLio(double lio_time, cv::Mat& semantic_mask) const;
  bool isDynamicSemanticPointForLio(const V3D& point_w, const M3D& Rcw, const V3D& Pcw, const cv::Mat& semantic_mask) const;
  void filterDynamicCloudBeforeLio(const PointCloudXYZI::Ptr &input_cloud,
                                   double lio_time,
                                   const StatesGroup &state,
                                   PointCloudXYZI::Ptr &output_cloud,
                                   int &removed_points) const;
  void filterDynamicPointsBeforeMapUpdate(const std::vector<pointWithVar>& input_points,
                                          double lio_time,
                                          const StatesGroup& state,
                                          std::vector<pointWithVar>& output_points,
                                          int& removed_points) const;
  void publishSemanticLoopMarkers(const SemanticLoopManager::LoopEvent& event);
  bool getLatestStaticSemanticMask(double query_time, cv::Mat& static_mask) const;
  std::vector<SemanticTopologyNode> buildSemanticTopology(
      const cv::Mat& static_mask,
      const PointCloudXYZI::Ptr& static_world_cloud,
      const StatesGroup& state) const;
  bool getNearestRgbFrame(double query_time, cv::Mat& rgb_image, double& time_diff);
  bool publishSemanticGraphKeyframe(const SemanticKeyframeData& keyframe);
  void tryCreateSemanticKeyframe(const PointCloudXYZI::Ptr& static_world_cloud,
                                 double timestamp,
                                 const StatesGroup& state);

  // 🔧 新增：SETIC性能参数
  double setic_max_processing_time;
  double setic_warning_threshold;
  int setic_adaptive_threshold;
  bool setic_adaptive_enable;
  bool setic_monitoring_enable;
  int setic_max_queue_size;
  int setic_viz_skip_interval;
  int setic_patch_skip_interval;
  double setic_image_scale_threshold;
  bool setic_heavy_voxel_update_en_ = false;
  bool setic_semantic_loop_en_ = false;
  
  // 语义可用性门控参数
  bool setic_use_semantic_if_valid;
  int setic_min_nonzero_pixels;
  double setic_semantic_valid_timeout;
  
  // 语义可用性状态
  bool setic_fallback_mode_;
  int setic_invalid_frame_count_;
  int setic_valid_frame_count_;
  double last_valid_semantic_time_;
  
  // LIO动态点入图剔除（第一阶段）
  bool lio_dynamic_filter_en_ = true;
  double lio_semantic_sync_threshold_ = 0.10;
  mutable std::mutex semantic_cache_mutex_;
  cv::Mat latest_semantic_mask_id_;
  double latest_semantic_mask_time_ = -1.0;

  // 静态语义mask缓存（语义回环链路）
  std::string static_semantic_topic_ = "/semantic/static_mask";
  std::string semantic_pg_frame_id_ = "camera_init";
  std::string semantic_pg_keyframe_topic_ = "/semantic_pg/keyframe";
  std::string semantic_pg_loop_topic_ = "/semantic_pg/loop_constraint";
  ros::Subscriber sub_static_semantic;
  mutable std::mutex static_semantic_mutex_;
  cv::Mat latest_static_semantic_mask_;
  double latest_static_semantic_time_ = -1.0;

  // 语义回环关键帧触发参数
  bool semantic_loop_enable_ = false;
  double semantic_frame_min_translation_ = 1.0;
  double semantic_frame_min_rotation_deg_ = 10.0;
  double semantic_frame_min_interval_ = 1.0;
  int min_static_pixels_ = 300;
  int min_static_points_ = 200;
  int semantic_top_k_ = 5;
  double semantic_score_threshold_ = 0.55;
  int semantic_max_queue_size_ = 3;
  double semantic_candidate_min_time_gap_ = 10.0;
  double semantic_candidate_min_path_distance_ = 20.0;
  double semantic_candidate_min_euclidean_distance_ = 0.0;
  double semantic_candidate_max_euclidean_distance_ = 0.0;
  bool semantic_topology_verify_enable_ = true;
  int semantic_topology_verify_top_n_ = 3;
  int semantic_topology_min_component_pixels_ = 80;
  int semantic_topology_min_component_points_ = 8;
  double semantic_topology_cluster_tolerance_ = 0.45;
  double semantic_topology_fragment_merge_distance_ = 0.60;
  int semantic_topology_max_clusters_per_class_ = 3;
  int semantic_topology_min_nodes_ = 3;
  int semantic_topology_min_edges_ = 2;
  double semantic_topology_max_edge_distance_ = 8.0;
  double semantic_topology_edge_tolerance_ = 0.5;
  double semantic_topology_score_threshold_ = 0.55;
  bool semantic_topology_fallback_enable_ = true;
  double semantic_topology_fallback_voxel_size_ = 0.8;
  double semantic_topology_fallback_inlier_ratio_threshold_ = 0.12;
  int semantic_topology_fallback_min_inliers_ = 80;
  bool semantic_icp_verify_enable_ = true;
  int semantic_icp_verify_top_n_ = 3;
  int semantic_icp_min_inliers_ = 80;
  double semantic_icp_downsample_leaf_size_ = 0.5;
  double semantic_icp_max_correspondence_distance_ = 1.5;
  int semantic_icp_max_iterations_ = 30;
  double semantic_icp_fitness_threshold_ = 0.35;
  bool semantic_icp_use_local_cloud_for_relative_pose_ = true;
  double semantic_loop_info_geo_ratio_ref_ = 0.8;
  double semantic_loop_info_inlier_ref_ = 2000.0;
  double semantic_loop_info_non_icp_scale_ = 0.5;
  bool semantic_verbose_debug_log_ = false;
  bool semantic_print_top_candidates_ = true;
  bool semantic_print_icp_detail_ = false;
  bool semantic_save_loop_event_pcd_ = false;
  int semantic_max_geometric_accepts_per_query_ = 1;
  double semantic_accept_cooldown_time_gap_ = 20.0;
  double semantic_accept_cooldown_path_gap_ = 8.0;
  double semantic_global_loop_cooldown_time_gap_ = 8.0;
  double semantic_global_loop_cooldown_path_gap_ = 5.0;
  int semantic_global_loop_cooldown_match_id_window_ = 2;
  double semantic_max_relative_rotation_deg_ = 0.0;

  // 语义关键帧状态
  int semantic_keyframe_next_id_ = 0;
  double semantic_cumulative_path_length_ = 0.0;
  double last_semantic_keyframe_time_ = -1.0;
  V3D last_semantic_keyframe_pos_ = V3D::Zero();
  M3D last_semantic_keyframe_rot_ = M3D::Identity();
  bool has_last_semantic_keyframe_pose_ = false;
  std::unique_ptr<SemanticLoopManager> semantic_loop_manager_;
  std::mutex semantic_loop_marker_mutex_;
  std::vector<geometry_msgs::Point> semantic_loop_marker_line_points_;
  std::vector<geometry_msgs::Point> semantic_loop_marker_node_points_;
  
  // 🔧 新增：帧处理计数器
  int frame_processing_count = 0;

  // 新增：帧同步函数
};
#endif
