/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <ros/ros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define VOXELMAP_HASH_P 116101//哈希计算的一个常量（定义为 116101）
#define VOXELMAP_MAX_N 10000000000//哈希值的最大范围（定义为 10000000000）

static int voxel_plane_id = 0;

typedef struct VoxelMapConfig
{
  double max_voxel_size_;
  int max_layer_;
  int max_iterations_;
  std::vector<int> layer_init_num_;
  int max_points_num_;
  double planner_threshold_;
  double beam_err_;
  double dept_err_;
  double sigma_num_;
  bool is_pub_plane_map_;
  bool pub_semantic_color_en; // 是否发布语义颜色体素地图
  int update_size_threshold_;

  // config of local map sliding
  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;
} VoxelMapConfig;

typedef struct PointToPlane
{
  Eigen::Vector3d point_b_;
  Eigen::Vector3d point_w_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d center_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  M3D body_cov_;
  int layer_;
  double d_;
  double eigen_value_;
  bool is_valid_;
  float dis_to_plane_;
} PointToPlane;

typedef struct VoxelPlane
{
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  float d_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
  bool is_init_ = false;
  int id_ = 0;
  bool is_update_ = false;
  
  // 关键修复：添加语义信息存储
  int semantic_label_ = 0;                    // 主要语义标签
  std::map<int, int> semantic_count_;         // 各语义标签的点数统计
  bool has_semantic_ = false;                 // 是否包含语义信息
  
  // 增强语义信息
  double semantic_confidence_ = 0.0;          // 语义置信度
  int projection_count_ = 0;                  // 被投影到的次数
  double last_semantic_update_time_ = 0.0;    // 最后语义更新时间
  std::map<int, double> semantic_weights_;    // 各语义标签的权重
  bool force_semantic_color_ = false;         // 强制使用语义颜色
  
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
    semantic_label_ = 0;
    has_semantic_ = false;
    semantic_confidence_ = 0.0;
    projection_count_ = 0;
    last_semantic_update_time_ = 0.0;
    force_semantic_color_ = false;
  }
  
  // 添加语义点的统计
  void addSemanticPoint(int semantic_label) {
    if (semantic_label > 0) {
      semantic_count_[semantic_label]++;
      has_semantic_ = true;
      updateDominantSemanticLabel();
    }
  }
  
  // 通过投影添加语义信息（带权重和置信度）- 更宽松的接受条件
  void addSemanticProjection(int semantic_label, double confidence, double current_time) {
    if (semantic_label > 0 && confidence > 0.01) { // 大幅降低置信度阈值，从0.1降到0.01
      semantic_weights_[semantic_label] += confidence;
      projection_count_++;
      last_semantic_update_time_ = current_time;
      has_semantic_ = true;
      updateDominantSemanticLabelWeighted();
      
      // 降低强制语义颜色的置信度要求
      if (confidence > 0.3) { // 从0.7降到0.3
        force_semantic_color_ = true;
        semantic_confidence_ = std::max(semantic_confidence_, confidence);
      }
      
      // 即使低置信度也累积语义信息
      if (confidence > 0.05) { // 新增：中低置信度也更新语义置信度
        semantic_confidence_ = std::max(semantic_confidence_, confidence * 0.8);
      }
    }
  }
  
  // 更新主要语义标签（选择出现次数最多的）
  void updateDominantSemanticLabel() {
    int max_count = 0;
    int dominant_label = 0;
    for (const auto& pair : semantic_count_) {
      if (pair.second > max_count) {
        max_count = pair.second;
        dominant_label = pair.first;
      }
    }
    semantic_label_ = dominant_label;
  }
  
  // 基于权重更新主要语义标签
  void updateDominantSemanticLabelWeighted() {
    double max_weight = 0.0;
    int dominant_label = 0;
    for (const auto& pair : semantic_weights_) {
      if (pair.second > max_weight) {
        max_weight = pair.second;
        dominant_label = pair.first;
      }
    }
    if (max_weight > 0.0) {
      semantic_label_ = dominant_label;
      semantic_confidence_ = max_weight / (projection_count_ + 1);
    }
  }
} VoxelPlane;

class VOXEL_LOCATION//新类
{
public:
  int64_t x, y, z;
  //VOXEL_LOCATION 用于表示体素的三维位置，常用于体素地图的存储和操作
  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}//初始化体素位置
  //通过重载 operator== 和提供哈希函数特化，VOXEL_LOCATION 可以直接用作 std::unordered_map 或 std::unordered_set 的键

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
/*
在 std 命名空间中，为 VOXEL_LOCATION 提供了哈希函数的特化，以支持在 std::unordered_map 等哈希容器中使用。
*/
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} 
// namespace std

struct DS_POINT//用于存储点云数据的基本信息，包括坐标、强度和计数
{
  float xyz[3];
  float intensity;
  int count = 0;
};

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

class VoxelOctoTree
{
public:
  VoxelOctoTree() = default;
  std::vector<pointWithVar> temp_points_;
  VoxelPlane *plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  VoxelOctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_init_num_;
  float quater_length_;
  float planer_threshold_;
  int points_size_threshold_;
  int update_size_threshold_;
  int max_points_num_;
  int max_layer_;
  int new_points_;
  bool init_octo_;
  bool update_enable_;
  V3D last_update_position_;  // 最后更新时的位置
  M3D last_update_rotation_;  // 最后更新时的旋转
  double last_update_time_;   // 最后更新时间
/*
int max_layer:八叉树的最大层数。用于限制八叉树的深度，避免无限递归。
int layer:当前节点的层数。表示当前节点在八叉树中的深度。
int points_size_threshold:点云大小阈值。用于判断当前节点是否需要初始化或分割。
int max_points_num:点允许存储的最大点数，如果点数超过此值，可能会清空点云或禁用更新。
float planer_threshold:平面判断的特征值阈值。用于判断点云是否构成平面。
*/
  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold)
  {
    temp_points_.clear();//临时存储点云数据的容器。在构造函数中清空
    octo_state_ = 0;//0 表示叶子节点或平面节点，1 表示需要继续分割。
    new_points_ = 0;//用于记录插入到当前节点的新点数量
    update_size_threshold_ = 5;//当新点数量超过此值时，可能会重新初始化平面。
    init_octo_ = false;//false 表示节点未初始化，true 表示节点已初始化
    update_enable_ = true;//true 表示允许更新，false 表示禁止更新
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;//初始化为 nullptr，表示当前节点没有子节点。
    }
    plane_ptr_ = new VoxelPlane;//在构造函数中分配内存，初始化为一个新的 VoxelPlane 对象
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);
  void init_octo_tree();
  void cut_octo_tree();
  void UpdateOctoTree(const pointWithVar &pv);
  void addPoint(const cv::Vec3f &point, int semantic_label);

  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);//记录中心点在哪
  VoxelOctoTree *Insert(const pointWithVar &pv);//添加内点
  
  // 关键修复：添加语义信息处理功能
  bool has_semantic_info() const {
    return plane_ptr_ && plane_ptr_->has_semantic_;
  }
  
  int getDominantSemanticLabel() const {
    if (plane_ptr_ && plane_ptr_->has_semantic_) {
      return plane_ptr_->semantic_label_;
    }
    return 0;
  }
  
  void addSemanticLabel(int semantic_label) {
    if (plane_ptr_ && semantic_label > 0) {
      plane_ptr_->addSemanticPoint(semantic_label);
    }
  }
};

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config);

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelMapConfig config_setting_;
  int current_frame_id_ = 0;
  ros::Publisher voxel_map_pub_;
  ros::Publisher semantic_voxel_map_pub_; // 语义体素地图发布器
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;//用于存储每个体素的位置和对应的八叉树节点

  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;

  M3D extR_;
  V3D extT_;
  float build_residual_time, ekf_time;
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;
  int scan_count = 0;
  StatesGroup state_;
  V3D position_last_;

  V3D last_slide_position = {0,0,0};

  geometry_msgs::Quaternion geoQuat_;

  int feats_down_size_;
  int effct_feat_num_;
  std::vector<M3D> cross_mat_list_;
  std::vector<M3D> body_cov_list_;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;

  int points_size_threshold_;

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
  {
    current_frame_id_ = 0;
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  };

  void StateEstimation(StatesGroup &state_propagat);
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

  void BuildVoxelMap();
  V3F RGBFromVoxel(const V3D &input_point);

  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);

  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);

  void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob,
                             PointToPlane &single_ptpl);

  void pubVoxelMap();

  void mapSliding();
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );

  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);
  void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane,
                      const float alpha, const Eigen::Vector3d rgb);
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                       geometry_msgs::Quaternion &q);
  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
  
  // 关键修复：添加语义颜色相关函数
  V3F getSemanticColor(int semantic_label);
  void updateVoxelSemantic(const std::vector<pointWithVar> &input_points);
  void pubSemanticVoxelMap();
  Eigen::Vector3d getSemanticColorForVoxel(const VoxelPlane& plane);
  void initSemanticColorMap();
  
  // 基于视角的语义映射函数
  void projectSemanticToVoxels(const cv::Mat& semantic_img, const Eigen::Matrix3d& rot, const Eigen::Vector3d& pos, 
                               const Eigen::Matrix3d& camera_intrinsic);
  void updateVoxelSemanticFromProjection(const cv::Mat& semantic_img, const Eigen::Matrix3d& rot, 
                                        const Eigen::Vector3d& pos, const Eigen::Matrix3d& camera_intrinsic);
  bool projectVoxelToImage(const Eigen::Vector3d& voxel_center, const Eigen::Matrix3d& rot, 
                          const Eigen::Vector3d& pos, const Eigen::Matrix3d& camera_intrinsic, 
                          cv::Point2f& image_point);
  void enhanceSemanticVisualization();
  void pubEnhancedSemanticVoxelMap();
  
private:
  // 语义颜色映射表
  std::map<int, Eigen::Vector3d> semantic_color_map_;
  bool semantic_initialized_ = false;
  
  // 视角相关的语义映射
  Eigen::Matrix3d current_camera_intrinsic_;
  bool camera_intrinsic_set_ = false;
  double semantic_confidence_threshold_ = 0.7;
  int recent_semantic_frames_ = 0;
  double last_semantic_projection_time_ = 0.0;
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_