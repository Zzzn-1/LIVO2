#include "/home/liu/code/LIVO2/ws_livo2/src/FAST-LIVO2/include/semantic_loop_manager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Geometry>

namespace
{
constexpr int kGridRows = 4;
constexpr int kGridCols = 4;
constexpr std::array<uint8_t, 7> kStaticClassIds = {10, 11, 12, 13, 14, 15, 16};
constexpr int kNeighborRange = 1;
constexpr std::size_t kMinIcpPoints = 50;

using IcpPoint = pcl::PointXYZ;
using IcpCloud = pcl::PointCloud<IcpPoint>;

struct VoxelKey
{
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const VoxelKey &other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey &key) const
  {
    std::size_t h1 = std::hash<int>()(key.x);
    std::size_t h2 = std::hash<int>()(key.y);
    std::size_t h3 = std::hash<int>()(key.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

struct TopologyEdge
{
  int class_pair = 0;
  double distance = 0.0;
};

std::vector<TopologyEdge> buildTopologyEdges(const std::vector<SemanticTopologyNode> &nodes,
                                             double max_edge_distance)
{
  std::vector<TopologyEdge> edges;
  for (size_t i = 0; i < nodes.size(); ++i)
  {
    for (size_t j = i + 1; j < nodes.size(); ++j)
    {
      const double dx = nodes[i].centroid_x - nodes[j].centroid_x;
      const double dy = nodes[i].centroid_y - nodes[j].centroid_y;
      const double dz = nodes[i].centroid_z - nodes[j].centroid_z;
      const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (!std::isfinite(distance) || distance <= 0.0 || distance > max_edge_distance)
      {
        continue;
      }
      const int class_a = std::min(nodes[i].class_id, nodes[j].class_id);
      const int class_b = std::max(nodes[i].class_id, nodes[j].class_id);
      edges.push_back(TopologyEdge{class_a * 256 + class_b, distance});
    }
  }
  std::sort(edges.begin(), edges.end(), [](const TopologyEdge &a, const TopologyEdge &b) {
    if (a.class_pair != b.class_pair) return a.class_pair < b.class_pair;
    return a.distance < b.distance;
  });
  return edges;
}

int classIdToBin(uint8_t class_id)
{
  for (size_t i = 0; i < kStaticClassIds.size(); ++i)
  {
    if (kStaticClassIds[i] == class_id)
    {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void l2Normalize(std::vector<float> &vec)
{
  double norm_sq = 0.0;
  for (float v : vec)
  {
    norm_sq += static_cast<double>(v) * static_cast<double>(v);
  }
  if (norm_sq <= 1e-12)
  {
    return;
  }
  const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm_sq));
  for (float &v : vec)
  {
    v *= inv_norm;
  }
}

IcpCloud::Ptr makeIcpCloud(const PointCloudXYZI::Ptr &cloud, double leaf_size)
{
  IcpCloud::Ptr xyz_cloud(new IcpCloud());
  if (!cloud)
  {
    return xyz_cloud;
  }

  xyz_cloud->reserve(cloud->size());
  for (const auto &pt : cloud->points)
  {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
    {
      continue;
    }
    xyz_cloud->push_back(IcpPoint(pt.x, pt.y, pt.z));
  }

  if (leaf_size <= 0.0 || xyz_cloud->size() < kMinIcpPoints)
  {
    return xyz_cloud;
  }

  // Avoid PCL voxel filter here to reduce runtime instability from mixed binary deps.
  IcpCloud::Ptr downsampled(new IcpCloud());
  downsampled->reserve(xyz_cloud->size());
  const double inv_leaf = 1.0 / std::max(0.05, leaf_size);
  std::unordered_set<VoxelKey, VoxelKeyHash> occupied;
  occupied.reserve(xyz_cloud->size());
  for (const auto &pt : xyz_cloud->points)
  {
    VoxelKey key;
    key.x = static_cast<int>(std::floor(pt.x * inv_leaf));
    key.y = static_cast<int>(std::floor(pt.y * inv_leaf));
    key.z = static_cast<int>(std::floor(pt.z * inv_leaf));
    if (occupied.insert(key).second)
    {
      downsampled->push_back(pt);
    }
  }
  return downsampled;
}

Eigen::Vector3d computeCentroid(const IcpCloud::Ptr &cloud)
{
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  if (!cloud || cloud->empty())
  {
    return centroid;
  }

  for (const auto &pt : cloud->points)
  {
    centroid += Eigen::Vector3d(pt.x, pt.y, pt.z);
  }
  centroid /= static_cast<double>(cloud->size());
  return centroid;
}

void coarseCentroidVerify(const IcpCloud::Ptr &query_cloud,
                          const IcpCloud::Ptr &target_cloud,
                          double max_correspondence_distance,
                          int &inliers,
                          double &fitness,
                          Eigen::Quaterniond &q_match_to_query,
                          Eigen::Vector3d &t_match_to_query,
                          int max_iterations)
{
  inliers = 0;
  fitness = -1.0;
  q_match_to_query = Eigen::Quaterniond::Identity();
  t_match_to_query = Eigen::Vector3d::Zero();
  if (!query_cloud || !target_cloud || query_cloud->empty() || target_cloud->empty())
  {
    return;
  }

  const double voxel = std::max(0.05, max_correspondence_distance);
  const double inv_voxel = 1.0 / voxel;
  const double max_dist_sq = max_correspondence_distance * max_correspondence_distance;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = computeCentroid(target_cloud) - computeCentroid(query_cloud);

  std::unordered_map<VoxelKey, std::vector<IcpPoint>, VoxelKeyHash> target_voxels;
  target_voxels.reserve(target_cloud->size());
  for (const auto &pt : target_cloud->points)
  {
    VoxelKey key;
    key.x = static_cast<int>(std::floor(pt.x * inv_voxel));
    key.y = static_cast<int>(std::floor(pt.y * inv_voxel));
    key.z = static_cast<int>(std::floor(pt.z * inv_voxel));
    target_voxels[key].push_back(pt);
  }

  const int max_iter = std::max(1, std::min(20, max_iterations));
  double best_fitness = -1.0;
  int best_inliers = 0;
  Eigen::Matrix3d best_R = R;
  Eigen::Vector3d best_t = t;

  for (int iter = 0; iter < max_iter; ++iter)
  {
    std::vector<Eigen::Vector3d> src_corr;
    std::vector<Eigen::Vector3d> dst_corr;
    src_corr.reserve(query_cloud->size());
    dst_corr.reserve(query_cloud->size());
    double sum_sq_dist = 0.0;
    int iter_inliers = 0;

    for (const auto &src : query_cloud->points)
    {
      const Eigen::Vector3d src_v(src.x, src.y, src.z);
      const Eigen::Vector3d transformed = R * src_v + t;
      const int qx = static_cast<int>(std::floor(transformed.x() * inv_voxel));
      const int qy = static_cast<int>(std::floor(transformed.y() * inv_voxel));
      const int qz = static_cast<int>(std::floor(transformed.z() * inv_voxel));

      double local_best_sq = max_dist_sq;
      bool matched = false;
      Eigen::Vector3d best_dst = transformed;
      for (int dx = -1; dx <= 1; ++dx)
      {
        for (int dy = -1; dy <= 1; ++dy)
        {
          for (int dz = -1; dz <= 1; ++dz)
          {
            VoxelKey key;
            key.x = qx + dx;
            key.y = qy + dy;
            key.z = qz + dz;
            const auto it = target_voxels.find(key);
            if (it == target_voxels.end())
            {
              continue;
            }
            for (const auto &tgt : it->second)
            {
              const double ddx = transformed.x() - tgt.x;
              const double ddy = transformed.y() - tgt.y;
              const double ddz = transformed.z() - tgt.z;
              const double dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
              if (dist_sq <= local_best_sq)
              {
                local_best_sq = dist_sq;
                matched = true;
                best_dst = Eigen::Vector3d(tgt.x, tgt.y, tgt.z);
              }
            }
          }
        }
      }

      if (matched)
      {
        ++iter_inliers;
        sum_sq_dist += local_best_sq;
        src_corr.push_back(src_v);
        dst_corr.push_back(best_dst);
      }
    }

    if (iter_inliers <= 0)
    {
      break;
    }

    const double iter_fitness = sum_sq_dist / static_cast<double>(iter_inliers);
    if (best_fitness < 0.0 || iter_fitness < best_fitness)
    {
      best_fitness = iter_fitness;
      best_inliers = iter_inliers;
      best_R = R;
      best_t = t;
    }

    if (static_cast<int>(src_corr.size()) < 3)
    {
      break;
    }

    Eigen::Vector3d src_centroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d dst_centroid = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < src_corr.size(); ++i)
    {
      src_centroid += src_corr[i];
      dst_centroid += dst_corr[i];
    }
    src_centroid /= static_cast<double>(src_corr.size());
    dst_centroid /= static_cast<double>(dst_corr.size());

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < src_corr.size(); ++i)
    {
      H += (src_corr[i] - src_centroid) * (dst_corr[i] - dst_centroid).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_new = svd.matrixV() * svd.matrixU().transpose();
    if (R_new.determinant() < 0.0)
    {
      Eigen::Matrix3d V = svd.matrixV();
      V.col(2) *= -1.0;
      R_new = V * svd.matrixU().transpose();
    }
    const Eigen::Vector3d t_new = dst_centroid - R_new * src_centroid;

    const double delta_t = (t_new - t).norm();
    const Eigen::Quaterniond q_delta(R_new * R.transpose());
    const double delta_r = q_delta.angularDistance(Eigen::Quaterniond::Identity());
    R = R_new;
    t = t_new;
    if (delta_t < 1e-4 && delta_r < 1e-4)
    {
      break;
    }
  }

  inliers = best_inliers;
  fitness = best_fitness;
  q_match_to_query = Eigen::Quaterniond(best_R);
  q_match_to_query.normalize();
  t_match_to_query = best_t;
}

void countTransformedInliers(const IcpCloud::Ptr &source_cloud,
                             const IcpCloud::Ptr &target_cloud,
                             const Eigen::Matrix4d &source_to_target,
                             double max_correspondence_distance,
                             int &inliers)
{
  inliers = 0;
  if (!source_cloud || !target_cloud || source_cloud->empty() || target_cloud->empty())
  {
    return;
  }

  const double voxel = std::max(0.05, max_correspondence_distance);
  const double inv_voxel = 1.0 / voxel;
  const double max_dist_sq = max_correspondence_distance * max_correspondence_distance;

  std::unordered_map<VoxelKey, std::vector<IcpPoint>, VoxelKeyHash> target_voxels;
  target_voxels.reserve(target_cloud->size());
  for (const auto &pt : target_cloud->points)
  {
    VoxelKey key;
    key.x = static_cast<int>(std::floor(pt.x * inv_voxel));
    key.y = static_cast<int>(std::floor(pt.y * inv_voxel));
    key.z = static_cast<int>(std::floor(pt.z * inv_voxel));
    target_voxels[key].push_back(pt);
  }

  const Eigen::Matrix3d R = source_to_target.block<3, 3>(0, 0);
  const Eigen::Vector3d t = source_to_target.block<3, 1>(0, 3);
  for (const auto &src : source_cloud->points)
  {
    const Eigen::Vector3d transformed = R * Eigen::Vector3d(src.x, src.y, src.z) + t;
    const int qx = static_cast<int>(std::floor(transformed.x() * inv_voxel));
    const int qy = static_cast<int>(std::floor(transformed.y() * inv_voxel));
    const int qz = static_cast<int>(std::floor(transformed.z() * inv_voxel));

    bool matched = false;
    for (int dx = -1; dx <= 1 && !matched; ++dx)
    {
      for (int dy = -1; dy <= 1 && !matched; ++dy)
      {
        for (int dz = -1; dz <= 1 && !matched; ++dz)
        {
          VoxelKey key;
          key.x = qx + dx;
          key.y = qy + dy;
          key.z = qz + dz;
          const auto it = target_voxels.find(key);
          if (it == target_voxels.end())
          {
            continue;
          }
          for (const auto &tgt : it->second)
          {
            const double ddx = transformed.x() - tgt.x;
            const double ddy = transformed.y() - tgt.y;
            const double ddz = transformed.z() - tgt.z;
            if ((ddx * ddx + ddy * ddy + ddz * ddz) <= max_dist_sq)
            {
              matched = true;
              break;
            }
          }
        }
      }
    }
    if (matched)
    {
      ++inliers;
    }
  }
}

bool runPclIcpWithInitialGuess(const IcpCloud::Ptr &query_cloud,
                               const IcpCloud::Ptr &hist_cloud,
                               const Eigen::Quaterniond &q_query_in_match_init,
                               const Eigen::Vector3d &t_query_in_match_init,
                               double max_correspondence_distance,
                               int max_iterations,
                               bool print_detail,
                               Eigen::Quaterniond &q_query_in_match,
                               Eigen::Vector3d &t_query_in_match,
                               int &inliers,
                               double &fitness)
{
  inliers = 0;
  fitness = -1.0;
  q_query_in_match = q_query_in_match_init;
  t_query_in_match = t_query_in_match_init;

  if (!query_cloud || !hist_cloud || query_cloud->size() < kMinIcpPoints || hist_cloud->size() < kMinIcpPoints)
  {
    return false;
  }

  Eigen::Quaterniond q_init = q_query_in_match_init;
  q_init.normalize();
  Eigen::Matrix4d initial_guess = Eigen::Matrix4d::Identity();
  initial_guess.block<3, 3>(0, 0) = q_init.toRotationMatrix();
  initial_guess.block<3, 1>(0, 3) = t_query_in_match_init;

  pcl::IterativeClosestPoint<IcpPoint, IcpPoint> icp;
  icp.setInputSource(query_cloud);
  icp.setInputTarget(hist_cloud);
  icp.setMaxCorrespondenceDistance(max_correspondence_distance);
  icp.setMaximumIterations(std::max(1, max_iterations));
  icp.setTransformationEpsilon(1e-7);
  icp.setEuclideanFitnessEpsilon(1e-7);

  IcpCloud aligned;
  icp.align(aligned, initial_guess.cast<float>());
  if (!icp.hasConverged())
  {
    return false;
  }

  const Eigen::Matrix4d final_transform = icp.getFinalTransformation().cast<double>();
  q_query_in_match = Eigen::Quaterniond(final_transform.block<3, 3>(0, 0));
  q_query_in_match.normalize();
  t_query_in_match = final_transform.block<3, 1>(0, 3);
  fitness = icp.getFitnessScore(max_correspondence_distance);
  countTransformedInliers(query_cloud, hist_cloud, final_transform, max_correspondence_distance, inliers);

  if (print_detail)
  {
    const double delta_t = (t_query_in_match - t_query_in_match_init).norm();
    const double delta_r = q_init.angularDistance(q_query_in_match) * 57.29577951308232;
    std::cout << ", init_delta_t=" << delta_t
              << ", init_delta_r_deg=" << delta_r;
  }
  return true;
}

double computeInfoDiagFromQuality(double v, double v_min, double v_max, double info_min, double info_max)
{
  const double vv = std::max(v_min, std::min(v_max, v));
  const double alpha = (vv - v_min) / std::max(1e-9, (v_max - v_min));
  return info_min + alpha * (info_max - info_min);
}

double computeRelativeRotationDeg(const Eigen::Quaterniond &qa, const Eigen::Quaterniond &qb)
{
  Eigen::Quaterniond q1 = qa;
  Eigen::Quaterniond q2 = qb;
  q1.normalize();
  q2.normalize();
  double dot = std::abs(q1.dot(q2));
  dot = std::max(-1.0, std::min(1.0, dot));
  constexpr double kRadToDeg = 57.29577951308232;
  return 2.0 * std::acos(dot) * kRadToDeg;
}
} // namespace

SemanticLoopManager::SemanticLoopManager(const SemanticLoopConfig &config)
    : config_(config), running_(false)
{
  config_.topology_verify_top_n = std::max(1, config_.topology_verify_top_n);
  config_.topology_min_nodes = std::max(2, config_.topology_min_nodes);
  config_.topology_min_edges = std::max(1, config_.topology_min_edges);
  config_.topology_max_edge_distance = std::max(0.1, config_.topology_max_edge_distance);
  config_.topology_edge_tolerance = std::max(0.01, config_.topology_edge_tolerance);
  config_.topology_score_threshold =
      std::max(0.0, std::min(1.0, config_.topology_score_threshold));
  config_.topology_fallback_voxel_size = std::max(0.05, config_.topology_fallback_voxel_size);
  config_.topology_fallback_inlier_ratio_threshold =
      std::max(0.0, std::min(1.0, config_.topology_fallback_inlier_ratio_threshold));
  config_.topology_fallback_min_inliers = std::max(1, config_.topology_fallback_min_inliers);
  config_.icp_min_inliers = std::max(3, config_.icp_min_inliers);
  initializeCandidateCsv();
  initializeEventCsv();
  initializeKeyframeCsv();
  initializeConstraintCsv();
  initializeLoopPcdDir();
}

SemanticLoopManager::~SemanticLoopManager()
{
  stop();
}

void SemanticLoopManager::start()
{
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true))
  {
    return;
  }
  worker_ = std::thread(&SemanticLoopManager::workerLoop, this);
}

void SemanticLoopManager::stop()
{
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false))
  {
    return;
  }

  cv_.notify_all();
  if (worker_.joinable())
  {
    worker_.join();
  }
}

void SemanticLoopManager::enqueueKeyframe(const SemanticKeyframeData &keyframe)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (static_cast<int>(queue_.size()) >= config_.max_queue_size)
  {
    queue_.pop();
    std::cout << "[SemanticLoop] queue full, drop oldest pending keyframe" << std::endl;
  }
  queue_.push(keyframe);
  cv_.notify_one();
}

void SemanticLoopManager::setLoopEventCallback(const std::function<void(const LoopEvent &)> &callback)
{
  loop_event_callback_ = callback;
}

void SemanticLoopManager::initializeCandidateCsv()
{
  const std::string log_dir = std::string(ROOT_DIR) + "Log";
  const std::string csv_path = log_dir + "/semantic_loop_candidates.csv";
  mkdir(log_dir.c_str(), 0755);

  bool need_header = true;
  {
    std::ifstream existing(csv_path.c_str(), std::ios::binary | std::ios::ate);
    if (existing.is_open())
    {
      need_header = (existing.tellg() == 0);
      existing.close();
    }
  }

  candidate_csv_.open(csv_path.c_str(), std::ios::out | std::ios::app);
  if (!candidate_csv_.is_open())
  {
    std::cout << "[SemanticLoop] failed to open candidate csv: " << csv_path << std::endl;
    return;
  }

  candidate_csv_ready_ = true;
  if (need_header)
  {
    candidate_csv_ << "query_id,query_time,query_travel,rank,cand_id,semantic_score,dt,path,euclid,"
                   << "geo_checked,geo_pass,inliers,ratio,semantic_accepted,"
                   << "icp_checked,icp_converged,icp_fitness,geometric_accepted\n";
  }
  std::cout << "[SemanticLoop] candidate csv: " << csv_path << std::endl;
}

void SemanticLoopManager::initializeEventCsv()
{
  const std::string log_dir = std::string(ROOT_DIR) + "Log";
  const std::string csv_path = log_dir + "/semantic_loop_events.csv";
  mkdir(log_dir.c_str(), 0755);

  bool need_header = true;
  {
    std::ifstream existing(csv_path.c_str(), std::ios::binary | std::ios::ate);
    if (existing.is_open())
    {
      need_header = (existing.tellg() == 0);
      existing.close();
    }
  }

  event_csv_.open(csv_path.c_str(), std::ios::out | std::ios::app);
  if (!event_csv_.is_open())
  {
    std::cout << "[SemanticLoop] failed to open event csv: " << csv_path << std::endl;
    return;
  }

  event_csv_ready_ = true;
  if (need_header)
  {
    event_csv_ << "query_id,query_time,query_travel,match_id,match_time,match_travel,rank,"
               << "semantic_score,dt,path,euclid,geo_inliers,geo_ratio,icp_fitness,"
               << "query_x,query_y,query_z,match_x,match_y,match_z,rel_x,rel_y,rel_z,rel_dist\n";
  }
  std::cout << "[SemanticLoop] event csv: " << csv_path << std::endl;
}

void SemanticLoopManager::initializeKeyframeCsv()
{
  const std::string log_dir = std::string(ROOT_DIR) + "Log";
  const std::string csv_path = log_dir + "/semantic_keyframes.csv";
  mkdir(log_dir.c_str(), 0755);

  bool need_header = true;
  {
    std::ifstream existing(csv_path.c_str(), std::ios::binary | std::ios::ate);
    if (existing.is_open())
    {
      need_header = (existing.tellg() == 0);
      existing.close();
    }
  }

  keyframe_csv_.open(csv_path.c_str(), std::ios::out | std::ios::app);
  if (!keyframe_csv_.is_open())
  {
    std::cout << "[SemanticLoop] failed to open keyframe csv: " << csv_path << std::endl;
    return;
  }

  keyframe_csv_ready_ = true;
  if (need_header)
  {
    keyframe_csv_ << "id,time,travel,x,y,z,qx,qy,qz,qw,static_pixels,cloud_points\n";
  }
  std::cout << "[SemanticLoop] keyframe csv: " << csv_path << std::endl;
}

void SemanticLoopManager::initializeConstraintCsv()
{
  const std::string log_dir = std::string(ROOT_DIR) + "Log";
  const std::string csv_path = log_dir + "/semantic_loop_constraints.csv";
  mkdir(log_dir.c_str(), 0755);

  bool need_header = true;
  {
    std::ifstream existing(csv_path.c_str(), std::ios::binary | std::ios::ate);
    if (existing.is_open())
    {
      need_header = (existing.tellg() == 0);
      existing.close();
    }
  }

  constraint_csv_.open(csv_path.c_str(), std::ios::out | std::ios::app);
  if (!constraint_csv_.is_open())
  {
    std::cout << "[SemanticLoop] failed to open constraint csv: " << csv_path << std::endl;
    return;
  }

  constraint_csv_ready_ = true;
  if (need_header)
  {
    constraint_csv_ << "edge_type,from_id,to_id,from_time,to_time,"
                    << "from_x,from_y,from_z,from_qx,from_qy,from_qz,from_qw,"
                    << "to_x,to_y,to_z,to_qx,to_qy,to_qz,to_qw,"
                    << "rel_x,rel_y,rel_z,rel_qx,rel_qy,rel_qz,rel_qw,rel_dist,"
                    << "info_x,info_y,info_z,info_roll,info_pitch,info_yaw,"
                    << "score,geo_ratio,geo_inliers,icp_fitness\n";
  }
  std::cout << "[SemanticLoop] constraint csv: " << csv_path << std::endl;
}

void SemanticLoopManager::initializeLoopPcdDir()
{
  const std::string log_dir = std::string(ROOT_DIR) + "Log";
  mkdir(log_dir.c_str(), 0755);
  if (config_.save_loop_event_pcd)
  {
    loop_pcd_dir_ = log_dir + "/semantic_loop_keyframes_pcd";
    mkdir(loop_pcd_dir_.c_str(), 0755);
    std::cout << "[SemanticLoop] loop keyframe pcd dir: " << loop_pcd_dir_ << std::endl;
  }
  loop_rgb_dir_ = log_dir + "/semantic_loop_keyframes_rgb";
  mkdir(loop_rgb_dir_.c_str(), 0755);
  std::cout << "[SemanticLoop] loop keyframe rgb dir: " << loop_rgb_dir_ << std::endl;
}

void SemanticLoopManager::saveSingleKeyframeCloud(const PointCloudXYZI::Ptr &cloud,
                                                  int keyframe_id,
                                                  double timestamp,
                                                  const std::string &role_tag)
{
  if (!config_.save_loop_event_pcd)
  {
    return;
  }
  if (!cloud || cloud->empty() || keyframe_id < 0)
  {
    return;
  }
  if (saved_loop_pcd_keyframe_ids_.count(keyframe_id) > 0)
  {
    return;
  }
  std::ostringstream oss;
  oss << loop_pcd_dir_ << "/kf_" << keyframe_id
      << "_" << role_tag
      << "_t_" << std::fixed << std::setprecision(6) << timestamp
      << ".pcd";
  const std::string pcd_path = oss.str();
  if (pcl::io::savePCDFileBinary(pcd_path, *cloud) == 0)
  {
    saved_loop_pcd_keyframe_ids_.insert(keyframe_id);
    if (config_.verbose_debug_log)
    {
      std::cout << "[SemanticLoop] saved loop keyframe pcd: " << pcd_path
                << " points=" << cloud->size() << std::endl;
    }
  }
  else
  {
    std::cout << "[SemanticLoop] failed to save loop keyframe pcd: " << pcd_path << std::endl;
  }
}

void SemanticLoopManager::saveSingleKeyframeImage(const cv::Mat &rgb_image,
                                                  int keyframe_id,
                                                  double timestamp,
                                                  const std::string &role_tag)
{
  if (rgb_image.empty() || keyframe_id < 0)
  {
    return;
  }
  if (saved_loop_rgb_keyframe_ids_.count(keyframe_id) > 0)
  {
    return;
  }
  cv::Mat out = rgb_image;
  if (out.type() != CV_8UC3)
  {
    if (out.type() == CV_8UC1)
    {
      cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
    }
    else
    {
      return;
    }
  }
  std::ostringstream oss;
  oss << loop_rgb_dir_ << "/kf_" << keyframe_id
      << "_" << role_tag
      << "_t_" << std::fixed << std::setprecision(6) << timestamp
      << ".png";
  const std::string image_path = oss.str();
  if (cv::imwrite(image_path, out))
  {
    saved_loop_rgb_keyframe_ids_.insert(keyframe_id);
    if (config_.verbose_debug_log)
    {
      std::cout << "[SemanticLoop] saved loop keyframe rgb: " << image_path
                << " (" << out.cols << "x" << out.rows << ")" << std::endl;
    }
  }
  else
  {
    std::cout << "[SemanticLoop] failed to save loop keyframe rgb: " << image_path << std::endl;
  }
}

void SemanticLoopManager::saveLoopKeyframeClouds(const SemanticKeyframeData &query,
                                                 const Candidate &candidate)
{
  saveSingleKeyframeCloud(query.static_cloud, query.id, query.timestamp, "query");
  saveSingleKeyframeCloud(query.static_cloud_local, query.id, query.timestamp, "query_local");
  saveSingleKeyframeImage(query.rgb_image, query.id, query.timestamp, "query");
  saveSingleKeyframeCloud(candidate.hist_cloud, candidate.id, candidate.hist_timestamp, "match");
  saveSingleKeyframeCloud(candidate.hist_cloud_local, candidate.id, candidate.hist_timestamp, "match_local");
  saveSingleKeyframeImage(candidate.hist_rgb_image, candidate.id, candidate.hist_timestamp, "match");
}

void SemanticLoopManager::appendCandidateCsvRow(const SemanticKeyframeData &query,
                                                const Candidate &candidate,
                                                int rank,
                                                bool semantic_accepted,
                                                bool geometric_accepted)
{
  if (!candidate_csv_ready_ || !candidate_csv_.is_open())
  {
    return;
  }

  candidate_csv_ << query.id << ","
                 << std::fixed << std::setprecision(6) << query.timestamp << ","
                 << std::setprecision(3) << query.travel_distance << ","
                 << rank << ","
                 << candidate.id << ","
                 << candidate.score << ","
                 << std::setprecision(3) << candidate.dt << ","
                 << candidate.path_dist << ","
                 << candidate.euclidean_dist << ","
                 << candidate.topology_checked << ","
                 << candidate.topology_pass << ","
                 << candidate.topology_matched_edges << ","
                 << candidate.topology_score << ","
                 << semantic_accepted << ","
                 << candidate.icp_checked << ","
                 << candidate.icp_converged << ","
                 << candidate.icp_fitness << ","
                 << geometric_accepted << "\n";
}

bool SemanticLoopManager::isSemanticAcceptedCandidate(const Candidate &candidate) const
{
  return candidate.topology_checked && candidate.topology_pass;
}

bool SemanticLoopManager::isGeometricAcceptedCandidate(const Candidate &candidate) const
{
  return isSemanticAcceptedCandidate(candidate) &&
         candidate.icp_checked &&
         candidate.icp_pass;
}

bool SemanticLoopManager::isSuppressedByGlobalLoopCooldown(const SemanticKeyframeData &query,
                                                           const Candidate &candidate) const
{
  for (const auto &accepted : accepted_loop_events_)
  {
    if (std::abs(candidate.id - accepted.match_id) > config_.global_loop_cooldown_match_id_window)
    {
      continue;
    }
    const double dt = std::abs(query.timestamp - accepted.query_time);
    const double dpath = std::abs(query.travel_distance - accepted.query_travel);
    if (dt < config_.global_loop_cooldown_time_gap || dpath < config_.global_loop_cooldown_path_gap)
    {
      return true;
    }
  }
  return false;
}

void SemanticLoopManager::appendLoopEventCsvRow(const SemanticKeyframeData &query,
                                                const Candidate &candidate,
                                                int rank)
{
  if (!event_csv_ready_ || !event_csv_.is_open())
  {
    return;
  }

  event_csv_ << query.id << ","
             << std::fixed << std::setprecision(6) << query.timestamp << ","
             << std::setprecision(3) << query.travel_distance << ","
             << candidate.id << ","
             << std::setprecision(6) << candidate.hist_timestamp << ","
             << std::setprecision(3) << candidate.hist_travel_distance << ","
             << rank << ","
             << candidate.score << ","
             << std::setprecision(3) << candidate.dt << ","
             << candidate.path_dist << ","
             << candidate.euclidean_dist << ","
             << candidate.topology_matched_edges << ","
             << candidate.topology_score << ","
             << candidate.icp_fitness << ","
             << query.position_x << ","
             << query.position_y << ","
             << query.position_z << ","
             << candidate.hist_position_x << ","
             << candidate.hist_position_y << ","
             << candidate.hist_position_z << ","
             << (query.position_x - candidate.hist_position_x) << ","
             << (query.position_y - candidate.hist_position_y) << ","
             << (query.position_z - candidate.hist_position_z) << ","
             << std::sqrt(
                    (query.position_x - candidate.hist_position_x) * (query.position_x - candidate.hist_position_x) +
                    (query.position_y - candidate.hist_position_y) * (query.position_y - candidate.hist_position_y) +
                    (query.position_z - candidate.hist_position_z) * (query.position_z - candidate.hist_position_z))
             << "\n";
}

void SemanticLoopManager::appendKeyframeCsvRow(const SemanticKeyframeData &keyframe)
{
  if (!keyframe_csv_ready_ || !keyframe_csv_.is_open())
  {
    return;
  }
  keyframe_csv_ << keyframe.id << ","
                << std::fixed << std::setprecision(6) << keyframe.timestamp << ","
                << std::setprecision(3) << keyframe.travel_distance << ","
                << keyframe.position_x << ","
                << keyframe.position_y << ","
                << keyframe.position_z << ","
                << keyframe.quat_x << ","
                << keyframe.quat_y << ","
                << keyframe.quat_z << ","
                << keyframe.quat_w << ","
                << keyframe.static_pixels << ","
                << keyframe.cloud_points << "\n";
}

void SemanticLoopManager::appendConstraintCsvRow(const SemanticKeyframeData &query,
                                                 const Candidate &candidate)
{
  if (!constraint_csv_ready_ || !constraint_csv_.is_open())
  {
    return;
  }

  Eigen::Vector3d t_rel;
  Eigen::Quaterniond q_rel;
  if (candidate.has_icp_relative_pose)
  {
    t_rel = Eigen::Vector3d(candidate.icp_rel_x, candidate.icp_rel_y, candidate.icp_rel_z);
    q_rel = Eigen::Quaterniond(candidate.icp_rel_qw, candidate.icp_rel_qx, candidate.icp_rel_qy, candidate.icp_rel_qz);
    q_rel.normalize();
  }
  else
  {
    const Eigen::Vector3d t_from(candidate.hist_position_x, candidate.hist_position_y, candidate.hist_position_z);
    const Eigen::Vector3d t_to(query.position_x, query.position_y, query.position_z);
    Eigen::Quaterniond q_from(candidate.hist_quat_w, candidate.hist_quat_x, candidate.hist_quat_y, candidate.hist_quat_z);
    Eigen::Quaterniond q_to(query.quat_w, query.quat_x, query.quat_y, query.quat_z);
    q_from.normalize();
    q_to.normalize();
    q_rel = q_from.conjugate() * q_to;
    t_rel = q_from.conjugate() * (t_to - t_from);
  }
  const double rel_dist = t_rel.norm();

  const double info_trans = computeInfoDiagFromQuality(candidate.topology_score, 0.0, 1.0, 20.0, 200.0);
  const double icp_q = (candidate.icp_fitness < 0.0) ? 0.0 : (1.0 / (1.0 + candidate.icp_fitness));
  const double info_rot = computeInfoDiagFromQuality(icp_q, 0.0, 1.0, 10.0, 100.0);

  constraint_csv_ << "semantic_loop,"
                  << candidate.id << ","
                  << query.id << ","
                  << std::fixed << std::setprecision(6) << candidate.hist_timestamp << ","
                  << query.timestamp << ","
                  << std::setprecision(3)
                  << candidate.hist_position_x << ","
                  << candidate.hist_position_y << ","
                  << candidate.hist_position_z << ","
                  << candidate.hist_quat_x << ","
                  << candidate.hist_quat_y << ","
                  << candidate.hist_quat_z << ","
                  << candidate.hist_quat_w << ","
                  << query.position_x << ","
                  << query.position_y << ","
                  << query.position_z << ","
                  << query.quat_x << ","
                  << query.quat_y << ","
                  << query.quat_z << ","
                  << query.quat_w << ","
                  << t_rel.x() << ","
                  << t_rel.y() << ","
                  << t_rel.z() << ","
                  << q_rel.x() << ","
                  << q_rel.y() << ","
                  << q_rel.z() << ","
                  << q_rel.w() << ","
                  << rel_dist << ","
                  << info_trans << ","
                  << info_trans << ","
                  << info_trans << ","
                  << info_rot << ","
                  << info_rot << ","
                  << info_rot << ","
                  << candidate.score << ","
                  << candidate.topology_score << ","
                  << candidate.topology_matched_edges << ","
                  << candidate.icp_fitness << "\n";
}

void SemanticLoopManager::workerLoop()
{
  while (true)
  {
    SemanticKeyframeData keyframe;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return !running_.load() || !queue_.empty(); });
      if (!running_.load() && queue_.empty())
      {
        break;
      }
      keyframe = queue_.front();
      queue_.pop();
    }

    computeDescriptors(keyframe);
    appendKeyframeCsvRow(keyframe);
    std::vector<Candidate> top_candidates = findTopCandidates(keyframe);
    if (config_.verbose_debug_log)
    {
      std::cout << "[SemanticLoop] debug keyframe id=" << keyframe.id
                << " candidates_after_semantic=" << top_candidates.size() << std::endl;
    }
    topologyVerifyCandidates(keyframe, top_candidates);
    if (config_.verbose_debug_log)
    {
      std::cout << "[SemanticLoop] debug keyframe id=" << keyframe.id
                << " topology_done" << std::endl;
    }
    icpVerifySemanticAcceptedCandidates(keyframe, top_candidates);
    if (config_.verbose_debug_log)
    {
      std::cout << "[SemanticLoop] debug keyframe id=" << keyframe.id
                << " icp_done" << std::endl;
    }

    if (config_.print_top_candidates)
    {
      std::cout << "[SemanticLoop] keyframe id=" << keyframe.id
                << " top-" << config_.top_k << " candidates:";
      if (top_candidates.empty())
      {
        std::cout << " none";
      }
      for (const auto &cand : top_candidates)
      {
        std::cout << " (id=" << cand.id
                  << ", score=" << std::fixed << std::setprecision(3) << cand.score
                  << ", dt=" << std::setprecision(2) << cand.dt << "s"
                  << ", path=" << cand.path_dist << "m"
                  << ", euclid=" << cand.euclidean_dist << "m";
        if (cand.topology_checked)
        {
          std::cout << ", verify_mode="
                    << (cand.topology_fallback_used ? "cloud_fallback" : "topology")
                    << ", topology=" << (cand.topology_pass ? "PASS" : "FAIL")
                    << ", matched_edges=" << cand.topology_matched_edges
                    << ", topology_score=" << std::setprecision(3) << cand.topology_score;
        }
        std::cout << ")";
      }
      std::cout << std::endl;
    }

    std::vector<int> geometric_selected_idx;
    for (int i = 0; i < static_cast<int>(top_candidates.size()); ++i)
    {
      const Candidate &cand = top_candidates[i];
      const bool semantic_accepted = isSemanticAcceptedCandidate(cand);
      bool geometric_accepted = isGeometricAcceptedCandidate(cand);
      if (geometric_accepted)
      {
        bool too_close_to_existing = false;
        for (int selected_idx : geometric_selected_idx)
        {
          const Candidate &selected = top_candidates[selected_idx];
          const bool close_in_time = std::abs(cand.dt - selected.dt) < config_.accept_cooldown_time_gap;
          const bool close_in_path = std::abs(cand.path_dist - selected.path_dist) < config_.accept_cooldown_path_gap;
          if (close_in_time || close_in_path)
          {
            too_close_to_existing = true;
            break;
          }
        }
        if (too_close_to_existing ||
            static_cast<int>(geometric_selected_idx.size()) >= config_.max_geometric_accepts_per_query)
        {
          geometric_accepted = false;
        }
        else
        {
          geometric_selected_idx.push_back(i);
        }
      }

      if (geometric_accepted && config_.max_relative_rotation_deg > 0.0)
      {
        const Eigen::Quaterniond q_query(keyframe.quat_w, keyframe.quat_x, keyframe.quat_y, keyframe.quat_z);
        const Eigen::Quaterniond q_match(cand.hist_quat_w, cand.hist_quat_x, cand.hist_quat_y, cand.hist_quat_z);
        const double rel_rot_deg = computeRelativeRotationDeg(q_match, q_query);
        if (rel_rot_deg > config_.max_relative_rotation_deg)
        {
          geometric_accepted = false;
          if (config_.verbose_debug_log)
          {
            std::cout << "[SemanticLoop] reject by rot gate: query=" << keyframe.id
                      << ", match=" << cand.id
                      << ", rel_rot=" << std::fixed << std::setprecision(2) << rel_rot_deg
                      << "deg > " << config_.max_relative_rotation_deg << "deg" << std::endl;
          }
        }
      }

      appendCandidateCsvRow(keyframe, cand, i + 1, semantic_accepted, geometric_accepted);
      if (geometric_accepted)
      {
        if (isSuppressedByGlobalLoopCooldown(keyframe, cand))
        {
          geometric_accepted = false;
        }
      }

      if (geometric_accepted)
      {
        Eigen::Quaterniond q_query(keyframe.quat_w, keyframe.quat_x, keyframe.quat_y, keyframe.quat_z);
        Eigen::Quaterniond q_match(cand.hist_quat_w, cand.hist_quat_x, cand.hist_quat_y, cand.hist_quat_z);
        q_query.normalize();
        q_match.normalize();
        const Eigen::Quaterniond q_rel_odom = q_match.conjugate() * q_query;
        const Eigen::Vector3d t_world(keyframe.position_x - cand.hist_position_x,
                                      keyframe.position_y - cand.hist_position_y,
                                      keyframe.position_z - cand.hist_position_z);
        const Eigen::Vector3d t_rel_odom = q_match.conjugate() * t_world;
        const Eigen::Quaterniond q_rel = cand.has_icp_relative_pose
                                             ? Eigen::Quaterniond(cand.icp_rel_qw, cand.icp_rel_qx, cand.icp_rel_qy, cand.icp_rel_qz)
                                             : q_rel_odom;
        const Eigen::Vector3d t_rel = cand.has_icp_relative_pose
                                          ? Eigen::Vector3d(cand.icp_rel_x, cand.icp_rel_y, cand.icp_rel_z)
                                          : t_rel_odom;

        std::cout << "[SemanticLoop] LOOP_EVENT query=" << keyframe.id
                  << ", match=" << cand.id
                  << ", rank=" << (i + 1)
                  << ", score=" << std::fixed << std::setprecision(3) << cand.score
                  << ", dt=" << std::setprecision(2) << cand.dt << "s"
                  << ", path=" << cand.path_dist << "m"
                  << ", euclid=" << cand.euclidean_dist << "m"
                  << ", topology_edges=" << cand.topology_matched_edges
                  << ", topology_score=" << std::setprecision(3) << cand.topology_score
                  << ", icp_fitness=" << cand.icp_fitness
                  << ", qpos=(" << keyframe.position_x << "," << keyframe.position_y << "," << keyframe.position_z << ")"
                  << ", mpos=(" << cand.hist_position_x << "," << cand.hist_position_y << "," << cand.hist_position_z << ")"
                  << ", rel_source=" << (cand.has_icp_relative_pose ? "icp_lite" : "odom")
                  << ", rel_pos=("
                  << t_rel.x() << ","
                  << t_rel.y() << ","
                  << t_rel.z() << ")"
                  << ", rel_dist="
                  << t_rel.norm()
                  << std::endl;
        appendLoopEventCsvRow(keyframe, cand, i + 1);
        appendConstraintCsvRow(keyframe, cand);
        saveLoopKeyframeClouds(keyframe, cand);

        if (loop_event_callback_)
        {
          LoopEvent event;
          event.query_id = keyframe.id;
          event.match_id = cand.id;
          event.rank = i + 1;
          event.score = cand.score;
          event.dt = cand.dt;
          event.path_dist = cand.path_dist;
          event.euclidean_dist = cand.euclidean_dist;
          event.geo_inliers = cand.topology_matched_edges;
          event.geo_ratio = cand.topology_score;
          event.icp_fitness = cand.icp_fitness;
          event.query_time = keyframe.timestamp;
          event.query_travel = keyframe.travel_distance;
          event.query_pos_x = keyframe.position_x;
          event.query_pos_y = keyframe.position_y;
          event.query_pos_z = keyframe.position_z;
          event.match_time = cand.hist_timestamp;
          event.match_travel = cand.hist_travel_distance;
          event.match_pos_x = cand.hist_position_x;
          event.match_pos_y = cand.hist_position_y;
          event.match_pos_z = cand.hist_position_z;
          event.query_quat_x = keyframe.quat_x;
          event.query_quat_y = keyframe.quat_y;
          event.query_quat_z = keyframe.quat_z;
          event.query_quat_w = keyframe.quat_w;
          event.match_quat_x = cand.hist_quat_x;
          event.match_quat_y = cand.hist_quat_y;
          event.match_quat_z = cand.hist_quat_z;
          event.match_quat_w = cand.hist_quat_w;
          event.rel_pos_x = t_rel.x();
          event.rel_pos_y = t_rel.y();
          event.rel_pos_z = t_rel.z();
          event.rel_quat_x = q_rel.x();
          event.rel_quat_y = q_rel.y();
          event.rel_quat_z = q_rel.z();
          event.rel_quat_w = q_rel.w();
          event.rel_dist = t_rel.norm();
          event.rel_pose_from_icp = cand.has_icp_relative_pose;
          loop_event_callback_(event);
          accepted_loop_events_.push_back(event);
        }
        else
        {
          LoopEvent event;
          event.query_id = keyframe.id;
          event.match_id = cand.id;
          event.rank = i + 1;
          event.query_time = keyframe.timestamp;
          event.query_travel = keyframe.travel_distance;
          event.query_quat_x = keyframe.quat_x;
          event.query_quat_y = keyframe.quat_y;
          event.query_quat_z = keyframe.quat_z;
          event.query_quat_w = keyframe.quat_w;
          event.match_quat_x = cand.hist_quat_x;
          event.match_quat_y = cand.hist_quat_y;
          event.match_quat_z = cand.hist_quat_z;
          event.match_quat_w = cand.hist_quat_w;
          event.rel_pos_x = t_rel.x();
          event.rel_pos_y = t_rel.y();
          event.rel_pos_z = t_rel.z();
          event.rel_quat_x = q_rel.x();
          event.rel_quat_y = q_rel.y();
          event.rel_quat_z = q_rel.z();
          event.rel_quat_w = q_rel.w();
          event.rel_dist = t_rel.norm();
          event.rel_pose_from_icp = cand.has_icp_relative_pose;
          accepted_loop_events_.push_back(event);
        }
      }
    }
    if (candidate_csv_ready_)
    {
      candidate_csv_.flush();
    }
    if (event_csv_ready_)
    {
      event_csv_.flush();
    }
    if (keyframe_csv_ready_)
    {
      keyframe_csv_.flush();
    }
    if (constraint_csv_ready_)
    {
      constraint_csv_.flush();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    history_.push_back(keyframe);
  }
}

void SemanticLoopManager::computeDescriptors(SemanticKeyframeData &keyframe) const
{
  const int hist_bins = static_cast<int>(kStaticClassIds.size());
  keyframe.global_hist.assign(hist_bins, 0.0f);
  keyframe.grid_hist.assign(kGridRows * kGridCols * hist_bins, 0.0f);

  if (keyframe.static_mask.empty() || keyframe.static_mask.type() != CV_8UC1)
  {
    return;
  }

  const int rows = keyframe.static_mask.rows;
  const int cols = keyframe.static_mask.cols;
  const int cell_h = std::max(1, rows / kGridRows);
  const int cell_w = std::max(1, cols / kGridCols);

  for (int y = 0; y < rows; ++y)
  {
    const uint8_t *ptr = keyframe.static_mask.ptr<uint8_t>(y);
    const int gy = std::min(kGridRows - 1, y / cell_h);
    for (int x = 0; x < cols; ++x)
    {
      const int bin = classIdToBin(ptr[x]);
      if (bin < 0)
      {
        continue;
      }

      keyframe.global_hist[bin] += 1.0f;
      const int gx = std::min(kGridCols - 1, x / cell_w);
      const int grid_idx = (gy * kGridCols + gx) * hist_bins + bin;
      keyframe.grid_hist[grid_idx] += 1.0f;
    }
  }

  l2Normalize(keyframe.global_hist);
  l2Normalize(keyframe.grid_hist);
}

double SemanticLoopManager::cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) const
{
  if (a.size() != b.size() || a.empty())
  {
    return 0.0;
  }

  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
  {
    const double av = static_cast<double>(a[i]);
    const double bv = static_cast<double>(b[i]);
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }

  if (na <= 1e-12 || nb <= 1e-12)
  {
    return 0.0;
  }
  return dot / std::sqrt(na * nb);
}

std::vector<SemanticLoopManager::Candidate> SemanticLoopManager::findTopCandidates(const SemanticKeyframeData &query) const
{
  std::vector<Candidate> candidates;
  std::lock_guard<std::mutex> lock(mutex_);
  candidates.reserve(history_.size());

  for (const auto &hist_kf : history_)
  {
    const double dt = query.timestamp - hist_kf.timestamp;
    const double dt_abs = std::abs(dt);
    if (dt_abs < config_.min_candidate_time_gap)
    {
      continue;
    }

    const double path_dist = std::abs(query.travel_distance - hist_kf.travel_distance);
    if (path_dist < config_.min_candidate_path_distance)
    {
      continue;
    }

    const double dx = query.position_x - hist_kf.position_x;
    const double dy = query.position_y - hist_kf.position_y;
    const double dz = query.position_z - hist_kf.position_z;
    const double spatial_dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (config_.min_candidate_euclidean_distance > 0.0 &&
        spatial_dist < config_.min_candidate_euclidean_distance)
    {
      continue;
    }
    if (config_.max_candidate_euclidean_distance > 0.0 &&
        spatial_dist > config_.max_candidate_euclidean_distance)
    {
      continue;
    }

    const double global_score = cosineSimilarity(query.global_hist, hist_kf.global_hist);
    const double grid_score = cosineSimilarity(query.grid_hist, hist_kf.grid_hist);
    const double final_score = 0.6 * global_score + 0.4 * grid_score;
    if (final_score < config_.score_threshold)
    {
      continue;
    }
    Candidate candidate;
    candidate.id = hist_kf.id;
    candidate.score = final_score;
    candidate.dt = dt;
    candidate.path_dist = path_dist;
    candidate.euclidean_dist = spatial_dist;
    candidate.hist_timestamp = hist_kf.timestamp;
    candidate.hist_travel_distance = hist_kf.travel_distance;
    candidate.hist_position_x = hist_kf.position_x;
    candidate.hist_position_y = hist_kf.position_y;
    candidate.hist_position_z = hist_kf.position_z;
    candidate.hist_quat_x = hist_kf.quat_x;
    candidate.hist_quat_y = hist_kf.quat_y;
    candidate.hist_quat_z = hist_kf.quat_z;
    candidate.hist_quat_w = hist_kf.quat_w;
    candidate.hist_cloud = hist_kf.static_cloud;
    candidate.hist_cloud_local = hist_kf.static_cloud_local;
    candidate.hist_rgb_image = hist_kf.rgb_image;
    candidate.hist_cloud_points = hist_kf.cloud_points;
    candidate.hist_topology_nodes = hist_kf.topology_nodes;
    candidates.push_back(candidate);
  }

  std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
    return a.score > b.score;
  });

  if (static_cast<int>(candidates.size()) > config_.top_k)
  {
    candidates.resize(config_.top_k);
  }
  return candidates;
}

void SemanticLoopManager::topologyVerifyCandidates(const SemanticKeyframeData &query,
                                                   std::vector<Candidate> &candidates) const
{
  if (!config_.topology_verify_enable)
  {
    for (auto &candidate : candidates)
    {
      candidate.topology_checked = true;
      candidate.topology_pass = true;
      candidate.topology_score = 1.0;
    }
    return;
  }

  const int verify_count = std::min<int>(config_.topology_verify_top_n, static_cast<int>(candidates.size()));
  for (int i = 0; i < verify_count; ++i)
  {
    Candidate &candidate = candidates[i];
    candidate.topology_checked = true;
    const auto runCloudFallback = [&]() {
      if (!config_.topology_fallback_enable)
      {
        return;
      }
      candidate.topology_fallback_used = true;
      computeFastOverlapRatio(query.static_cloud,
                              candidate.hist_cloud,
                              candidate.topology_matched_edges,
                              candidate.topology_score);
      candidate.topology_pass =
          candidate.topology_matched_edges >= config_.topology_fallback_min_inliers &&
          candidate.topology_score >= config_.topology_fallback_inlier_ratio_threshold;
      if (config_.verbose_debug_log)
      {
        std::cout << "[SemanticLoop] cloud fallback query=" << query.id
                  << ", cand=" << candidate.id
                  << ", pass=" << candidate.topology_pass
                  << ", inliers=" << candidate.topology_matched_edges
                  << ", ratio=" << std::fixed << std::setprecision(3)
                  << candidate.topology_score << std::endl;
      }
    };

    const auto &query_nodes = query.topology_nodes;
    const auto &hist_nodes = candidate.hist_topology_nodes;
    if (static_cast<int>(query_nodes.size()) < config_.topology_min_nodes ||
        static_cast<int>(hist_nodes.size()) < config_.topology_min_nodes)
    {
      runCloudFallback();
      continue;
    }

    std::unordered_map<int, int> query_class_counts;
    std::unordered_map<int, int> hist_class_counts;
    for (const auto &node : query_nodes) ++query_class_counts[node.class_id];
    for (const auto &node : hist_nodes) ++hist_class_counts[node.class_id];
    int matched_nodes = 0;
    for (const auto &entry : query_class_counts)
    {
      const auto it = hist_class_counts.find(entry.first);
      if (it != hist_class_counts.end())
      {
        matched_nodes += std::min(entry.second, it->second);
      }
    }
    const double node_score =
        static_cast<double>(matched_nodes) /
        static_cast<double>(std::max(query_nodes.size(), hist_nodes.size()));

    const std::vector<TopologyEdge> query_edges =
        buildTopologyEdges(query_nodes, config_.topology_max_edge_distance);
    const std::vector<TopologyEdge> hist_edges =
        buildTopologyEdges(hist_nodes, config_.topology_max_edge_distance);
    if (query_edges.empty() || hist_edges.empty())
    {
      runCloudFallback();
      continue;
    }

    std::unordered_set<int> hist_class_pairs;
    hist_class_pairs.reserve(hist_edges.size());
    for (const auto &edge : hist_edges)
    {
      hist_class_pairs.insert(edge.class_pair);
    }
    bool has_comparable_edge = false;
    for (const auto &edge : query_edges)
    {
      if (hist_class_pairs.count(edge.class_pair) > 0)
      {
        has_comparable_edge = true;
        break;
      }
    }
    if (!has_comparable_edge)
    {
      runCloudFallback();
      continue;
    }

    std::vector<bool> hist_used(hist_edges.size(), false);
    double matched_edge_quality = 0.0;
    for (const auto &query_edge : query_edges)
    {
      int best_index = -1;
      double best_error = config_.topology_edge_tolerance;
      for (size_t j = 0; j < hist_edges.size(); ++j)
      {
        if (hist_used[j] || hist_edges[j].class_pair != query_edge.class_pair)
        {
          continue;
        }
        const double error = std::abs(query_edge.distance - hist_edges[j].distance);
        if (error <= best_error)
        {
          best_error = error;
          best_index = static_cast<int>(j);
        }
      }
      if (best_index >= 0)
      {
        hist_used[best_index] = true;
        ++candidate.topology_matched_edges;
        matched_edge_quality +=
            1.0 - best_error / std::max(1e-6, config_.topology_edge_tolerance);
      }
    }

    const double edge_score =
        matched_edge_quality /
        static_cast<double>(std::max(query_edges.size(), hist_edges.size()));
    candidate.topology_score = 0.3 * node_score + 0.7 * edge_score;
    candidate.topology_pass =
        candidate.topology_matched_edges >= config_.topology_min_edges &&
        candidate.topology_score >= config_.topology_score_threshold;
  }
}

void SemanticLoopManager::icpVerifySemanticAcceptedCandidates(const SemanticKeyframeData &query,
                                                              std::vector<Candidate> &candidates) const
{
  if (!config_.icp_verify_enable || candidates.empty() || !query.static_cloud || query.static_cloud->empty())
  {
    return;
  }

  int verified_count = 0;
  for (auto &candidate : candidates)
  {
    if (!isSemanticAcceptedCandidate(candidate))
    {
      continue;
    }
    if (verified_count >= config_.icp_verify_top_n)
    {
      break;
    }
    ++verified_count;

    candidate.icp_checked = true;
    candidate.icp_converged = false;
    candidate.icp_pass = false;
    candidate.icp_fitness = -1.0;
    candidate.has_icp_relative_pose = false;
    candidate.icp_rel_x = 0.0;
    candidate.icp_rel_y = 0.0;
    candidate.icp_rel_z = 0.0;
    candidate.icp_rel_qx = 0.0;
    candidate.icp_rel_qy = 0.0;
    candidate.icp_rel_qz = 0.0;
    candidate.icp_rel_qw = 1.0;

    const bool has_local_clouds = query.static_cloud_local && !query.static_cloud_local->empty() &&
                                  candidate.hist_cloud_local && !candidate.hist_cloud_local->empty();
    if (!has_local_clouds)
    {
      if (config_.print_icp_detail)
      {
        std::cout << "[SemanticLoop] PCL ICP skip query=" << query.id
                  << ", cand=" << candidate.id
                  << ", reason=no_local_clouds" << std::endl;
      }
      continue;
    }

    const PointCloudXYZI::Ptr query_cloud_for_icp = query.static_cloud_local;
    const PointCloudXYZI::Ptr hist_cloud_for_icp = candidate.hist_cloud_local;

    if (!query_cloud_for_icp || query_cloud_for_icp->empty() ||
        !hist_cloud_for_icp || hist_cloud_for_icp->empty())
    {
      continue;
    }

    IcpCloud::Ptr query_icp = makeIcpCloud(query_cloud_for_icp, config_.icp_downsample_leaf_size);
    IcpCloud::Ptr hist_icp = makeIcpCloud(hist_cloud_for_icp, config_.icp_downsample_leaf_size);
    if (!query_icp || query_icp->size() < kMinIcpPoints ||
        !hist_icp || hist_icp->size() < kMinIcpPoints)
    {
      continue;
    }

    if (config_.print_icp_detail)
    {
      std::cout << "[SemanticLoop] PCL ICP start query=" << query.id
                << ", cand=" << candidate.id
                << ", mode=local"
                << ", source=" << query_icp->size()
                << ", target=" << hist_icp->size() << std::endl;
    }

    Eigen::Quaterniond q_query(query.quat_w, query.quat_x, query.quat_y, query.quat_z);
    Eigen::Quaterniond q_match(candidate.hist_quat_w,
                               candidate.hist_quat_x,
                               candidate.hist_quat_y,
                               candidate.hist_quat_z);
    q_query.normalize();
    q_match.normalize();
    const Eigen::Quaterniond q_rel_odom = q_match.conjugate() * q_query;
    const Eigen::Vector3d t_world(query.position_x - candidate.hist_position_x,
                                  query.position_y - candidate.hist_position_y,
                                  query.position_z - candidate.hist_position_z);
    const Eigen::Vector3d t_rel_odom = q_match.conjugate() * t_world;

    int icp_inliers = 0;
    double icp_fitness = -1.0;
    Eigen::Quaterniond icp_q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d icp_t = Eigen::Vector3d::Zero();
    const bool pcl_icp_converged =
        runPclIcpWithInitialGuess(query_icp,
                                  hist_icp,
                                  q_rel_odom,
                                  t_rel_odom,
                                  config_.icp_max_correspondence_distance,
                                  config_.icp_max_iterations,
                                  config_.print_icp_detail,
                                  icp_q,
                                  icp_t,
                                  icp_inliers,
                                  icp_fitness);
    candidate.icp_converged = pcl_icp_converged &&
                              (icp_inliers >= config_.icp_min_inliers);
    if (config_.print_icp_detail)
    {
      std::cout << "[SemanticLoop] PCL ICP done query=" << query.id
                << ", cand=" << candidate.id
                << ", converged=" << candidate.icp_converged
                << ", inliers=" << icp_inliers;
    }
    if (candidate.icp_converged)
    {
      candidate.icp_fitness = icp_fitness;
      candidate.icp_pass = (candidate.icp_fitness >= 0.0) &&
                           (candidate.icp_fitness <= config_.icp_fitness_threshold);
      if (candidate.icp_pass)
      {
        icp_q.normalize();
        candidate.has_icp_relative_pose = true;
        candidate.icp_rel_x = icp_t.x();
        candidate.icp_rel_y = icp_t.y();
        candidate.icp_rel_z = icp_t.z();
        candidate.icp_rel_qx = icp_q.x();
        candidate.icp_rel_qy = icp_q.y();
        candidate.icp_rel_qz = icp_q.z();
        candidate.icp_rel_qw = icp_q.w();
      }
      if (config_.print_icp_detail)
      {
        std::cout << ", fitness=" << candidate.icp_fitness
                  << ", pass=" << candidate.icp_pass
                  << ", rel_t=(" << candidate.icp_rel_x << ","
                  << candidate.icp_rel_y << "," << candidate.icp_rel_z << ")";
      }
    }
    if (config_.print_icp_detail)
    {
      std::cout << std::endl;
    }
  }
}

void SemanticLoopManager::computeFastOverlapRatio(const PointCloudXYZI::Ptr &query_cloud,
                                                  const PointCloudXYZI::Ptr &hist_cloud,
                                                  int &inliers,
                                                  double &inlier_ratio) const
{
  inliers = 0;
  inlier_ratio = 0.0;
  if (!query_cloud || !hist_cloud || query_cloud->empty() || hist_cloud->empty())
  {
    return;
  }

  const double voxel = config_.topology_fallback_voxel_size;
  const double inv_voxel = 1.0 / voxel;
  std::unordered_set<VoxelKey, VoxelKeyHash> hist_voxels;
  hist_voxels.reserve(hist_cloud->size());
  for (const auto &point : hist_cloud->points)
  {
    hist_voxels.insert({
        static_cast<int>(std::floor(point.x * inv_voxel)),
        static_cast<int>(std::floor(point.y * inv_voxel)),
        static_cast<int>(std::floor(point.z * inv_voxel))});
  }

  for (const auto &point : query_cloud->points)
  {
    const int qx = static_cast<int>(std::floor(point.x * inv_voxel));
    const int qy = static_cast<int>(std::floor(point.y * inv_voxel));
    const int qz = static_cast<int>(std::floor(point.z * inv_voxel));
    bool matched = false;
    for (int dx = -kNeighborRange; dx <= kNeighborRange && !matched; ++dx)
    {
      for (int dy = -kNeighborRange; dy <= kNeighborRange && !matched; ++dy)
      {
        for (int dz = -kNeighborRange; dz <= kNeighborRange; ++dz)
        {
          if (hist_voxels.count({qx + dx, qy + dy, qz + dz}) > 0)
          {
            ++inliers;
            matched = true;
            break;
          }
        }
      }
    }
  }

  inlier_ratio = static_cast<double>(inliers) /
                 static_cast<double>(std::max<size_t>(1, query_cloud->size()));
  inlier_ratio = std::max(0.0, std::min(1.0, inlier_ratio));
}
