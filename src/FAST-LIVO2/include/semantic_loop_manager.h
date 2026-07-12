/*
Semantic loop manager:
- Semantic histogram retrieval
- Lightweight 3D semantic topology verification
- ICP relative-pose verification
*/

#ifndef SEMANTIC_LOOP_MANAGER_H
#define SEMANTIC_LOOP_MANAGER_H

#include "common_lib.h"
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <thread>

struct SemanticTopologyNode
{
  int class_id = -1;
  double centroid_x = 0.0;
  double centroid_y = 0.0;
  double centroid_z = 0.0;
  double size_x = 0.0;
  double size_y = 0.0;
  double size_z = 0.0;
  int pixel_count = 0;
  int point_count = 0;
};

struct SemanticKeyframeData
{
  int id = -1;
  double timestamp = -1.0;
  double travel_distance = 0.0;      // accumulated trajectory length along semantic keyframes
  double position_x = 0.0;            // world-frame keyframe position, kept as plain doubles to avoid Eigen alignment issues
  double position_y = 0.0;
  double position_z = 0.0;
  double quat_x = 0.0;
  double quat_y = 0.0;
  double quat_z = 0.0;
  double quat_w = 1.0;
  cv::Mat static_mask;                // mono8, class-id mask
  PointCloudXYZI::Ptr static_cloud;   // world-frame static cloud snapshot
  PointCloudXYZI::Ptr static_cloud_local; // keyframe-local static cloud snapshot
  cv::Mat rgb_image;                  // bgr8 image near keyframe timestamp
  int static_pixels = 0;
  int cloud_points = 0;
  std::vector<float> global_hist;
  std::vector<float> grid_hist;
  std::vector<SemanticTopologyNode> topology_nodes;
};

struct SemanticLoopConfig
{
  int top_k = 5;
  double score_threshold = 0.55;
  int max_queue_size = 3;
  double min_candidate_time_gap = 10.0; // seconds
  double min_candidate_path_distance = 20.0;      // meters along trajectory
  double min_candidate_euclidean_distance = 0.0;  // optional hard gate, 0 disables
  double max_candidate_euclidean_distance = 0.0;  // optional hard gate, 0 disables
  bool topology_verify_enable = true;
  int topology_verify_top_n = 3;
  int topology_min_nodes = 3;
  int topology_min_edges = 2;
  double topology_max_edge_distance = 8.0;
  double topology_edge_tolerance = 0.5;
  double topology_score_threshold = 0.55;
  bool topology_fallback_enable = true;
  double topology_fallback_voxel_size = 0.8;
  double topology_fallback_inlier_ratio_threshold = 0.12;
  int topology_fallback_min_inliers = 80;
  bool icp_verify_enable = true;
  int icp_verify_top_n = 3;
  int icp_min_inliers = 80;
  double icp_downsample_leaf_size = 0.5;      // meters
  double icp_max_correspondence_distance = 1.5; // meters
  int icp_max_iterations = 30;
  double icp_fitness_threshold = 0.35;
  bool icp_use_local_cloud_for_relative_pose = true;
  bool verbose_debug_log = false;
  bool print_top_candidates = false;
  bool print_icp_detail = false;
  bool save_loop_event_pcd = false;
  int max_geometric_accepts_per_query = 1;
  double accept_cooldown_time_gap = 20.0;   // seconds
  double accept_cooldown_path_gap = 8.0;    // meters
  double global_loop_cooldown_time_gap = 8.0;   // seconds
  double global_loop_cooldown_path_gap = 5.0;   // meters
  int global_loop_cooldown_match_id_window = 2; // keyframe-id neighborhood
  double max_relative_rotation_deg = 0.0;       // 0 disables rotation gate
};

class SemanticLoopManager
{
public:
  struct LoopEvent
  {
    int query_id = -1;
    int match_id = -1;
    int rank = -1;
    double score = 0.0;
    double dt = 0.0;
    double path_dist = 0.0;
    double euclidean_dist = 0.0;
    int geo_inliers = 0;
    double geo_ratio = 0.0;
    double icp_fitness = -1.0;
    double query_time = -1.0;
    double query_travel = 0.0;
    double query_pos_x = 0.0;
    double query_pos_y = 0.0;
    double query_pos_z = 0.0;
    double match_time = -1.0;
    double match_travel = 0.0;
    double match_pos_x = 0.0;
    double match_pos_y = 0.0;
    double match_pos_z = 0.0;
    double query_quat_x = 0.0;
    double query_quat_y = 0.0;
    double query_quat_z = 0.0;
    double query_quat_w = 1.0;
    double match_quat_x = 0.0;
    double match_quat_y = 0.0;
    double match_quat_z = 0.0;
    double match_quat_w = 1.0;
    double rel_pos_x = 0.0;
    double rel_pos_y = 0.0;
    double rel_pos_z = 0.0;
    double rel_quat_x = 0.0;
    double rel_quat_y = 0.0;
    double rel_quat_z = 0.0;
    double rel_quat_w = 1.0;
    double rel_dist = 0.0;
    bool rel_pose_from_icp = false;
  };

  explicit SemanticLoopManager(const SemanticLoopConfig &config);
  ~SemanticLoopManager();

  void start();
  void stop();
  void enqueueKeyframe(const SemanticKeyframeData &keyframe);
  void setLoopEventCallback(const std::function<void(const LoopEvent &)> &callback);

private:
  struct Candidate
  {
    int id = -1;
    double score = 0.0;
    double dt = 0.0;
    double path_dist = 0.0;
    double euclidean_dist = 0.0;
    double hist_timestamp = -1.0;
    double hist_travel_distance = 0.0;
    double hist_position_x = 0.0;
    double hist_position_y = 0.0;
    double hist_position_z = 0.0;
    double hist_quat_x = 0.0;
    double hist_quat_y = 0.0;
    double hist_quat_z = 0.0;
    double hist_quat_w = 1.0;
    PointCloudXYZI::Ptr hist_cloud;
    PointCloudXYZI::Ptr hist_cloud_local;
    cv::Mat hist_rgb_image;
    int hist_cloud_points = 0;
    std::vector<SemanticTopologyNode> hist_topology_nodes;
    bool topology_checked = false;
    bool topology_pass = false;
    int topology_matched_edges = 0;
    double topology_score = 0.0;
    bool topology_fallback_used = false;
    bool icp_checked = false;
    bool icp_converged = false;
    bool icp_pass = false;
    double icp_fitness = -1.0;
    bool has_icp_relative_pose = false;
    double icp_rel_x = 0.0;
    double icp_rel_y = 0.0;
    double icp_rel_z = 0.0;
    double icp_rel_qx = 0.0;
    double icp_rel_qy = 0.0;
    double icp_rel_qz = 0.0;
    double icp_rel_qw = 1.0;
  };

  void workerLoop();
  void initializeCandidateCsv();
  void initializeEventCsv();
  void initializeKeyframeCsv();
  void initializeConstraintCsv();
  void initializeLoopPcdDir();
  void saveLoopKeyframeClouds(const SemanticKeyframeData &query,
                              const Candidate &candidate);
  void saveSingleKeyframeCloud(const PointCloudXYZI::Ptr &cloud,
                               int keyframe_id,
                               double timestamp,
                               const std::string &role_tag);
  void saveSingleKeyframeImage(const cv::Mat &rgb_image,
                               int keyframe_id,
                               double timestamp,
                               const std::string &role_tag);
  void appendCandidateCsvRow(const SemanticKeyframeData &query,
                             const Candidate &candidate,
                             int rank,
                             bool semantic_accepted,
                             bool geometric_accepted);
  void appendLoopEventCsvRow(const SemanticKeyframeData &query,
                             const Candidate &candidate,
                             int rank);
  void appendConstraintCsvRow(const SemanticKeyframeData &query,
                              const Candidate &candidate);
  void appendKeyframeCsvRow(const SemanticKeyframeData &keyframe);
  bool isSemanticAcceptedCandidate(const Candidate &candidate) const;
  bool isGeometricAcceptedCandidate(const Candidate &candidate) const;
  bool isSuppressedByGlobalLoopCooldown(const SemanticKeyframeData &query,
                                        const Candidate &candidate) const;
  void computeDescriptors(SemanticKeyframeData &keyframe) const;
  double cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) const;
  std::vector<Candidate> findTopCandidates(const SemanticKeyframeData &query) const;
  void topologyVerifyCandidates(const SemanticKeyframeData &query, std::vector<Candidate> &candidates) const;
  void icpVerifySemanticAcceptedCandidates(const SemanticKeyframeData &query, std::vector<Candidate> &candidates) const;
  void computeFastOverlapRatio(const PointCloudXYZI::Ptr &query_cloud,
                               const PointCloudXYZI::Ptr &hist_cloud,
                               int &inliers,
                               double &inlier_ratio) const;

  SemanticLoopConfig config_;
  std::atomic<bool> running_;
  std::thread worker_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<SemanticKeyframeData> queue_;
  std::vector<SemanticKeyframeData> history_;
  std::ofstream candidate_csv_;
  std::ofstream event_csv_;
  std::ofstream keyframe_csv_;
  std::ofstream constraint_csv_;
  bool candidate_csv_ready_ = false;
  bool event_csv_ready_ = false;
  bool keyframe_csv_ready_ = false;
  bool constraint_csv_ready_ = false;
  std::string loop_pcd_dir_;
  std::string loop_rgb_dir_;
  std::unordered_set<int> saved_loop_pcd_keyframe_ids_;
  std::unordered_set<int> saved_loop_rgb_keyframe_ids_;
  std::function<void(const LoopEvent &)> loop_event_callback_;
  std::vector<LoopEvent> accepted_loop_events_;
};

#endif
