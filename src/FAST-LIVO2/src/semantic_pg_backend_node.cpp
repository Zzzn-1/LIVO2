#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

#include <fast_livo/Keyframe.h>
#include <fast_livo/LoopConstraint.h>

#ifdef HAVE_GTSAM
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#endif

namespace {

double quatNorm(const geometry_msgs::Quaternion &q) {
  return std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
}

double transNorm(const geometry_msgs::Vector3 &t) {
  return std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
}

geometry_msgs::Quaternion quatConj(const geometry_msgs::Quaternion &q) {
  geometry_msgs::Quaternion out;
  out.x = -q.x;
  out.y = -q.y;
  out.z = -q.z;
  out.w = q.w;
  return out;
}

geometry_msgs::Quaternion quatMul(const geometry_msgs::Quaternion &a,
                                  const geometry_msgs::Quaternion &b) {
  geometry_msgs::Quaternion out;
  out.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
  out.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
  out.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
  out.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
  return out;
}

geometry_msgs::Quaternion quatNormalize(const geometry_msgs::Quaternion &q) {
  geometry_msgs::Quaternion out = q;
  const double n = quatNorm(q);
  if (n > 1e-12) {
    out.x /= n;
    out.y /= n;
    out.z /= n;
    out.w /= n;
  } else {
    out.x = 0.0;
    out.y = 0.0;
    out.z = 0.0;
    out.w = 1.0;
  }
  return out;
}

geometry_msgs::Vector3 rotateByQuatConj(const geometry_msgs::Quaternion &q,
                                        const geometry_msgs::Vector3 &v) {
  geometry_msgs::Quaternion vq;
  vq.x = v.x;
  vq.y = v.y;
  vq.z = v.z;
  vq.w = 0.0;
  const geometry_msgs::Quaternion qn = quatNormalize(q);
  const geometry_msgs::Quaternion qc = quatConj(qn);
  const geometry_msgs::Quaternion out_q = quatMul(quatMul(qc, vq), qn);
  geometry_msgs::Vector3 out;
  out.x = out_q.x;
  out.y = out_q.y;
  out.z = out_q.z;
  return out;
}

geometry_msgs::Vector3 rotateByQuat(const geometry_msgs::Quaternion &q,
                                    const geometry_msgs::Vector3 &v) {
  geometry_msgs::Quaternion vq;
  vq.x = v.x;
  vq.y = v.y;
  vq.z = v.z;
  vq.w = 0.0;
  const geometry_msgs::Quaternion qn = quatNormalize(q);
  const geometry_msgs::Quaternion qc = quatConj(qn);
  const geometry_msgs::Quaternion out_q = quatMul(quatMul(qn, vq), qc);
  geometry_msgs::Vector3 out;
  out.x = out_q.x;
  out.y = out_q.y;
  out.z = out_q.z;
  return out;
}

geometry_msgs::Quaternion quatSlerp(const geometry_msgs::Quaternion &qa,
                                    const geometry_msgs::Quaternion &qb,
                                    double t) {
  geometry_msgs::Quaternion a = quatNormalize(qa);
  geometry_msgs::Quaternion b = quatNormalize(qb);
  t = std::max(0.0, std::min(1.0, t));

  double dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  if (dot < 0.0) {
    dot = -dot;
    b.x = -b.x;
    b.y = -b.y;
    b.z = -b.z;
    b.w = -b.w;
  }

  const double kEps = 1e-9;
  if (1.0 - dot < 1e-6) {
    geometry_msgs::Quaternion out;
    out.x = a.x + t * (b.x - a.x);
    out.y = a.y + t * (b.y - a.y);
    out.z = a.z + t * (b.z - a.z);
    out.w = a.w + t * (b.w - a.w);
    return quatNormalize(out);
  }

  const double theta = std::acos(std::max(-1.0, std::min(1.0, dot)));
  const double sin_theta = std::sin(theta);
  if (std::abs(sin_theta) < kEps) {
    return a;
  }
  const double w1 = std::sin((1.0 - t) * theta) / sin_theta;
  const double w2 = std::sin(t * theta) / sin_theta;
  geometry_msgs::Quaternion out;
  out.x = w1 * a.x + w2 * b.x;
  out.y = w1 * a.y + w2 * b.y;
  out.z = w1 * a.z + w2 * b.z;
  out.w = w1 * a.w + w2 * b.w;
  return quatNormalize(out);
}

std::string loopTag(const fast_livo::LoopConstraint &loop) {
  std::ostringstream oss;
  oss << loop.from_id << "->" << loop.to_id << "@" << std::fixed << loop.to_time;
  return oss.str();
}

#ifdef HAVE_GTSAM
gtsam::Pose3 toPose3(const fast_livo::Keyframe &n) {
  const geometry_msgs::Quaternion q = quatNormalize(n.orientation);
  const gtsam::Rot3 R = gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z);
  return gtsam::Pose3(R, gtsam::Point3(n.position.x, n.position.y, n.position.z));
}

gtsam::Pose3 toRelPose3(double x, double y, double z, double qx, double qy, double qz, double qw) {
  geometry_msgs::Quaternion q;
  q.x = qx;
  q.y = qy;
  q.z = qz;
  q.w = qw;
  q = quatNormalize(q);
  const gtsam::Rot3 R = gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z);
  return gtsam::Pose3(R, gtsam::Point3(x, y, z));
}
#endif

}  // namespace

class SemanticPgBackendNode {
 public:
  SemanticPgBackendNode(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh) {
    pnh_.param<std::string>("keyframe_topic", keyframe_topic_, "/semantic_pg/keyframe");
    pnh_.param<std::string>("loop_topic", loop_topic_, "/semantic_pg/loop_constraint");
    pnh_.param<double>("min_loop_score", min_loop_score_, 0.0);
    pnh_.param<double>("min_geo_ratio", min_geo_ratio_, 0.0);
    pnh_.param<double>("max_icp_fitness", max_icp_fitness_, 1e9);
    pnh_.param<double>("max_loop_trans_m", max_loop_trans_m_, 1e9);
    pnh_.param<double>("max_pending_wait_s", max_pending_wait_s_, 10.0);
    pnh_.param<double>("status_period_s", status_period_s_, 1.0);
    pnh_.param<bool>("export_graph_csv", export_graph_csv_, true);
    pnh_.param<std::string>("export_dir", export_dir_, std::string(ROOT_DIR) + "Log");
    pnh_.param<std::string>("nodes_csv_name", nodes_csv_name_, "pose_graph_nodes_online.csv");
    pnh_.param<std::string>("edges_csv_name", edges_csv_name_, "pose_graph_edges_online.csv");
    pnh_.param<std::string>("corrected_path_tum_name", corrected_path_tum_name_, "corrected_path_online.txt");
    pnh_.param<std::string>("optimized_nodes_csv_name", optimized_nodes_csv_name_,
                            "pose_graph_optimized_nodes_online_backend.csv");
    pnh_.param<bool>("enable_optimize", enable_optimize_, true);
    pnh_.param<double>("odom_weight_scale", odom_weight_scale_, 0.5);
    pnh_.param<double>("loop_weight_scale", loop_weight_scale_, 0.2);
    pnh_.param<bool>("loop_robust_kernel_enable", loop_robust_kernel_enable_, true);
    pnh_.param<double>("loop_robust_huber_k", loop_robust_huber_k_, 1.345);
    pnh_.param<int>("optimize_every_n_keyframes", optimize_every_n_keyframes_, 10);
    pnh_.param<double>("optimize_period_s", optimize_period_s_, 2.0);
    pnh_.param<int>("min_nodes_for_optimize", min_nodes_for_optimize_, 5);
    pnh_.param<double>("isam2_relinearize_threshold", isam2_relinearize_threshold_, 0.01);
    pnh_.param<int>("isam2_relinearize_skip", isam2_relinearize_skip_, 1);
    pnh_.param<std::string>("frame_id", frame_id_, "camera_init");
    pnh_.param<std::string>("optimized_path_topic", optimized_path_topic_, "/semantic_pg/optimized_path");
    pnh_.param<std::string>("raw_path_topic", raw_path_topic_, "/semantic_pg/keyframe_path");
    pnh_.param<std::string>("input_path_topic", input_path_topic_, "/path");
    pnh_.param<std::string>("corrected_path_topic", corrected_path_topic_, "/semantic_pg/corrected_path");
    pnh_.param<std::string>("loop_markers_topic", loop_markers_topic_, "/semantic_pg/loop_markers");
    pnh_.param<std::string>("keyframe_markers_topic", keyframe_markers_topic_, "/semantic_pg/keyframe_markers");

    sub_keyframe_ = nh_.subscribe(keyframe_topic_, 500, &SemanticPgBackendNode::onKeyframe, this);
    sub_loop_ = nh_.subscribe(loop_topic_, 500, &SemanticPgBackendNode::onLoop, this);
    sub_input_path_ = nh_.subscribe(input_path_topic_, 10, &SemanticPgBackendNode::onInputPath, this);
    timer_status_ =
        nh_.createTimer(ros::Duration(status_period_s_), &SemanticPgBackendNode::onStatusTimer, this);
    timer_optimize_ =
        nh_.createTimer(ros::Duration(optimize_period_s_), &SemanticPgBackendNode::onOptimizeTimer, this);
    pub_keyframe_path_ = nh_.advertise<nav_msgs::Path>(raw_path_topic_, 2, true);
    pub_optimized_path_ = nh_.advertise<nav_msgs::Path>(optimized_path_topic_, 2, true);
    pub_corrected_path_ = nh_.advertise<nav_msgs::Path>(corrected_path_topic_, 2, true);
    pub_loop_markers_ = nh_.advertise<visualization_msgs::MarkerArray>(loop_markers_topic_, 2, true);
    pub_keyframe_markers_ =
        nh_.advertise<visualization_msgs::MarkerArray>(keyframe_markers_topic_, 2, true);

    ROS_INFO("[SemanticPgBackend] started. keyframe_topic=%s loop_topic=%s",
             keyframe_topic_.c_str(), loop_topic_.c_str());
    ROS_INFO(
        "[SemanticPgBackend] gates: min_loop_score=%.3f min_geo_ratio=%.3f max_icp_fitness=%.3f "
        "max_loop_trans_m=%.3f",
        min_loop_score_, min_geo_ratio_, max_icp_fitness_, max_loop_trans_m_);
    ROS_INFO("[SemanticPgBackend] export_graph_csv=%s dir=%s nodes=%s edges=%s",
             export_graph_csv_ ? "true" : "false", export_dir_.c_str(), nodes_csv_name_.c_str(),
             edges_csv_name_.c_str());
    ROS_INFO("[SemanticPgBackend] optimize enable=%s mode=isam2 odom_weight_scale=%.3f loop_weight_scale=%.3f loop_robust=%s huber_k=%.3f every_n=%d period=%.2fs min_nodes=%d frame_id=%s raw_topic=%s optimized_topic=%s input_path_topic=%s corrected_path_topic=%s loop_markers_topic=%s keyframe_markers_topic=%s",
             enable_optimize_ ? "true" : "false",
             odom_weight_scale_, loop_weight_scale_, loop_robust_kernel_enable_ ? "true" : "false", loop_robust_huber_k_,
             optimize_every_n_keyframes_, optimize_period_s_, min_nodes_for_optimize_, frame_id_.c_str(), raw_path_topic_.c_str(),
             optimized_path_topic_.c_str(), input_path_topic_.c_str(), corrected_path_topic_.c_str(), loop_markers_topic_.c_str(),
             keyframe_markers_topic_.c_str());
  }

 private:
  struct PendingLoop {
    fast_livo::LoopConstraint msg;
    ros::Time arrival_wall;
  };

  struct GraphEdgeRecord {
    std::string edge_type;
    uint32_t from_id = 0;
    uint32_t to_id = 0;
    double from_time = 0.0;
    double to_time = 0.0;
    double rel_x = 0.0;
    double rel_y = 0.0;
    double rel_z = 0.0;
    double rel_qx = 0.0;
    double rel_qy = 0.0;
    double rel_qz = 0.0;
    double rel_qw = 1.0;
    double info_x = 0.0;
    double info_y = 0.0;
    double info_z = 0.0;
    double info_roll = 0.0;
    double info_pitch = 0.0;
    double info_yaw = 0.0;
    double score = 0.0;
    double geo_ratio = 0.0;
    int geo_inliers = 0;
    double icp_fitness = 0.0;
  };

  struct GraphSnapshot {
    std::vector<fast_livo::Keyframe> nodes;
    std::vector<GraphEdgeRecord> edges;
  };

  struct OptimizeSummary {
    bool ok = false;
    size_t used_edges = 0;
    size_t skipped_edges = 0;
    double initial_error = 0.0;
    double final_error = 0.0;
  };

  void onKeyframe(const fast_livo::KeyframeConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!isSemanticKeyframe(*msg)) {
      ROS_WARN_THROTTLE(2.0,
                        "[SemanticPgBackend] ignore non-semantic keyframe id=%u source=%s",
                        msg->id, msg->source.c_str());
      return;
    }
    const bool is_new = keyframes_.count(msg->id) == 0;
    keyframes_[msg->id] = *msg;
    ++keyframes_received_;
    if (is_new) {
      graph_nodes_.insert(msg->id);
      if (has_last_backend_semantic_keyframe_) {
        addSemanticOdomEdgeIfPossibleLocked(last_backend_semantic_keyframe_id_, msg->id);
      }
      last_backend_semantic_keyframe_id_ = msg->id;
      has_last_backend_semantic_keyframe_ = true;
      graph_dirty_ = true;
      status_dirty_ = true;
      ++graph_revision_;
    }
    resolvePendingLocked();
  }

  void onLoop(const fast_livo::LoopConstraintConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    ++loops_received_;
    status_dirty_ = true;
    handleLoopLocked(*msg, false);
    cleanupPendingLocked();
  }

  void onInputPath(const nav_msgs::PathConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    latest_input_path_ = *msg;
  }

  bool passBasicGates(const fast_livo::LoopConstraint &loop, std::string *reason) const {
    if (loop.score < min_loop_score_) {
      *reason = "score";
      return false;
    }
    if (loop.geo_ratio < min_geo_ratio_) {
      *reason = "geo_ratio";
      return false;
    }
    if (loop.icp_fitness > max_icp_fitness_) {
      *reason = "icp_fitness";
      return false;
    }
    if (transNorm(loop.rel_translation) > max_loop_trans_m_) {
      *reason = "rel_translation";
      return false;
    }
    const double qn = quatNorm(loop.rel_rotation);
    if (std::abs(qn - 1.0) > 0.05) {
      *reason = "rel_rotation_norm";
      return false;
    }
    if (loop.to_time < loop.from_time) {
      *reason = "time_order";
      return false;
    }
    return true;
  }

  void handleLoopLocked(const fast_livo::LoopConstraint &loop, bool from_pending) {
    std::string reason;
    if (!passBasicGates(loop, &reason)) {
      ++loops_rejected_;
      ROS_WARN_THROTTLE(1.0, "[SemanticPgBackend] reject loop %s reason=%s", loopTag(loop).c_str(),
                        reason.c_str());
      return;
    }

    const bool has_from = keyframes_.count(loop.from_id) > 0;
    const bool has_to = keyframes_.count(loop.to_id) > 0;
    if (!has_from || !has_to) {
      if (!from_pending) {
        pending_loops_.push_back(PendingLoop{loop, ros::Time::now()});
      }
      return;
    }

    addLoopEdgeLocked(loop);
    ++loops_valid_;
    ROS_INFO_THROTTLE(1.0, "[SemanticPgBackend] valid loop accepted. total_valid=%zu", loops_valid_);
  }

  std::string edgeKey(uint32_t from_id, uint32_t to_id, const std::string &type) const {
    std::ostringstream oss;
    oss << type << ":" << from_id << "->" << to_id;
    return oss.str();
  }

  static bool isSemanticKeyframe(const fast_livo::Keyframe &kf) {
    return kf.source == "online_livmapper_semantic_keyframe";
  }

  void fillRelativeEdgeFromKeyframes(const fast_livo::Keyframe &a,
                                     const fast_livo::Keyframe &b,
                                     const std::string &edge_type,
                                     GraphEdgeRecord *edge) const {
    geometry_msgs::Vector3 t_map;
    t_map.x = b.position.x - a.position.x;
    t_map.y = b.position.y - a.position.y;
    t_map.z = b.position.z - a.position.z;
    const geometry_msgs::Vector3 t_rel = rotateByQuatConj(a.orientation, t_map);
    const geometry_msgs::Quaternion q_rel = quatNormalize(quatMul(quatConj(a.orientation), b.orientation));

    edge->edge_type = edge_type;
    edge->from_id = a.id;
    edge->to_id = b.id;
    edge->from_time = a.time;
    edge->to_time = b.time;
    edge->rel_x = t_rel.x;
    edge->rel_y = t_rel.y;
    edge->rel_z = t_rel.z;
    edge->rel_qx = q_rel.x;
    edge->rel_qy = q_rel.y;
    edge->rel_qz = q_rel.z;
    edge->rel_qw = q_rel.w;
    edge->score = 1.0;
    edge->geo_ratio = 1.0;
    edge->geo_inliers = 0;
    edge->icp_fitness = 0.0;
  }

  void addSemanticOdomEdgeIfPossibleLocked(uint32_t from_id, uint32_t to_id) {
    if (from_id == to_id) return;
    if (keyframes_.count(from_id) == 0 || keyframes_.count(to_id) == 0) return;
    const auto &a = keyframes_.at(from_id);
    const auto &b = keyframes_.at(to_id);
    if (!isSemanticKeyframe(a) || !isSemanticKeyframe(b)) return;

    const std::string key = edgeKey(from_id, to_id, "semantic_odom");
    if (!semantic_odom_edge_keys_.insert(key).second) return;

    GraphEdgeRecord edge;
    fillRelativeEdgeFromKeyframes(a, b, "semantic_odom", &edge);
    const double distance =
        std::sqrt(edge.rel_x * edge.rel_x + edge.rel_y * edge.rel_y + edge.rel_z * edge.rel_z);
    const double sigma_xy = 0.08 + 0.03 * distance;
    const double sigma_z = 0.12 + 0.05 * distance;
    constexpr double kDegToRad = 0.017453292519943295;
    const double sigma_roll_pitch = (5.0 + distance) * kDegToRad;
    const double sigma_yaw = (4.0 + 1.5 * distance) * kDegToRad;

    edge.info_x = 1.0 / (sigma_xy * sigma_xy);
    edge.info_y = edge.info_x;
    edge.info_z = 1.0 / (sigma_z * sigma_z);
    edge.info_roll = 1.0 / (sigma_roll_pitch * sigma_roll_pitch);
    edge.info_pitch = edge.info_roll;
    edge.info_yaw = 1.0 / (sigma_yaw * sigma_yaw);
    edge.score = 1.0;
    edge.geo_ratio = 1.0;
    edge.geo_inliers = 0;
    edge.icp_fitness = 0.0;
    semantic_odom_edges_map_[key] = edge;

    ++semantic_odom_edges_;
    graph_dirty_ = true;
    ++graph_revision_;
  }

  void addLoopEdgeLocked(const fast_livo::LoopConstraint &loop) {
    const std::string key = edgeKey(loop.from_id, loop.to_id, "semantic_loop");
    if (!loop_edge_keys_.insert(key).second) return;

    GraphEdgeRecord edge;
    edge.edge_type = "semantic_loop";
    edge.from_id = loop.from_id;
    edge.to_id = loop.to_id;
    edge.from_time = loop.from_time;
    edge.to_time = loop.to_time;
    edge.rel_x = loop.rel_translation.x;
    edge.rel_y = loop.rel_translation.y;
    edge.rel_z = loop.rel_translation.z;
    edge.rel_qx = loop.rel_rotation.x;
    edge.rel_qy = loop.rel_rotation.y;
    edge.rel_qz = loop.rel_rotation.z;
    edge.rel_qw = loop.rel_rotation.w;
    edge.info_x = loop.info_x;
    edge.info_y = loop.info_y;
    edge.info_z = loop.info_z;
    edge.info_roll = loop.info_roll;
    edge.info_pitch = loop.info_pitch;
    edge.info_yaw = loop.info_yaw;
    edge.score = loop.score;
    edge.geo_ratio = loop.geo_ratio;
    edge.geo_inliers = static_cast<int>(loop.geo_inliers);
    edge.icp_fitness = loop.icp_fitness;
    loop_edges_map_[key] = edge;

    ++loop_edges_;
    graph_dirty_ = true;
    ++graph_revision_;
  }

  void resolvePendingLocked() {
    if (pending_loops_.empty()) return;
    std::deque<PendingLoop> kept;
    for (const auto &pending : pending_loops_) {
      const bool has_from = keyframes_.count(pending.msg.from_id) > 0;
      const bool has_to = keyframes_.count(pending.msg.to_id) > 0;
      if (has_from && has_to) {
        handleLoopLocked(pending.msg, true);
      } else {
        kept.push_back(pending);
      }
    }
    pending_loops_.swap(kept);
  }

  void cleanupPendingLocked() {
    if (pending_loops_.empty()) return;
    std::deque<PendingLoop> kept;
    const ros::Time now = ros::Time::now();
    for (const auto &pending : pending_loops_) {
      const double wait_s = (now - pending.arrival_wall).toSec();
      if (wait_s > max_pending_wait_s_) {
        ++loops_rejected_;
        ROS_WARN_THROTTLE(1.0, "[SemanticPgBackend] drop stale pending loop %s wait=%.3fs",
                          loopTag(pending.msg).c_str(), wait_s);
      } else {
        kept.push_back(pending);
      }
    }
    pending_loops_.swap(kept);
  }

  void exportGraphCsvLocked() {
    const std::string nodes_path = export_dir_ + "/" + nodes_csv_name_;
    const std::string edges_path = export_dir_ + "/" + edges_csv_name_;
    std::ofstream nodes_out(nodes_path.c_str());
    if (!nodes_out.is_open()) {
      ROS_WARN_THROTTLE(2.0, "[SemanticPgBackend] failed to write nodes csv: %s", nodes_path.c_str());
      return;
    }
    std::ofstream edges_out(edges_path.c_str());
    if (!edges_out.is_open()) {
      ROS_WARN_THROTTLE(2.0, "[SemanticPgBackend] failed to write edges csv: %s", edges_path.c_str());
      return;
    }

    nodes_out << "id,time,x,y,z,qx,qy,qz,qw,travel\n";
    nodes_out << std::setprecision(17);
    for (const auto &it : keyframes_) {
      const auto &k = it.second;
      nodes_out << k.id << "," << k.time << "," << k.position.x << "," << k.position.y << ","
                << k.position.z << "," << k.orientation.x << "," << k.orientation.y << ","
                << k.orientation.z << "," << k.orientation.w << "," << k.travel << "\n";
    }

    edges_out << "edge_type,from_id,to_id,from_time,to_time,rel_x,rel_y,rel_z,rel_qx,rel_qy,rel_qz,rel_qw,info_x,info_y,info_z,info_roll,info_pitch,info_yaw,score,geo_ratio,geo_inliers,icp_fitness\n";
    edges_out << std::setprecision(17);
    std::vector<GraphEdgeRecord> all_edges;
    all_edges.reserve(semantic_odom_edges_map_.size() + loop_edges_map_.size());
    for (const auto &it : semantic_odom_edges_map_) all_edges.push_back(it.second);
    for (const auto &it : loop_edges_map_) all_edges.push_back(it.second);
    std::sort(all_edges.begin(), all_edges.end(),
              [](const GraphEdgeRecord &a, const GraphEdgeRecord &b) {
                if (a.from_id != b.from_id) return a.from_id < b.from_id;
                if (a.to_id != b.to_id) return a.to_id < b.to_id;
                return a.edge_type < b.edge_type;
              });
    for (const auto &e : all_edges) {
      edges_out << e.edge_type << "," << e.from_id << "," << e.to_id << "," << e.from_time << ","
                << e.to_time << "," << e.rel_x << "," << e.rel_y << "," << e.rel_z << ","
                << e.rel_qx << "," << e.rel_qy << "," << e.rel_qz << "," << e.rel_qw << ","
                << e.info_x << "," << e.info_y << "," << e.info_z << "," << e.info_roll << ","
                << e.info_pitch << "," << e.info_yaw << "," << e.score << "," << e.geo_ratio
                << "," << e.geo_inliers << "," << e.icp_fitness << "\n";
    }

    graph_dirty_ = false;

  }

  void exportCorrectedPathTum(const nav_msgs::Path &path) const {
    if (!export_graph_csv_ || path.poses.empty()) return;
    const std::string path_tum = export_dir_ + "/" + corrected_path_tum_name_;
    std::ofstream tum_out(path_tum.c_str());
    if (!tum_out.is_open()) {
      ROS_WARN_THROTTLE(2.0, "[SemanticPgBackend] failed to write corrected path tum txt: %s",
                        path_tum.c_str());
      return;
    }

    tum_out << std::fixed << std::setprecision(9);
    for (const auto &ps : path.poses) {
      const geometry_msgs::Quaternion q = quatNormalize(ps.pose.orientation);
      tum_out << ps.header.stamp.toSec() << " " << ps.pose.position.x << " "
              << ps.pose.position.y << " " << ps.pose.position.z << " "
              << q.x << " " << q.y << " " << q.z << " " << q.w << "\n";
    }
  }

  void exportOptimizedNodesCsv(const std::vector<fast_livo::Keyframe> &nodes) const {
    if (!export_graph_csv_ || nodes.empty()) return;
    const std::string nodes_path = export_dir_ + "/" + optimized_nodes_csv_name_;
    std::ofstream out(nodes_path.c_str());
    if (!out.is_open()) {
      ROS_WARN_THROTTLE(2.0, "[SemanticPgBackend] failed to write optimized nodes csv: %s",
                        nodes_path.c_str());
      return;
    }

    out << "id,time,x,y,z,qx,qy,qz,qw,travel\n";
    out << std::setprecision(17);
    for (const auto &n : nodes) {
      out << n.id << "," << n.time << "," << n.position.x << "," << n.position.y << ","
          << n.position.z << "," << n.orientation.x << "," << n.orientation.y << ","
          << n.orientation.z << "," << n.orientation.w << "," << n.travel << "\n";
    }
  }

  void onStatusTimer(const ros::TimerEvent &) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (export_graph_csv_ && graph_dirty_) {
      exportGraphCsvLocked();
    }
    if (!status_dirty_) {
      return;
    }
    ROS_INFO(
        "[SemanticPgBackend] keyframes_received=%zu unique_keyframes=%zu loops_received=%zu valid=%zu "
        "pending=%zu rejected=%zu graph_nodes=%zu semantic_odom_edges=%zu loop_edges=%zu",
        keyframes_received_, keyframes_.size(), loops_received_, loops_valid_, pending_loops_.size(),
        loops_rejected_, graph_nodes_.size(), semantic_odom_edges_, loop_edges_);
    status_dirty_ = false;
  }

  GraphSnapshot makeSnapshotLocked() const {
    GraphSnapshot s;
    s.nodes.reserve(keyframes_.size());
    for (const auto &kv : keyframes_) {
      s.nodes.push_back(kv.second);
    }
    std::sort(s.nodes.begin(), s.nodes.end(),
              [](const fast_livo::Keyframe &a, const fast_livo::Keyframe &b) { return a.id < b.id; });
    s.edges.reserve(semantic_odom_edges_map_.size() + loop_edges_map_.size());
    for (const auto &kv : semantic_odom_edges_map_) s.edges.push_back(kv.second);
    for (const auto &kv : loop_edges_map_) s.edges.push_back(kv.second);
    return s;
  }

  static nav_msgs::Path buildPathMsgFromNodes(const std::vector<fast_livo::Keyframe> &nodes,
                                              const ros::Time &stamp,
                                              const std::string &frame_id) {
    nav_msgs::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = frame_id;
    path.poses.reserve(nodes.size());
    for (const auto &n : nodes) {
      geometry_msgs::PoseStamped ps;
      ps.header.stamp = stamp;
      ps.header.frame_id = frame_id;
      ps.pose.position = n.position;
      ps.pose.orientation = quatNormalize(n.orientation);
      path.poses.push_back(ps);
    }
    return path;
  }

  static visualization_msgs::MarkerArray buildLoopMarkersMsg(
      const GraphSnapshot &snapshot, const ros::Time &stamp, const std::string &frame_id) {
    std::map<uint32_t, fast_livo::Keyframe> node_by_id;
    for (const auto &n : snapshot.nodes) node_by_id[n.id] = n;

    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker lines;
    lines.header.stamp = stamp;
    lines.header.frame_id = frame_id;
    lines.ns = "semantic_loop_lines";
    lines.id = 0;
    lines.type = visualization_msgs::Marker::LINE_LIST;
    lines.action = visualization_msgs::Marker::ADD;
    lines.scale.x = 0.03;
    lines.color.r = 0.0f;
    lines.color.g = 1.0f;
    lines.color.b = 0.0f;
    lines.color.a = 1.0f;

    visualization_msgs::Marker nodes;
    nodes.header.stamp = stamp;
    nodes.header.frame_id = frame_id;
    nodes.ns = "semantic_loop_nodes";
    nodes.id = 1;
    nodes.type = visualization_msgs::Marker::SPHERE_LIST;
    nodes.action = visualization_msgs::Marker::ADD;
    nodes.scale.x = 0.08;
    nodes.scale.y = 0.08;
    nodes.scale.z = 0.08;
    nodes.color.r = 1.0f;
    nodes.color.g = 0.9f;
    nodes.color.b = 0.0f;
    nodes.color.a = 1.0f;

    for (const auto &e : snapshot.edges) {
      if (e.edge_type != "semantic_loop") continue;
      const auto it_from = node_by_id.find(e.from_id);
      const auto it_to = node_by_id.find(e.to_id);
      if (it_from == node_by_id.end() || it_to == node_by_id.end()) continue;

      geometry_msgs::Point p_from;
      p_from.x = it_from->second.position.x;
      p_from.y = it_from->second.position.y;
      p_from.z = it_from->second.position.z;
      geometry_msgs::Point p_to;
      p_to.x = it_to->second.position.x;
      p_to.y = it_to->second.position.y;
      p_to.z = it_to->second.position.z;

      lines.points.push_back(p_from);
      lines.points.push_back(p_to);
      nodes.points.push_back(p_from);
      nodes.points.push_back(p_to);
    }

    marker_array.markers.push_back(lines);
    marker_array.markers.push_back(nodes);
    return marker_array;
  }

  static visualization_msgs::MarkerArray buildKeyframeMarkersMsg(
      const std::vector<fast_livo::Keyframe> &raw_nodes,
      const std::vector<fast_livo::Keyframe> &optimized_nodes,
      const ros::Time &stamp,
      const std::string &frame_id) {
    visualization_msgs::MarkerArray marker_array;

    visualization_msgs::Marker raw;
    raw.header.stamp = stamp;
    raw.header.frame_id = frame_id;
    raw.ns = "semantic_pg_raw_keyframes";
    raw.id = 0;
    raw.type = visualization_msgs::Marker::SPHERE_LIST;
    raw.action = visualization_msgs::Marker::ADD;
    raw.scale.x = 0.10;
    raw.scale.y = 0.10;
    raw.scale.z = 0.10;
    raw.color.r = 1.0f;
    raw.color.g = 1.0f;
    raw.color.b = 0.0f;
    raw.color.a = 1.0f;

    visualization_msgs::Marker optimized;
    optimized.header.stamp = stamp;
    optimized.header.frame_id = frame_id;
    optimized.ns = "semantic_pg_optimized_keyframes";
    optimized.id = 1;
    optimized.type = visualization_msgs::Marker::SPHERE_LIST;
    optimized.action = visualization_msgs::Marker::ADD;
    optimized.scale.x = 0.14;
    optimized.scale.y = 0.14;
    optimized.scale.z = 0.14;
    optimized.color.r = 1.0f;
    optimized.color.g = 0.1f;
    optimized.color.b = 0.1f;
    optimized.color.a = 1.0f;

    raw.points.reserve(raw_nodes.size());
    for (const auto &n : raw_nodes) {
      geometry_msgs::Point p;
      p.x = n.position.x;
      p.y = n.position.y;
      p.z = n.position.z;
      raw.points.push_back(p);
    }

    optimized.points.reserve(optimized_nodes.size());
    for (const auto &n : optimized_nodes) {
      geometry_msgs::Point p;
      p.x = n.position.x;
      p.y = n.position.y;
      p.z = n.position.z;
      optimized.points.push_back(p);
    }

    marker_array.markers.push_back(raw);
    marker_array.markers.push_back(optimized);
    return marker_array;
  }

  nav_msgs::Path buildCorrectedPathMsg(const nav_msgs::Path &input_path,
                                       const std::vector<fast_livo::Keyframe> &raw_nodes,
                                       const std::vector<fast_livo::Keyframe> &optimized_nodes,
                                       const ros::Time &stamp) const {
    nav_msgs::Path corrected = input_path;
    corrected.header.stamp = stamp;
    if (corrected.header.frame_id.empty()) {
      corrected.header.frame_id = frame_id_;
    }
    if (input_path.poses.empty() || raw_nodes.empty() || optimized_nodes.empty()) {
      return corrected;
    }

    struct DeltaKf {
      double time = 0.0;
      geometry_msgs::Vector3 t;
      geometry_msgs::Quaternion q;
    };
    std::map<uint32_t, fast_livo::Keyframe> opt_by_id;
    for (const auto &n : optimized_nodes) opt_by_id[n.id] = n;
    std::vector<DeltaKf> deltas;
    deltas.reserve(raw_nodes.size());
    for (const auto &raw_n : raw_nodes) {
      const auto it = opt_by_id.find(raw_n.id);
      if (it == opt_by_id.end()) continue;
      const auto &opt_n = it->second;

      DeltaKf d;
      d.time = raw_n.time;
      d.q = quatNormalize(quatMul(quatNormalize(opt_n.orientation),
                                  quatConj(quatNormalize(raw_n.orientation))));
      geometry_msgs::Vector3 raw_p;
      raw_p.x = raw_n.position.x;
      raw_p.y = raw_n.position.y;
      raw_p.z = raw_n.position.z;
      const geometry_msgs::Vector3 raw_p_rot = rotateByQuat(d.q, raw_p);
      d.t.x = opt_n.position.x - raw_p_rot.x;
      d.t.y = opt_n.position.y - raw_p_rot.y;
      d.t.z = opt_n.position.z - raw_p_rot.z;
      deltas.push_back(d);
    }
    if (deltas.empty()) return corrected;
    std::sort(deltas.begin(), deltas.end(),
              [](const DeltaKf &a, const DeltaKf &b) { return a.time < b.time; });

    for (size_t i = 0; i < corrected.poses.size(); ++i) {
      auto &ps = corrected.poses[i];
      const double pose_t = ps.header.stamp.isZero() ? corrected.header.stamp.toSec()
                                                     : ps.header.stamp.toSec();
      size_t hi = 0;
      while (hi < deltas.size() && deltas[hi].time < pose_t) {
        ++hi;
      }

      geometry_msgs::Vector3 t_delta;
      geometry_msgs::Quaternion q_delta;
      if (hi == 0) {
        t_delta = deltas.front().t;
        q_delta = deltas.front().q;
      } else if (hi >= deltas.size()) {
        t_delta = deltas.back().t;
        q_delta = deltas.back().q;
      } else {
        const DeltaKf &a = deltas[hi - 1];
        const DeltaKf &b = deltas[hi];
        const double denom = std::max(1e-9, b.time - a.time);
        const double ratio = std::max(0.0, std::min(1.0, (pose_t - a.time) / denom));
        t_delta.x = a.t.x + ratio * (b.t.x - a.t.x);
        t_delta.y = a.t.y + ratio * (b.t.y - a.t.y);
        t_delta.z = a.t.z + ratio * (b.t.z - a.t.z);
        q_delta = quatSlerp(a.q, b.q, ratio);
      }

      geometry_msgs::Vector3 p;
      p.x = ps.pose.position.x;
      p.y = ps.pose.position.y;
      p.z = ps.pose.position.z;
      const geometry_msgs::Vector3 p_rot = rotateByQuat(q_delta, p);
      ps.pose.position.x = p_rot.x + t_delta.x;
      ps.pose.position.y = p_rot.y + t_delta.y;
      ps.pose.position.z = p_rot.z + t_delta.z;
      ps.pose.orientation = quatNormalize(quatMul(q_delta, ps.pose.orientation));
      if (ps.header.stamp.isZero()) {
        ps.header.stamp = ros::Time(pose_t);
      }
      if (ps.header.frame_id.empty()) {
        ps.header.frame_id = corrected.header.frame_id;
      }
    }
    return corrected;
  }

#ifdef HAVE_GTSAM
  static void addPriorFactorForNode(const fast_livo::Keyframe &first,
                                    gtsam::NonlinearFactorGraph *graph) {
    const gtsam::Vector6 prior_sigmas(
        (gtsam::Vector6() << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
    graph->add(gtsam::PriorFactor<gtsam::Pose3>(
        gtsam::Symbol('x', static_cast<size_t>(first.id)),
        toPose3(first),
        gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas)));
  }

  double edgeWeightScale(const GraphEdgeRecord &e) const {
    return std::max(e.edge_type == "semantic_loop" ? loop_weight_scale_ : odom_weight_scale_, 1e-9);
  }

  gtsam::SharedNoiseModel makeNoiseModel(const GraphEdgeRecord &e) const {
    const double weight_scale = edgeWeightScale(e);
    const gtsam::Vector6 info((gtsam::Vector6() << std::max(e.info_roll * weight_scale, 1e-9), std::max(e.info_pitch * weight_scale, 1e-9),
                               std::max(e.info_yaw * weight_scale, 1e-9), std::max(e.info_x * weight_scale, 1e-9),
                               std::max(e.info_y * weight_scale, 1e-9), std::max(e.info_z * weight_scale, 1e-9))
                                  .finished());
    const gtsam::Vector6 sigmas = info.array().sqrt().inverse().matrix();
    gtsam::SharedNoiseModel gaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    if (e.edge_type == "semantic_loop" && loop_robust_kernel_enable_)
    {
      const double huber_k = std::max(loop_robust_huber_k_, 1e-9);
      return gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(huber_k),
          gaussian);
    }
    return gaussian;
  }

  void addBetweenFactor(const GraphEdgeRecord &e, gtsam::NonlinearFactorGraph *graph) const {
    const gtsam::Pose3 rel = toRelPose3(e.rel_x, e.rel_y, e.rel_z, e.rel_qx, e.rel_qy, e.rel_qz, e.rel_qw);
    graph->add(gtsam::BetweenFactor<gtsam::Pose3>(
        gtsam::Symbol('x', static_cast<size_t>(e.from_id)),
        gtsam::Symbol('x', static_cast<size_t>(e.to_id)),
        rel,
        makeNoiseModel(e)));
  }

  static std::string fullEdgeKey(const GraphEdgeRecord &e) {
    std::ostringstream oss;
    oss << e.edge_type << ":" << e.from_id << "->" << e.to_id;
    return oss.str();
  }
#endif

  OptimizeSummary optimizeSnapshotIsam2(const GraphSnapshot &snapshot,
                                        std::vector<fast_livo::Keyframe> *optimized_nodes) {
    OptimizeSummary summary;
    *optimized_nodes = snapshot.nodes;
#ifdef HAVE_GTSAM
    if (snapshot.nodes.empty()) return summary;

    if (!isam2_) {
      gtsam::ISAM2Params params;
      params.relinearizeThreshold = isam2_relinearize_threshold_;
      params.relinearizeSkip = std::max(1, isam2_relinearize_skip_);
      isam2_.reset(new gtsam::ISAM2(params));
      isam2_seen_node_ids_.clear();
      isam2_seen_edge_keys_.clear();
      isam2_prior_added_ = false;
    }

    gtsam::NonlinearFactorGraph new_factors;
    gtsam::Values new_values;
    std::map<uint32_t, fast_livo::Keyframe> node_by_id;
    for (const auto &n : snapshot.nodes) {
      node_by_id[n.id] = n;
    }

    if (!isam2_prior_added_) {
      const fast_livo::Keyframe &first = snapshot.nodes.front();
      addPriorFactorForNode(first, &new_factors);
      if (!new_values.exists(gtsam::Symbol('x', static_cast<size_t>(first.id)))) {
        new_values.insert(gtsam::Symbol('x', static_cast<size_t>(first.id)), toPose3(first));
      }
      isam2_seen_node_ids_.insert(first.id);
      isam2_prior_added_ = true;
    }

    for (const auto &n : snapshot.nodes) {
      if (isam2_seen_node_ids_.insert(n.id).second) {
        new_values.insert(gtsam::Symbol('x', static_cast<size_t>(n.id)), toPose3(n));
      }
    }

    for (const auto &e : snapshot.edges) {
      const std::string ek = fullEdgeKey(e);
      if (!isam2_seen_edge_keys_.insert(ek).second) continue;
      if (node_by_id.count(e.from_id) == 0 || node_by_id.count(e.to_id) == 0) {
        ++summary.skipped_edges;
        continue;
      }
      addBetweenFactor(e, &new_factors);
      ++summary.used_edges;
    }

    gtsam::Values estimate_before;
    if (isam2_has_result_) {
      estimate_before = isam2_->calculateEstimate();
    }
    {
      gtsam::NonlinearFactorGraph full_graph;
      addPriorFactorForNode(snapshot.nodes.front(), &full_graph);
      for (const auto &e : snapshot.edges) {
        if (node_by_id.count(e.from_id) == 0 || node_by_id.count(e.to_id) == 0) continue;
        addBetweenFactor(e, &full_graph);
      }
      gtsam::Values initial_eval;
      if (isam2_has_result_) {
        initial_eval = estimate_before;
        // For newly arrived nodes, ISAM2 has no estimate yet before update.
        // Seed missing keys with current keyframe pose to evaluate a valid initial error.
        for (const auto &n : snapshot.nodes) {
          const gtsam::Symbol key('x', static_cast<size_t>(n.id));
          if (!initial_eval.exists(key)) {
            initial_eval.insert(key, toPose3(n));
          }
        }
      } else {
        for (const auto &n : snapshot.nodes) {
          initial_eval.insert(gtsam::Symbol('x', static_cast<size_t>(n.id)), toPose3(n));
        }
      }
      summary.initial_error = full_graph.error(initial_eval);
    }

    if (new_factors.size() > 0 || new_values.size() > 0) {
      isam2_->update(new_factors, new_values);
      isam2_has_result_ = true;
    }

    gtsam::Values estimate_after = isam2_->calculateEstimate();
    {
      gtsam::NonlinearFactorGraph full_graph;
      addPriorFactorForNode(snapshot.nodes.front(), &full_graph);
      for (const auto &e : snapshot.edges) {
        if (node_by_id.count(e.from_id) == 0 || node_by_id.count(e.to_id) == 0) continue;
        addBetweenFactor(e, &full_graph);
      }
      summary.final_error = full_graph.error(estimate_after);
    }

    optimized_nodes->clear();
    optimized_nodes->reserve(snapshot.nodes.size());
    for (const auto &n : snapshot.nodes) {
      fast_livo::Keyframe out = n;
      const gtsam::Pose3 p = estimate_after.at<gtsam::Pose3>(gtsam::Symbol('x', static_cast<size_t>(n.id)));
      const gtsam::Point3 t = p.translation();
      out.position.x = t.x();
      out.position.y = t.y();
      out.position.z = t.z();
      const auto q = p.rotation().toQuaternion();
      out.orientation.x = q.x();
      out.orientation.y = q.y();
      out.orientation.z = q.z();
      out.orientation.w = q.w();
      optimized_nodes->push_back(out);
    }
    summary.ok = true;
#else
    (void)snapshot;
#endif
    return summary;
  }

  void onOptimizeTimer(const ros::TimerEvent &) {
    if (!enable_optimize_) return;

    GraphSnapshot snapshot;
    size_t keyframe_count = 0;
    nav_msgs::Path input_path_snapshot;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      keyframe_count = keyframes_.size();
      if (keyframe_count < static_cast<size_t>(std::max(1, min_nodes_for_optimize_))) {
        return;
      }
      if (graph_revision_ == last_optimized_graph_revision_) {
        return;
      }
      const size_t delta_kf = keyframe_count - last_optimized_keyframe_count_;
      const bool n_trigger = delta_kf >= static_cast<size_t>(std::max(1, optimize_every_n_keyframes_));
      const bool first_run = last_optimize_wall_.isZero();
      const bool t_trigger = first_run || ((ros::Time::now() - last_optimize_wall_).toSec() >= optimize_period_s_);
      if (!n_trigger && !t_trigger) {
        return;
      }
      snapshot = makeSnapshotLocked();
      input_path_snapshot = latest_input_path_;
      last_optimized_keyframe_count_ = keyframe_count;
      last_optimized_graph_revision_ = graph_revision_;
      last_optimize_wall_ = ros::Time::now();
    }

    std::vector<fast_livo::Keyframe> optimized_nodes;
    const OptimizeSummary summary = optimizeSnapshotIsam2(snapshot, &optimized_nodes);
    const bool use_optimized = summary.ok && !optimized_nodes.empty();
    const std::vector<fast_livo::Keyframe> &path_nodes = use_optimized ? optimized_nodes : snapshot.nodes;

    const ros::Time now = ros::Time::now();
    const nav_msgs::Path raw_path_msg = buildPathMsgFromNodes(snapshot.nodes, now, frame_id_);
    pub_keyframe_path_.publish(raw_path_msg);
    const nav_msgs::Path path_msg = buildPathMsgFromNodes(path_nodes, now, frame_id_);
    pub_optimized_path_.publish(path_msg);
    exportOptimizedNodesCsv(path_nodes);
    const nav_msgs::Path corrected_path_msg =
        buildCorrectedPathMsg(input_path_snapshot, snapshot.nodes, path_nodes, now);
    if (!corrected_path_msg.poses.empty()) {
      pub_corrected_path_.publish(corrected_path_msg);
      exportCorrectedPathTum(corrected_path_msg);
    }
    const visualization_msgs::MarkerArray loop_markers_msg = buildLoopMarkersMsg(snapshot, now, frame_id_);
    pub_loop_markers_.publish(loop_markers_msg);
    const visualization_msgs::MarkerArray keyframe_markers_msg =
        buildKeyframeMarkersMsg(snapshot.nodes, path_nodes, now, frame_id_);
    pub_keyframe_markers_.publish(keyframe_markers_msg);
    if (!use_optimized) {
      ROS_WARN_THROTTLE(5.0,
                        "[SemanticPgBackend] optimize tick fallback to snapshot path (GTSAM unavailable or optimize failed)");
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber sub_keyframe_;
  ros::Subscriber sub_loop_;
  ros::Subscriber sub_input_path_;
  ros::Publisher pub_keyframe_path_;
  ros::Publisher pub_optimized_path_;
  ros::Publisher pub_corrected_path_;
  ros::Publisher pub_loop_markers_;
  ros::Publisher pub_keyframe_markers_;
  ros::Timer timer_status_;
  ros::Timer timer_optimize_;

  std::string keyframe_topic_;
  std::string loop_topic_;
  double min_loop_score_ = 0.0;
  double min_geo_ratio_ = 0.0;
  double max_icp_fitness_ = 1e9;
  double max_loop_trans_m_ = 1e9;
  double max_pending_wait_s_ = 10.0;
  double status_period_s_ = 1.0;
  bool export_graph_csv_ = true;
  std::string export_dir_;
  std::string nodes_csv_name_;
  std::string edges_csv_name_;
  std::string corrected_path_tum_name_;
  std::string optimized_nodes_csv_name_;
  bool graph_dirty_ = false;
  bool status_dirty_ = false;
  size_t graph_revision_ = 0;
  bool enable_optimize_ = true;
  double odom_weight_scale_ = 0.5;
  double loop_weight_scale_ = 0.2;
  bool loop_robust_kernel_enable_ = true;
  double loop_robust_huber_k_ = 1.345;
  int optimize_every_n_keyframes_ = 10;
  double optimize_period_s_ = 2.0;
  int min_nodes_for_optimize_ = 5;
  double isam2_relinearize_threshold_ = 0.01;
  int isam2_relinearize_skip_ = 1;
  std::string frame_id_;
  std::string raw_path_topic_;
  std::string optimized_path_topic_;
  std::string input_path_topic_;
  std::string corrected_path_topic_;
  std::string loop_markers_topic_;
  std::string keyframe_markers_topic_;
  size_t last_optimized_keyframe_count_ = 0;
  size_t last_optimized_graph_revision_ = 0;
  ros::Time last_optimize_wall_;

#ifdef HAVE_GTSAM
  std::unique_ptr<gtsam::ISAM2> isam2_;
  std::set<uint32_t> isam2_seen_node_ids_;
  std::set<std::string> isam2_seen_edge_keys_;
  bool isam2_prior_added_ = false;
  bool isam2_has_result_ = false;
#endif

  mutable std::mutex mtx_;
  std::map<uint32_t, fast_livo::Keyframe> keyframes_;
  nav_msgs::Path latest_input_path_;
  std::deque<PendingLoop> pending_loops_;
  std::set<uint32_t> graph_nodes_;
  std::set<std::string> semantic_odom_edge_keys_;
  std::set<std::string> loop_edge_keys_;
  std::map<std::string, GraphEdgeRecord> semantic_odom_edges_map_;
  std::map<std::string, GraphEdgeRecord> loop_edges_map_;
  uint32_t last_backend_semantic_keyframe_id_ = 0U;
  bool has_last_backend_semantic_keyframe_ = false;

  size_t keyframes_received_ = 0;
  size_t loops_received_ = 0;
  size_t loops_valid_ = 0;
  size_t loops_rejected_ = 0;
  size_t semantic_odom_edges_ = 0;
  size_t loop_edges_ = 0;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "semantic_pg_backend");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  SemanticPgBackendNode node(nh, pnh);
  ros::spin();
  return 0;
}
