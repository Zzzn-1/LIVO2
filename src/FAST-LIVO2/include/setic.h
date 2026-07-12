/*

*/
#ifndef SETIC_H_
#define SETIC_H_
#define VOXEL_SIZE 0.5 // 体素大小，单位为米

#include "voxel_map.h"

#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>
#include <unordered_map>
#include <vector>
#include "common_lib.h" // 确保只在 common_lib.h 中定义 pointWithVar
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <tuple>
#include <chrono>
#include <limits>

class VIOManager; 
class seticmanager; // 前向声明

// 自定义哈希函数，用于 unordered_map 中的 Vec3b 类型
struct Vec3bHash {
    std::size_t operator()(const cv::Vec3b &color) const {
        return std::hash<int>()(color[0]) ^ std::hash<int>()(color[1]) ^ std::hash<int>()(color[2]);
    }
};

// extern std::unordered_map<cv::Vec3b, std::string, Vec3bHash> color_to_class;

extern std::unordered_map<cv::Vec3b, uchar, Vec3bHash> color_to_class;

// 用于颜色比较的结构体
struct Vec3bCompare {
    bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const {
        return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
    }
};

class seticmanager
{
public:
    seticmanager();
    ~seticmanager();

    // 🔧 按照VIO主流程重新设计的核心函数
    void processFrame_setic(cv::Mat &img_setic, std::vector<pointWithVar> &pg_setic, const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map_setic, double setic_time);
    
    // VIO风格的主要处理步骤
    void retrieveSemanticFromVoxelMap(cv::Mat img_setic, std::vector<pointWithVar> &pg, const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &semantic_map);
    void computeSemanticProjectionAndUpdate(cv::Mat img_setic, std::vector<pointWithVar> &pg);
    void generateSemanticVoxelPoints(cv::Mat img_setic, std::vector<pointWithVar> &pg);
    void updateSemanticVoxelMap(cv::Mat img_setic);
    void updateSemanticReferencePatch(const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &semantic_map);
    
    // 语义处理辅助函数
    void parseSemanticImage(const cv::Mat &semantic_img, cv::Mat &class_mask);
    void extractInstanceInfo(const cv::Mat &class_mask, std::unordered_map<int, std::vector<cv::Point>> &class_centers);
    void resetSemanticGrid();
    void updateSemanticState(const StatesGroup &state);
    void plotSemanticPoints();
    
    // 体素地图相关函数
    void insertSemanticPointIntoVoxelMap(const pointWithVar &semantic_point);
    void associateSemanticWithPointCloud(const cv::Mat &class_mask, std::vector<pointWithVar> &pg, const cv::Mat &camera_intrinsics);
    void integratePointIntoVoxelMap(const cv::Vec3f &point_3d, int semantic_label, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map);
    void fuseImageWithVoxelMap(const cv::Mat &class_mask, const cv::Mat &depth_map, const cv::Mat &camera_intrinsics, const cv::Mat &camera_extrinsics, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map);

    // 🔧 新增：性能统计和监控函数
    double estimateProcessingTime(size_t point_count, size_t voxel_count);
    void updatePerformanceStats(double preprocess, double retrieval, double projection,
                               double generation, double visualization, double update,
                               double patch, double total);
    void printDetailedPerformanceReport(double preprocess, double retrieval, double projection,
                                       double generation, double visualization, double update,
                                       double patch, double total, size_t voxel_count, size_t point_count);
    void printFinalPerformanceStats();
    void resetPerformanceStats();
    double getCurrentProcessingTime() const { return total_processing_time; }
    bool isPerformanceAcceptable() const { return total_processing_time <= max_allowed_processing_time; }

    // 成员变量 - 按VIO风格组织
    cv::Mat img_rgb_setic;
    cv::Mat img_cp_setic;
    cv::Mat img_setic;

    // 状态管理
    vk::AbstractCamera *cam;
    vk::PinholeCamera *pinhole_cam;
    StatesGroup *state;
    StatesGroup *state_propagat;
    M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
    V3D Pli, Pci, Pcl, Pcw;

    // 网格管理 - 参考VIO设计
    std::vector<int> semantic_grid_num;
    std::vector<int> semantic_map_index;
    std::vector<int> semantic_border_flag;
    std::vector<int> semantic_update_flag;
    std::vector<float> semantic_map_dist;
    std::vector<float> semantic_scan_value;
    std::vector<float> semantic_patch_buffer;
    
    // 语义特征点管理
    std::vector<pointWithVar> retrieve_semantic_points;
    std::vector<pointWithVar> append_semantic_points;
    
    // 🔧 新增：详细的性能统计变量
    double total_processing_time;
    double preprocess_time;
    double retrieval_time;
    double projection_time;
    double generation_time;
    double update_time;
    double patch_update_time;
    double visualization_time;
    
    double max_processing_time;
    double min_processing_time;
    int frame_processing_count;
    
    // 性能阈值
    double max_allowed_processing_time;
    double warning_processing_time;
    
    // 历史数据用于估算
    size_t last_point_count;
    size_t last_voxel_count;
    
    // 时间统计（保留原有变量以兼容）
    double semantic_total_time = 0.0;
    double semantic_projection_time = 0.0;
    double semantic_update_time = 0.0;
    int semantic_frame_count = 0;
    
    // 算法参数
    int max_iterations, total_semantic_points;
    int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
    int width, height, length;
    float semantic_outlier_threshold;
    bool semantic_loop_en = false;
    
    // 相机参数
    double fx, fy, cx, cy;
    int border;
    
    // 体素地图管理
    std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> semantic_voxel_map;
    std::unordered_map<VOXEL_LOCATION, int> sub_semantic_map;
    
    // 语义分类管理
    std::map<int, std::string> semantic_class_names;
    std::map<int, cv::Vec3b> semantic_class_colors;

    FramePtr new_frame_setic_;
    typedef std::shared_ptr<seticmanager> seticmanagerPtr;
    seticmanagerPtr setic_manager_ptr_;

    // 初始化和配置
    void initializeVIO_setic();
    void resetGrid_setic();
    void updateFrameState(StatesGroup state);
    void set_max_edge();
    void center();
    
    // 语义类别枚举
    enum SemanticCellType
    {
        SEMANTIC_TYPE_MAP = 1,
        SEMANTIC_TYPE_POINTCLOUD,
        SEMANTIC_TYPE_UNKNOWN
    };

    // 辅助函数声明
    void initSemanticClassMapping();
    cv::Scalar getSemanticColor(int semantic_label);

};

// 🔧 移除重复的函数声明，避免链接冲突
// cv::Point2f projectPointToImage(const cv::Vec3f &point, const cv::Mat &camera_intrinsics);

#endif
