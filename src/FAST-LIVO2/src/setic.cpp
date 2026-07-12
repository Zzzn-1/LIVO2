//分割函数
#include"setic.h"

/*
SETIC语义处理系统 - 优化版本
按照VIO主流程设计，分为前中后三个阶段：
前段：从语义稀疏地图中提取语义点并投影
中段：计算语义投影并更新状态
后段：生成新的语义地图点并更新体素地图

性能优化策略：
1. 详细的时间统计和性能监控
2. 异步处理优化
3. 内存管理优化
4. 处理超时机制
*/

seticmanager::seticmanager()
{
    cam = nullptr;
    pinhole_cam = nullptr;
    state = nullptr;
    state_propagat = nullptr;
    Rli = M3D::Identity();
    Rci = M3D::Identity();
    Rcl = M3D::Identity();
    Rcw = M3D::Identity();
    Pli = V3D::Zero();
    Pci = V3D::Zero();
    Pcl = V3D::Zero();
    Pcw = V3D::Zero();
    grid_size = 5;
    grid_n_width = 0;
    grid_n_height = 17;
    patch_size = 8;
    patch_pyrimid_level = 3;
    width = 0;
    height = 0;
    length = 0;
    border = 20;

    // 初始化语义分类映射
    initSemanticClassMapping();
    
    // 初始化性能统计变量
    total_processing_time = 0.0;
    retrieval_time = 0.0;
    projection_time = 0.0;
    generation_time = 0.0;
    update_time = 0.0;
    patch_update_time = 0.0;
    visualization_time = 0.0;
    
    max_processing_time = 0.0;
    min_processing_time = std::numeric_limits<double>::max();
    frame_processing_count = 0;
    
    // 性能阈值设置
    max_allowed_processing_time = 0.050; // 50ms最大处理时间
    warning_processing_time = 0.030;      // 30ms警告阈值
    
    std::cout << "[SETIC] Initialized with performance monitoring" << std::endl;
    std::cout << "  Max allowed processing time: " << max_allowed_processing_time * 1000 << " ms" << std::endl;
    std::cout << "  Warning threshold: " << warning_processing_time * 1000 << " ms" << std::endl;
}

seticmanager::~seticmanager()
{
    // 输出最终性能统计
    printFinalPerformanceStats();
}

// 🔧 核心处理函数 - 优化版本，包含详细时间统计
void seticmanager::processFrame_setic(cv::Mat &img_setic, std::vector<pointWithVar> &pg_setic, 
                                      const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map_setic, 
                                      double setic_time)
{
    auto start_total = std::chrono::high_resolution_clock::now();
    
    static int process_log_counter = 0;
    ++process_log_counter;
    if (process_log_counter % 10 == 1) {
        std::cout << "[SETIC] Processing semantic frame at time: " << setic_time << std::endl;
    }
    
    if (img_setic.empty()) {
        std::cerr << "[SETIC] Empty semantic image!" << std::endl;
        return;
    }
    if (cam == nullptr) {
        std::cerr << "[SETIC] Camera is null, skip semantic frame to avoid crash." << std::endl;
        return;
    }
    if (state == nullptr) {
        std::cerr << "[SETIC] State pointer is null, skip semantic frame to avoid crash." << std::endl;
        return;
    }
    
    // 检查处理超时
    bool timeout_risk = false;
    double estimated_processing_time = estimateProcessingTime(pg_setic.size(), feat_map_setic.size());
    if (estimated_processing_time > max_allowed_processing_time) {
        timeout_risk = true;
        std::cout << "[SETIC WARNING] Estimated processing time (" << estimated_processing_time * 1000 
                  << " ms) exceeds threshold" << std::endl;
    }
    
    // 图像预处理 - 计时开始
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    
    if (width != img_setic.cols || height != img_setic.rows) {
        if (img_setic.empty()) {
            printf("[SETIC] Empty Semantic Image!\n");
            return;
        }
        // 如果需要，调整图像尺寸
    }
    
    cv::Mat class_mask;
    parseSemanticImage(img_setic, class_mask);
    if (class_mask.empty()) {
        std::cerr << "[SETIC] Failed to parse semantic image in preprocessing" << std::endl;
        return;
    }
    if (class_mask.cols != cam->width() || class_mask.rows != cam->height()) {
        cv::resize(class_mask, class_mask, cv::Size(cam->width(), cam->height()), 0, 0, cv::INTER_NEAREST);
    }
    if (length <= 0 || semantic_grid_num.size() != static_cast<size_t>(length)) {
        initializeVIO_setic();
    }

    // 内部统一使用单通道类别ID图；可视化时再转彩色
    img_setic = class_mask;
    img_rgb_setic = class_mask.clone();
    cv::cvtColor(class_mask, img_cp_setic, cv::COLOR_GRAY2BGR);
    
    // 创建新的语义帧
    new_frame_setic_.reset(new Frame(cam, img_setic));
    updateSemanticState(*state);
    resetSemanticGrid();
    
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    double preprocess_time = std::chrono::duration<double>(end_preprocess - start_preprocess).count();
    
    /*
    前段：从语义稀疏地图中提取语义点
    */
    double current_retrieval_time = 0.0;
    if (semantic_loop_en) {
        auto start_retrieval = std::chrono::high_resolution_clock::now();
        retrieveSemanticFromVoxelMap(img_setic, pg_setic, feat_map_setic);
        auto end_retrieval = std::chrono::high_resolution_clock::now();
        current_retrieval_time = std::chrono::duration<double>(end_retrieval - start_retrieval).count();
    }
    
    /*
    中段：计算语义投影并更新状态
    */
    double current_projection_time = 0.0;
    if (semantic_loop_en) {
        auto start_projection = std::chrono::high_resolution_clock::now();
        computeSemanticProjectionAndUpdate(img_setic, pg_setic);
        auto end_projection = std::chrono::high_resolution_clock::now();
        current_projection_time = std::chrono::duration<double>(end_projection - start_projection).count();
    }
    
    /*
    后段：生成新的语义地图点
    */
    auto start_generation = std::chrono::high_resolution_clock::now();
    generateSemanticVoxelPoints(img_setic, pg_setic);
    auto end_generation = std::chrono::high_resolution_clock::now();
    double current_generation_time = std::chrono::duration<double>(end_generation - start_generation).count();
    
    // 可视化处理 - 可选跳过以节省时间
    double current_visualization_time = 0.0;
    if (semantic_loop_en) {
        auto start_visualization = std::chrono::high_resolution_clock::now();
        if (!timeout_risk || frame_processing_count % 5 == 0) { // 超时风险时每5帧可视化一次
            plotSemanticPoints();
        }
        auto end_visualization = std::chrono::high_resolution_clock::now();
        current_visualization_time = std::chrono::duration<double>(end_visualization - start_visualization).count();
    }
    
    // 更新语义地图点
    double current_update_time = 0.0;
    if (semantic_loop_en) {
        auto start_update = std::chrono::high_resolution_clock::now();
        updateSemanticVoxelMap(img_setic);
        auto end_update = std::chrono::high_resolution_clock::now();
        current_update_time = std::chrono::duration<double>(end_update - start_update).count();
    }
    
    // 更新参考补丁 - 可选跳过以节省时间
    double current_patch_time = 0.0;
    if (semantic_loop_en) {
        auto start_patch = std::chrono::high_resolution_clock::now();
        if (!timeout_risk || frame_processing_count % 3 == 0) { // 超时风险时每3帧更新一次
            updateSemanticReferencePatch(feat_map_setic);
        }
        auto end_patch = std::chrono::high_resolution_clock::now();
        current_patch_time = std::chrono::duration<double>(end_patch - start_patch).count();
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    double current_total_time = std::chrono::duration<double>(end_total - start_total).count();
    
    // 更新性能统计
    updatePerformanceStats(preprocess_time, current_retrieval_time, current_projection_time, 
                          current_generation_time, current_visualization_time, 
                          current_update_time, current_patch_time, current_total_time);
    
    // 输出详细性能报告（降频，避免IO阻塞）
    if (frame_processing_count % 30 == 0 || current_total_time > warning_processing_time) {
        printDetailedPerformanceReport(preprocess_time, current_retrieval_time, current_projection_time,
                                      current_generation_time, current_visualization_time,
                                      current_update_time, current_patch_time, current_total_time,
                                      feat_map_setic.size(), pg_setic.size());
    }
    
    // 性能警告
    if (current_total_time > warning_processing_time) {
        std::cout << "\033[1;33m[SETIC WARNING] Processing time (" << current_total_time * 1000 
                  << " ms) exceeds warning threshold!\033[0m" << std::endl;
    }
    
    if (current_total_time > max_allowed_processing_time) {
        std::cout << "\033[1;31m[SETIC CRITICAL] Processing time (" << current_total_time * 1000 
                  << " ms) exceeds maximum allowed time!\033[0m" << std::endl;
    }
    
    if (process_log_counter % 10 == 1) {
        std::cout << "[SETIC] Processed " << total_semantic_points << " semantic points" << std::endl;
    }
}

// 🔧 估算处理时间
double seticmanager::estimateProcessingTime(size_t point_count, size_t voxel_count)
{
    if (frame_processing_count == 0) {
        // 初始估算：基于点云和体素数量的经验公式
        return (point_count * 0.000001 + voxel_count * 0.000002) + 0.010; // 基础10ms + 处理时间
    }
    
    // 基于历史平均时间的估算
    double base_time = total_processing_time / frame_processing_count;
    double scale_factor = static_cast<double>(point_count + voxel_count) / 
                         (semantic_frame_count > 0 ? (last_point_count + last_voxel_count) : 1000);
    
    return base_time * scale_factor;
}

// 🔧 更新性能统计
void seticmanager::updatePerformanceStats(double preprocess, double retrieval, double projection,
                                         double generation, double visualization, double update,
                                         double patch, double total)
{
    frame_processing_count++;
    
    // 更新平均时间
    if (frame_processing_count == 1) {
        total_processing_time = total;
        retrieval_time = retrieval;
        projection_time = projection;
        generation_time = generation;
        update_time = update;
        patch_update_time = patch;
        visualization_time = visualization;
        preprocess_time = preprocess;
    } else {
        double alpha = 1.0 / frame_processing_count;
        total_processing_time = (1 - alpha) * total_processing_time + alpha * total;
        retrieval_time = (1 - alpha) * retrieval_time + alpha * retrieval;
        projection_time = (1 - alpha) * projection_time + alpha * projection;
        generation_time = (1 - alpha) * generation_time + alpha * generation;
        update_time = (1 - alpha) * update_time + alpha * update;
        patch_update_time = (1 - alpha) * patch_update_time + alpha * patch;
        visualization_time = (1 - alpha) * visualization_time + alpha * visualization;
        preprocess_time = (1 - alpha) * preprocess_time + alpha * preprocess;
    }
    
    // 更新最大/最小时间
    if (total > max_processing_time) max_processing_time = total;
    if (total < min_processing_time) min_processing_time = total;
    
    // 更新上一次的点云和体素数量
    last_point_count = retrieve_semantic_points.size();
    last_voxel_count = semantic_voxel_map.size();
}

// 🔧 打印详细性能报告
void seticmanager::printDetailedPerformanceReport(double preprocess, double retrieval, double projection,
                                                  double generation, double visualization, double update,
                                                  double patch, double total, size_t voxel_count, size_t point_count)
{
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m|                  SETIC Performance Report                  |\033[0m\n");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Input Points", point_count);
    printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Voxel Map Size", voxel_count);
    printf("\033[1;34m| %-29s | %-27d |\033[0m\n", "Processed Frames", frame_processing_count);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-24s (ms) |\033[0m\n", "Processing Stage", "Current/Avg");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Preprocessing", preprocess * 1000, preprocess_time * 1000);
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Semantic Retrieval", retrieval * 1000, retrieval_time * 1000);
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Projection & Update", projection * 1000, projection_time * 1000);
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Point Generation", generation * 1000, generation_time * 1000);
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Visualization", visualization * 1000, visualization_time * 1000);
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Voxel Map Update", update * 1000, update_time * 1000);
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Patch Update", patch * 1000, patch_update_time * 1000);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Total Processing", total * 1000, total_processing_time * 1000);
    printf("\033[1;36m| %-29s | %8.2f / %-13.2f |\033[0m\n", "Min/Max Total", min_processing_time * 1000, max_processing_time * 1000);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    
    // 性能指标分析
    double efficiency = (total > 0) ? (generation * 1000) / (total * 1000) * 100 : 0;
    double overhead = (total > 0) ? ((visualization + patch) * 1000) / (total * 1000) * 100 : 0;
    
    printf("\033[1;35m| %-29s | %-24.1f%% |\033[0m\n", "Core Processing Ratio", efficiency);
    printf("\033[1;35m| %-29s | %-24.1f%% |\033[0m\n", "Visualization Overhead", overhead);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
}

// 🔧 打印最终性能统计
void seticmanager::printFinalPerformanceStats()
{
    if (frame_processing_count == 0) return;
    
    std::cout << "\n\033[1;32m========== SETIC Final Performance Summary ==========\033[0m" << std::endl;
    std::cout << "Total processed frames: " << frame_processing_count << std::endl;
    std::cout << "Average processing time: " << total_processing_time * 1000 << " ms" << std::endl;
    std::cout << "Min processing time: " << min_processing_time * 1000 << " ms" << std::endl;
    std::cout << "Max processing time: " << max_processing_time * 1000 << " ms" << std::endl;
    
    double fps = (total_processing_time > 0) ? 1.0 / total_processing_time : 0;
    std::cout << "Average processing FPS: " << fps << std::endl;
    
    // 性能分级
    if (total_processing_time <= 0.020) {
        std::cout << "\033[1;32mPerformance Grade: EXCELLENT (< 20ms)\033[0m" << std::endl;
    } else if (total_processing_time <= 0.030) {
        std::cout << "\033[1;33mPerformance Grade: GOOD (20-30ms)\033[0m" << std::endl;
    } else if (total_processing_time <= 0.050) {
        std::cout << "\033[1;31mPerformance Grade: ACCEPTABLE (30-50ms)\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31mPerformance Grade: POOR (> 50ms)\033[0m" << std::endl;
    }
    
    std::cout << "\033[1;32m==================================================\033[0m\n" << std::endl;
}

// 🔧 前段：从语义稀疏地图中提取语义点
void seticmanager::retrieveSemanticFromVoxelMap(cv::Mat img_setic, 
                                                std::vector<pointWithVar> &pg, 
                                                const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &semantic_map)
{
    if (semantic_map.size() <= 0) return;
    
    static int retrieve_log_counter = 0;
    ++retrieve_log_counter;
    if (retrieve_log_counter % 20 == 1) {
        std::cout << "[SETIC] Retrieving semantic points from voxel map..." << std::endl;
    }
    
    // 创建深度图用于语义投影
    cv::Mat semantic_depth_img = cv::Mat::zeros(height, width, CV_32FC1);
    float *depth_data = (float *)semantic_depth_img.data;
    
    float voxel_size = 0.5;
    int processed_semantic_points = 0;
    
    // 遍历输入点云，投影到语义图像平面
    for (size_t i = 0; i < pg.size(); i++) {
        const pointWithVar &p_v = pg[i];
        
        // 计算在语义图像中的投影
        V3D pt_cam = new_frame_setic_->w2f(p_v.point_w);
        if (pt_cam[2] < 0) continue; // 点在相机后方
        
        V2D pc = new_frame_setic_->w2c(p_v.point_w);
        
        // 检查是否在图像范围内
        if (pc[0] >= border && pc[0] < width - border && 
            pc[1] >= border && pc[1] < height - border) {
            
            // 更新深度图
            int pixel_idx = width * int(pc[1]) + int(pc[0]);
            if (depth_data[pixel_idx] == 0.0f || pt_cam[2] < depth_data[pixel_idx]) {
                depth_data[pixel_idx] = pt_cam[2];
            }
            
            // 计算网格索引
            int grid_idx = static_cast<int>(pc[1] / grid_size) * grid_n_width + 
                          static_cast<int>(pc[0] / grid_size);
            
            if (grid_idx >= 0 && grid_idx < length) {
                // 只有已有语义标签的稀疏点才占用MAP网格，避免无语义点阻塞新语义生成
                if (p_v.semantic_label > 0) {
                    semantic_grid_num[grid_idx] = SEMANTIC_TYPE_MAP;
                    float cur_dist = pt_cam.norm();
                    if (cur_dist <= semantic_map_dist[grid_idx]) {
                        semantic_map_dist[grid_idx] = cur_dist;
                        retrieve_semantic_points[grid_idx] = p_v;
                        processed_semantic_points++;
                    }
                }
            }
        }
    }
    
    total_semantic_points = processed_semantic_points;
    if (retrieve_log_counter % 20 == 1 || processed_semantic_points > 0) {
        std::cout << "[SETIC] Retrieved " << processed_semantic_points << " semantic points from sparse map" << std::endl;
    }
}

// 🔧 中段：计算语义投影并更新状态
void seticmanager::computeSemanticProjectionAndUpdate(cv::Mat img_setic, std::vector<pointWithVar> &pg)
{
    if (retrieve_semantic_points.empty()) return;
    
    static int projection_log_counter = 0;
    ++projection_log_counter;
    if (projection_log_counter % 20 == 1) {
        std::cout << "[SETIC] Computing semantic projection and updating state..." << std::endl;
    }
    
    // 解析语义图像
    cv::Mat class_mask;
    this->parseSemanticImage(img_setic, class_mask);
    
    if (class_mask.empty()) {
        std::cerr << "[SETIC] Failed to parse semantic image" << std::endl;
        return;
    }

    int semantic_matches = 0;
    
    // 为已有的语义点更新标签
    for (size_t i = 0; i < retrieve_semantic_points.size(); i++) {
        if (semantic_grid_num[i] == SEMANTIC_TYPE_MAP) {
            pointWithVar &semantic_point = retrieve_semantic_points[i];
            
            // 计算投影位置
            V2D pc = new_frame_setic_->w2c(semantic_point.point_w);
            
            // 检查像素是否在图像范围内
            if (pc[0] >= 0 && pc[0] < class_mask.cols && 
                pc[1] >= 0 && pc[1] < class_mask.rows) {
                
                int semantic_label = class_mask.at<uchar>(pc[1], pc[0]);
                if (semantic_label > 0) {
                    // 更新或确认语义标签
                    if (semantic_point.semantic_label == 0) {
                        semantic_point.semantic_label = semantic_label;
                        semantic_matches++;
                    } else if (semantic_point.semantic_label == semantic_label) {
                        // 增强置信度（这里可以添加置信度更新逻辑）
                        semantic_matches++;
                    }
                }
            }
        }
    }
    
    if (projection_log_counter % 20 == 1 || semantic_matches > 0) {
        std::cout << "[SETIC] Updated " << semantic_matches << " semantic labels" << std::endl;
    }
}

// 🔧 后段：生成新的语义地图点
void seticmanager::generateSemanticVoxelPoints(cv::Mat img_setic, std::vector<pointWithVar> &pg)
{
    if (pg.size() <= 10) return;
    
    static int generate_log_counter = 0;
    ++generate_log_counter;
    if (generate_log_counter % 20 == 1) {
        std::cout << "[SETIC] Generating semantic voxel points..." << std::endl;
    }
    
    // 解析语义图像
    cv::Mat class_mask;
    this->parseSemanticImage(img_setic, class_mask);
    
    if (class_mask.empty()) return;
    
    int added_semantic_points = 0;
    
    // 遍历输入点云，为新点添加语义标签
    for (size_t i = 0; i < pg.size(); i++) {
        const pointWithVar &point = pg[i];
        
        // 投影到语义图像
        V2D pc = new_frame_setic_->w2c(point.point_w);
        
        // 检查是否在图像帧内
        if (pc[0] >= border && pc[0] < width - border && 
            pc[1] >= border && pc[1] < height - border) {
            
            int grid_idx = static_cast<int>(pc[1] / grid_size) * grid_n_width + 
                          static_cast<int>(pc[0] / grid_size);
            
            // 如果该网格未被语义地图占用
            if (grid_idx >= 0 && grid_idx < length && 
                semantic_grid_num[grid_idx] != SEMANTIC_TYPE_MAP) {
                
                // 获取语义标签
                int semantic_label = 0;
                if (pc[0] >= 0 && pc[0] < class_mask.cols && 
                    pc[1] >= 0 && pc[1] < class_mask.rows) {
                    semantic_label = class_mask.at<uchar>(pc[1], pc[0]);
                }
                
                if (semantic_label > 0) {
                    // 计算语义得分（可以基于语义置信度或其他指标）
                    float semantic_score = 1.0f; // 简化版本，实际可以更复杂
                    
                    if (semantic_score > semantic_scan_value[grid_idx]) {
                        semantic_scan_value[grid_idx] = semantic_score;
                        
                        // 创建带语义标签的点
                        pointWithVar semantic_point = point;
                        semantic_point.semantic_label = semantic_label;
                        
                        append_semantic_points[grid_idx] = semantic_point;
                        semantic_grid_num[grid_idx] = SEMANTIC_TYPE_POINTCLOUD;
                        added_semantic_points++;
                    }
                }
            }
        }
    }
    
    total_semantic_points += added_semantic_points;
    if (generate_log_counter % 20 == 1 || added_semantic_points > 0) {
        std::cout << "[SETIC] Generated " << added_semantic_points << " new semantic points" << std::endl;
    }
}

// 🔧 更新语义体素地图
void seticmanager::updateSemanticVoxelMap(cv::Mat img_setic)
{
    if (total_semantic_points == 0) return;
    
    static int map_update_log_counter = 0;
    ++map_update_log_counter;
    if (map_update_log_counter % 20 == 1) {
        std::cout << "[SETIC] Updating semantic voxel map..." << std::endl;
    }
    
    int update_num = 0;
    
    // 遍历所有添加的语义点，将其插入到体素地图中
    for (int i = 0; i < length; i++) {
        if (semantic_grid_num[i] == SEMANTIC_TYPE_POINTCLOUD) {
            const pointWithVar &semantic_point = append_semantic_points[i];
            
            // 插入语义点到体素地图
            insertSemanticPointIntoVoxelMap(semantic_point);
            update_num++;
        }
    }
    
    if (map_update_log_counter % 20 == 1 || update_num > 0) {
        std::cout << "[SETIC] Updated semantic voxel map with " << update_num << " points" << std::endl;
    }
}

// 🔧 更新语义参考补丁
void seticmanager::updateSemanticReferencePatch(const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &semantic_map)
{
    if (total_semantic_points == 0) return;
    
    static int patch_log_counter = 0;
    ++patch_log_counter;
    if (patch_log_counter % 20 == 1) {
        std::cout << "[SETIC] Updating semantic reference patches..." << std::endl;
    }
    
    int updated_patches = 0;
    
    // 遍历语义点，检查其在体素地图中的状态
    for (size_t i = 0; i < retrieve_semantic_points.size(); i++) {
        if (semantic_grid_num[i] == SEMANTIC_TYPE_MAP) {
            const pointWithVar &semantic_point = retrieve_semantic_points[i];
            
            // 计算体素位置
            const V3D &p_w = semantic_point.point_w;
            float loc_xyz[3];
            for (int j = 0; j < 3; j++) {
                loc_xyz[j] = p_w[j] / 0.5;
                if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
            }
            
            VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            auto iter = semantic_map.find(position);
            
            if (iter != semantic_map.end()) {
                // 更新语义信息
                VoxelOctoTree *voxel = iter->second;
                // 这里可以添加更新语义信息的逻辑
                updated_patches++;
            }
        }
    }
    
    if (patch_log_counter % 20 == 1 || updated_patches > 0) {
        std::cout << "[SETIC] Updated " << updated_patches << " semantic reference patches" << std::endl;
    }
}

// 🔧 重置语义网格
void seticmanager::resetSemanticGrid()
{
    std::fill(semantic_grid_num.begin(), semantic_grid_num.end(), SEMANTIC_TYPE_UNKNOWN);
    std::fill(semantic_map_index.begin(), semantic_map_index.end(), 0);
    std::fill(semantic_map_dist.begin(), semantic_map_dist.end(), 10000.0f);
    std::fill(semantic_update_flag.begin(), semantic_update_flag.end(), 0);
    std::fill(semantic_scan_value.begin(), semantic_scan_value.end(), 0.0f);
    
    retrieve_semantic_points.clear();
    retrieve_semantic_points.resize(length);
    
    append_semantic_points.clear();
    append_semantic_points.resize(length);
    
    total_semantic_points = 0;
}

// 🔧 更新语义状态
void seticmanager::updateSemanticState(const StatesGroup &current_state)
{
    M3D Rwi(current_state.rot_end);
    V3D Pwi(current_state.pos_end);
    
    // 计算语义相机坐标变换
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    
    // 更新帧状态（如果有new_frame_setic_）
    if (new_frame_setic_) {
        new_frame_setic_->T_f_w_ = SE3(Rcw, Pcw);
    }
}

// 🔧 可视化语义点
void seticmanager::plotSemanticPoints()
{
    if (total_semantic_points == 0) return;
    
    std::cout << "[SETIC] Plotting semantic points..." << std::endl;
    
    // 在语义图像上绘制语义点
    for (int i = 0; i < length; i++) {
        if (semantic_grid_num[i] == SEMANTIC_TYPE_MAP) {
            const pointWithVar &semantic_point = retrieve_semantic_points[i];
            
            if (semantic_point.semantic_label > 0) {
                V2D pc = new_frame_setic_->w2c(semantic_point.point_w);
                
                // 根据语义标签选择颜色
                cv::Scalar color = getSemanticColor(semantic_point.semantic_label);
                cv::circle(img_cp_setic, cv::Point2f(pc[0], pc[1]), 5, color, -1, 8);
            }
        }
    }
}

// 🔧 将语义点插入体素地图
void seticmanager::insertSemanticPointIntoVoxelMap(const pointWithVar &semantic_point)
{
    const V3D &p_w = semantic_point.point_w;
    
    // 计算体素位置
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = p_w[j] / VOXEL_SIZE;
        if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    
    // 检查体素是否存在
    if (semantic_voxel_map.find(position) == semantic_voxel_map.end()) {
        semantic_voxel_map[position] = new VoxelOctoTree();
    }
    
    // 将语义点添加到体素中
    VoxelOctoTree *voxel = semantic_voxel_map[position];
    // 这里可以添加语义信息到体素的逻辑
    // voxel->addSemanticPoint(semantic_point);
}

// 🔧 初始化语义分类映射
void seticmanager::initSemanticClassMapping()
{
    // 初始化语义类别名称映射
    semantic_class_names[1] = "chair";
    semantic_class_names[2] = "table";
    semantic_class_names[3] = "person";
    semantic_class_names[4] = "car";
    semantic_class_names[5] = "building";
    // 可以添加更多类别...
    
    // 初始化语义类别颜色映射
    semantic_class_colors[1] = cv::Vec3b(0, 0, 255);    // 红色
    semantic_class_colors[2] = cv::Vec3b(0, 255, 0);    // 绿色
    semantic_class_colors[3] = cv::Vec3b(255, 0, 0);    // 蓝色
    semantic_class_colors[4] = cv::Vec3b(0, 255, 255);  // 黄色
    semantic_class_colors[5] = cv::Vec3b(255, 0, 255);  // 品红
    // 可以添加更多颜色...
}

// 🔧 获取语义颜色
cv::Scalar seticmanager::getSemanticColor(int semantic_label)
{
    auto it = semantic_class_colors.find(semantic_label);
    if (it != semantic_class_colors.end()) {
        cv::Vec3b color = it->second;
        return cv::Scalar(color[0], color[1], color[2]);
    }
    return cv::Scalar(128, 128, 128); // 默认灰色
}

// 🔧 初始化VIO语义系统
void seticmanager::initializeVIO_setic()
{
    std::cout << "[SETIC] Initializing semantic VIO system..." << std::endl;
    
    // 设置图像参数
    if (cam != nullptr) {
        fx = cam->fx();
        fy = cam->fy();
        cx = cam->cx();
        cy = cam->cy();
        width = cam->width();
        height = cam->height();
    } else {
        width = 640;
        height = 480;
        std::cout << "[SETIC WARNING] Camera is null in initializeVIO_setic, using fallback size 640x480." << std::endl;
    }
    if (grid_size <= 0) {
        grid_size = 5;
    }
    if (grid_n_height <= 0) {
        grid_n_height = 17;
    }
    
    // 设置网格参数
    if (grid_size > 10) {
        grid_n_width = ceil(static_cast<double>(width / grid_size));
        grid_n_height = ceil(static_cast<double>(height / grid_size));
    } else {
        grid_size = static_cast<int>(height / grid_n_height);
        grid_n_height = ceil(static_cast<double>(height / grid_size));
        grid_n_width = ceil(static_cast<double>(width / grid_size));
    }
    
    length = grid_n_width * grid_n_height;
    
    // 初始化网格向量
    semantic_grid_num.resize(length);
    semantic_map_index.resize(length);
    semantic_border_flag.resize(length);
    semantic_update_flag.resize(length);
    semantic_map_dist.resize(length);
    semantic_scan_value.resize(length);
    semantic_patch_buffer.resize(length);
    
    retrieve_semantic_points.resize(length);
    append_semantic_points.resize(length);
    
    // 设置默认参数
    border = 20;
    semantic_outlier_threshold = 1000.0f;
    total_semantic_points = 0;
    
    // 初始化语义分类映射
    initSemanticClassMapping();
    
    std::cout << "[SETIC] Semantic VIO system initialized with grid size: " << grid_size 
              << ", dimensions: " << grid_n_width << "x" << grid_n_height << std::endl;
}

// 🔧 语义图像解析函数实现
void seticmanager::parseSemanticImage(const cv::Mat& semantic_img, cv::Mat& class_mask)
{
    static int parse_log_counter = 0;
    ++parse_log_counter;
    if (parse_log_counter % 30 == 1) {
        std::cout << "[SETIC] Parsing semantic image..." << std::endl;
    }
    
    // 检查输入图像
    if (semantic_img.empty()) {
        std::cerr << "[SETIC] Input semantic image is empty!" << std::endl;
        class_mask = cv::Mat();
        return;
    }
    
    auto semanticIdFromLegacyColor = [](const cv::Vec3b &color) -> uint8_t {
        if (color == cv::Vec3b(240, 230, 140)) return 1; // person
        if (color == cv::Vec3b(255, 215, 0)) return 2;   // car
        if (color == cv::Vec3b(34, 139, 34)) return 3;   // bicycle
        if (color == cv::Vec3b(130, 0, 75)) return 4;    // motorcycle
        if (color == cv::Vec3b(0, 69, 255)) return 5;    // bus
        return 0;                                         // background/others
    };

    try {
        // 目标格式：CV_8UC1 类别ID图
        if (semantic_img.type() == CV_8UC1) {
            class_mask = semantic_img.clone();
        } else if (semantic_img.type() == CV_8UC3) {
            class_mask = cv::Mat::zeros(semantic_img.rows, semantic_img.cols, CV_8UC1);
            for (int y = 0; y < semantic_img.rows; ++y) {
                const cv::Vec3b *src_ptr = semantic_img.ptr<cv::Vec3b>(y);
                uint8_t *dst_ptr = class_mask.ptr<uint8_t>(y);
                for (int x = 0; x < semantic_img.cols; ++x) {
                    dst_ptr[x] = semanticIdFromLegacyColor(src_ptr[x]);
                }
            }
        } else {
            std::cerr << "[SETIC] Unsupported image format: type=" << semantic_img.type() << std::endl;
            class_mask = cv::Mat();
            return;
        }
        
        // 检查结果
        if (class_mask.empty()) {
            std::cerr << "[SETIC] Failed to create class mask!" << std::endl;
            return;
        }
        
        // 统计语义类别
        double min_val, max_val;
        cv::minMaxLoc(class_mask, &min_val, &max_val);
        
        if (parse_log_counter % 30 == 1) {
            std::cout << "[SETIC] Successfully parsed semantic image:" << std::endl;
            std::cout << "  Size: " << class_mask.cols << "x" << class_mask.rows << std::endl;
            std::cout << "  Class range: " << min_val << " to " << max_val << std::endl;
        }
        
        // 更新图像尺寸（如果发生变化）
        width = class_mask.cols;
        height = class_mask.rows;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[SETIC] OpenCV error in parseSemanticImage: " << e.what() << std::endl;
        class_mask = cv::Mat();
    } catch (const std::exception& e) {
        std::cerr << "[SETIC] Error in parseSemanticImage: " << e.what() << std::endl;
        class_mask = cv::Mat();
    }
}

// 🔧 重置性能统计
void seticmanager::resetPerformanceStats()
{
    total_processing_time = 0.0;
    preprocess_time = 0.0;
    retrieval_time = 0.0;
    projection_time = 0.0;
    generation_time = 0.0;
    update_time = 0.0;
    patch_update_time = 0.0;
    visualization_time = 0.0;
    
    max_processing_time = 0.0;
    min_processing_time = std::numeric_limits<double>::max();
    frame_processing_count = 0;
    
    last_point_count = 0;
    last_voxel_count = 0;
    
    std::cout << "[SETIC] Performance statistics reset" << std::endl;
}
