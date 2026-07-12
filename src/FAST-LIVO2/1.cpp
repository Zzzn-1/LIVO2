void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");//LiDAR 数据的 ROS 话题名称，默认值为 /livox/lidar
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");//IMU 数据的 ROS 话题名称，默认值为 /livox/imu
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);//是否启用 LiDAR，默认值为 1（启用）
  nh.param<int>("common/lidar_en", lidar_en, 1);

  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");//图像数据的 ROS 话题名称，默认值为 /left_camera/image

  nh.param<string>("common/setic_topic", setic_topic, "/setic/image_raw");//语义部分订阅话题
  nh.param<int>("common/setic_en", setic_en, 1);

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



void LIVMapper::setic_cbk(const sensor_msgs::ImageConstPtr &msg_in_setic)
{
  std::cout << "setic_cbk called - synchronous mode" << std::endl;
  std::cout << "Received image encoding: " << msg_in_setic->encoding << std::endl;
  std::cout << "Image size: " << msg_in_setic->width << "x" << msg_in_setic->height << std::endl;

  if (!setic_en) return; // 是否使用语义部分
  
  ROS_INFO("Get image_setic");
  sensor_msgs::Image::Ptr msg_setic(new sensor_msgs::Image(*msg_in_setic));

  double msg_header_time_setic = msg_setic->header.stamp.toSec() + img_time_offset_setic;
  ROS_INFO("Get image_setic, its header time: %.6f", msg_header_time_setic);

  // 关键修复：时间戳顺序检查
  if (msg_header_time_setic < last_timestamp_img_setic)
  {
    ROS_ERROR("SETIC timestamp regression detected: %.6f -> %.6f", 
              last_timestamp_img_setic, msg_header_time_setic);
    return;
  }
  
  // 关键修复：同步模式 - 简化缓冲区管理
  mtx_buffer.lock();
  
  // 控制缓冲区大小，但不过度丢帧
  const int max_setic_buffer_size = 5; // 增加缓冲区大小确保顺序处理
  while (img_buffer_setic.size() >= max_setic_buffer_size) {
    img_buffer_setic.pop_front();
    img_time_buffer_setic.pop_front();
    std::cout << "[SETIC] Buffer full, removing oldest frame" << std::endl;
  }
  
  double img_time_correct_setic = msg_header_time_setic;
  cv::Mat img_cur_setic = getImageFromMsg_setic(msg_setic);
  
  // 关键修复：确保按时间戳顺序插入
  img_buffer_setic.push_back(img_cur_setic);
  img_time_buffer_setic.push_back(img_time_correct_setic);

  last_timestamp_img_setic = img_time_correct_setic;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
  
  std::cout << "[SETIC] Added frame to buffer in sync mode, buffer size: " 
            << img_buffer_setic.size() << ", timestamp: " << img_time_correct_setic << std::endl;
}

cv::Mat LIVMapper::getImageFromMsg_setic(const sensor_msgs::ImageConstPtr &img_msg_setic)//获取语义图像
{
    std::cout << "getImageFromMsg_setic" << std::endl;
    cv::Mat img_setic;
    img_setic = cv_bridge::toCvCopy(img_msg_setic, "bgr8")->image;
    if (img_setic.empty()) {
        std::cout << "empty image_setic" << std::endl;
        throw std::runtime_error("Received an empty image from ROS message");
    }
    return img_setic;
}

void LIVMapper::handleSETIC()
{
  // 检查是否在VIO中已经处理了SETIC，避免重复处理
  if (slam_mode_ == LIVO) {
    std::cout << "[SETIC] SETIC processing integrated into VIO+SETIC pipeline, skipping standalone processing" << std::endl;
    setic_processing = false;
    return;
  }

  std::cout << "[SETIC] handleSETIC called with timeout protection" << std::endl;
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (LidarMeasures.measures.empty()) {
    std::cout << "[SETIC] No measures data!" << std::endl;
    setic_processing = false;
    return;
  }

  const auto& measure = LidarMeasures.measures.back();
  std::cout << "[SETIC] measure setic_time: " << measure.setic_time << std::endl;

  // 检查语义图像数据是否有效
  cv::Mat current_img_setic = measure.img_setic;
  std::cout << "[DEBUG] handleSETIC: img_setic size = "
            << current_img_setic.rows << "x"
            << current_img_setic.cols << std::endl;

  if (current_img_setic.empty()) {
    std::cerr << "[SETIC] img_setic is empty, skip this frame!" << std::endl;
    setic_processing = false;
    setic_skip_count++;
    return;
  }
  
  std::cout << "[SETIC] Processing semantic image of size " 
            << current_img_setic.rows << "x" << current_img_setic.cols << std::endl;
  
  try {
    // 关键修复：带超时的SETIC处理
    auto process_start = std::chrono::high_resolution_clock::now();
    
    // 简化SETIC处理，减少计算复杂度
    setic_manager->processFrame_setic(current_img_setic, _pv_list, voxelmap_manager->voxel_map_);
    
    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);
    
    std::cout << "[SETIC] processFrame_setic completed in " << process_duration.count() << "ms" << std::endl;
    
    // 检查是否超时
    if (process_duration.count() > setic_timeout_ms) {
      std::cout << "[SETIC] Warning: Processing took " << process_duration.count() 
                << "ms, exceeding timeout of " << setic_timeout_ms << "ms" << std::endl;
      
      // 增加跳过计数，减少后续SETIC处理频率
      setic_skip_count = std::min(setic_skip_count + 2, max_setic_skip);
    } else {
      // 处理成功，重置跳过计数
      setic_skip_count = 0;
      last_successful_setic_time = measure.setic_time;
    }

    // 关键修复：不在SETIC中进行状态更新，避免里程计飘逸
    // 保持状态稳定，不发布里程计数据
    std::cout << "[SETIC] Skipping odometry update to maintain stability" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "[SETIC] Exception during processing: " << e.what() << std::endl;
    setic_skip_count = std::min(setic_skip_count + 3, max_setic_skip);
  }

  // 发布语义图像（如果需要可视化）
  try {
    publish_img_rgb_setic(pubImage_setic, setic_manager);
    std::cout << "[SETIC] publish_img_rgb_setic called" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[SETIC] Exception during image publishing: " << e.what() << std::endl;
  }
  
  // 标记SETIC处理完成
  setic_processing = false;
  
  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - start_time);
  
  std::cout << "[SETIC] Total processing time: " << total_duration.count() 
            << "ms, skip_count: " << setic_skip_count << std::endl;
}
