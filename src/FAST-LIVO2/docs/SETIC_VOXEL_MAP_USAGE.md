# SETIC体素地图使用说明

## 概述

SETIC体素地图是基于VIO体素地图构建方法实现的语义增强地图系统。它将语义信息与几何信息结合，构建语义感知的体素地图。

## 主要功能

### 1. 语义体素地图构建
- 基于VIO的体素地图构建方法
- 融合语义图像信息
- 异步处理保证实时性能

### 2. 发布的ROS话题

#### 输入话题
- `/setic/image_raw` - 语义图像输入
- `/left_camera/image` - RGB图像输入
- `/livox/lidar` - LiDAR点云输入
- `/livox/imu` - IMU数据输入

#### 输出话题
- `/setic_planes` - 语义平面可视化 (visualization_msgs/MarkerArray)
- `/setic_voxel_map` - 语义体素地图点云 (sensor_msgs/PointCloud2)
- `/setic_img` - 处理后的语义图像 (sensor_msgs/Image)

### 3. 核心组件

#### SETIC体素地图管理器 (setic_voxelmap_manager)
- 管理语义体素地图的构建和更新
- 执行状态估计和协方差传播
- 处理地图滑动窗口

#### 异步处理线程
- 独立的SETIC处理线程
- 避免阻塞主要的LIO/VIO处理流程
- 任务队列管理和超时处理

## 使用方法

### 1. 基本启动
```bash
# 启动FAST-LIVO2系统（包含SETIC功能）
roslaunch fast_livo fastlivo_mapping.launch

# 或使用专门的SETIC测试启动文件
roslaunch fast_livo test_setic_voxel.launch
```

### 2. 参数配置

在`config/setic_voxel_config.yaml`中配置SETIC相关参数：

```yaml
setic_voxel_config:
  voxel_size: 0.2                    # 体素大小
  enable_setic_voxel_pub: true       # 启用体素地图发布
  enable_semantic_plane_pub: true    # 启用平面发布
```

### 3. 可视化

使用RViz查看SETIC体素地图：
- 添加`/setic_voxel_map`话题显示语义点云
- 添加`/setic_planes`话题显示语义平面
- 添加`/setic_img`话题显示处理后的语义图像

## 技术特点

### 1. 与VIO体素地图的一致性
- 使用相同的体素地图构建方法
- 共享外参和状态信息
- 保持处理流程的一致性

### 2. 异步处理架构
- SETIC处理在独立线程中执行
- 不影响主要的LIO/VIO处理性能
- 通过任务队列管理处理流程

### 3. 协方差传播
- 完整的不确定性传播
- 考虑传感器外参的影响
- 保持与主体素地图相同的精度

## 调试和监控

### 1. 日志输出
系统会输出以下关键信息：
- `[SETIC] Building initial SETIC voxel map...` - 初始地图构建
- `[SETIC] Updated SETIC voxel map with X points` - 地图更新信息
- `[SETIC] Published SETIC voxel map with X points` - 发布信息

### 2. 性能监控
- 监控SETIC处理线程的执行时间
- 检查任务队列的积压情况
- 观察帧同步的成功率

### 3. 常见问题排查

#### 体素地图不更新
- 检查语义图像话题是否正常发布
- 确认SETIC处理线程是否正常运行
- 验证点云数据的有效性

#### 处理延迟过高
- 降低语义图像的分辨率
- 增加SETIC任务队列大小
- 优化体素地图参数

## 扩展开发

### 1. 添加新的语义类别
在SETIC管理器中添加新的语义处理逻辑：
```cpp
setic_manager->addSemanticClass(class_id, class_name, processing_func);
```

### 2. 自定义体素地图属性
扩展体素结构以包含更多语义信息：
```cpp
struct SemanticVoxel : public VoxelOctoTree {
    std::vector<int> semantic_labels;
    std::vector<float> confidence_scores;
};
```

### 3. 与其他模块集成
SETIC体素地图可以与路径规划、目标检测等模块集成使用。

## 注意事项

1. 确保语义图像与RGB图像时间同步
2. 调整体素大小以平衡精度和性能
3. 根据计算资源调整异步处理参数
4. 定期检查内存使用情况，避免地图过大 