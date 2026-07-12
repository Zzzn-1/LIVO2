# 独立SETIC语义映射功能

## 概述

本实现将语义图像处理(SETIC)从主SLAM节点中独立出来，并提供了将2D语义物体映射到LIO体素坐标的功能。

## 主要特性

### 1. 独立SETIC处理
- **独立线程**: SETIC处理运行在独立的线程中，不影响主SLAM系统的实时性能
- **异步处理**: 语义图像异步处理，避免阻塞LIO和VIO流程
- **线程安全**: 使用互斥锁和条件变量确保线程间数据安全

### 2. 语义物体检测
- **最多三种类别**: 每次处理最多返回三种类别的语义物体
- **2D轮廓提取**: 从语义分割结果中提取物体的2D轮廓点
- **置信度评估**: 每个检测到的物体都有相应的置信度分数

### 3. 2D到3D映射
- **深度估计**: 通过LiDAR点云为2D像素点估计深度
- **坐标变换**: 将2D像素坐标转换为3D世界坐标
- **体素映射**: 将3D世界坐标映射到体素地图中的具体体素位置

## 核心数据结构

### SemanticObject2D
```cpp
struct SemanticObject2D {
    int class_id;                       // 物体类别ID
    std::vector<cv::Point2f> contour;   // 物体轮廓点
    cv::Point2f center;                 // 物体中心点
    double confidence;                  // 置信度
};
```

### SemanticObject3D
```cpp
struct SemanticObject3D {
    int class_id;                       // 物体类别ID
    std::vector<V3D> points_3d;         // 3D点云
    V3D center_3d;                      // 3D中心点
    std::vector<VOXEL_LOCATION> voxel_locations; // 对应的体素位置
};
```

### SemanticVoxelMapping
```cpp
struct SemanticVoxelMapping {
    int class_id;                       // 物体类别ID
    std::vector<VOXEL_LOCATION> voxel_locations; // 体素位置列表
    double timestamp;                   // 时间戳
    bool is_valid;                      // 映射是否有效
};
```

## 主要函数功能

### 线程控制函数
- `startIndependentSeticThread()`: 启动独立SETIC处理线程
- `stopIndependentSeticThread()`: 停止独立SETIC处理线程
- `independentSeticProcessingLoop()`: 独立处理循环主函数

### 坐标映射函数
- `map2DToVoxelCoordinates()`: 将2D坐标数组映射到3D体素坐标
- `pixelToWorldCoordinate()`: 将单个像素坐标转换为世界坐标
- `worldToVoxelLocation()`: 将世界坐标转换为体素位置
- `getDepthFromLidarPoint()`: 从LiDAR点云获取深度信息

### 结果处理函数
- `publishSemanticVoxelMapping()`: 发布语义体素映射结果
- `clearSemanticMappingResults()`: 清空语义映射结果
- `extractSemanticObjects2D()`: 从语义图像中提取2D物体信息

## 使用方法

### 1. 启用独立SETIC处理
在构造函数中设置:
```cpp
setic_independent_processing = true;  // 启用独立SETIC处理
setic_mapping_enabled = true;         // 启用语义映射
max_semantic_objects = 3;             // 最多处理三种语义物体
```

### 2. 配置映射参数
```cpp
depth_estimation_range = 50.0;       // 深度估计范围(米)
voxel_search_radius = 0.5;           // 体素搜索半径(米)
min_depth_threshold = 0.1;           // 最小深度阈值(米)
max_depth_threshold = 100.0;         // 最大深度阈值(米)
```

### 3. 获取映射结果
通过访问成员变量获取结果:
```cpp
// 2D语义物体
std::vector<SemanticObject2D> semantic_objects_2d;

// 3D语义物体
std::vector<SemanticObject3D> semantic_objects_3d;

// 体素映射结果
std::vector<SemanticVoxelMapping> semantic_voxel_mapping;
```

## 输出示例

```
[SETIC-THREAD] Processing semantic image at timestamp: 1234567890.123
[SETIC-THREAD] Detected 3 semantic objects
[SETIC-THREAD] Object class 1 mapped to 15 voxel locations
[SETIC-THREAD] Object class 2 mapped to 23 voxel locations
[SETIC-THREAD] Object class 3 mapped to 18 voxel locations

[SETIC-MAPPING] Publishing semantic voxel mapping results:
  Class 1: 15 voxels
    Voxel[0]: (10, 5, 2)
    Voxel[1]: (10, 6, 2)
    Voxel[2]: (11, 5, 2)
    ... and 12 more voxels
  Class 2: 23 voxels
    Voxel[0]: (20, 15, 3)
    Voxel[1]: (20, 16, 3)
    ... and 21 more voxels
```

## 测试功能

使用 `testSemanticMapping()` 函数可以测试映射功能:
```cpp
void testSemanticMapping(); // 运行语义映射测试
```

## 注意事项

1. **相机内参**: 需要正确配置相机内参矩阵以确保坐标变换准确
2. **外参标定**: LiDAR-相机外参必须准确标定
3. **深度阈值**: 根据实际场景调整深度阈值参数
4. **体素大小**: 体素大小应该与实际应用需求匹配
5. **线程安全**: 在访问共享数据时注意线程安全

## 扩展建议

1. **实时可视化**: 可以在RViz中可视化语义体素映射结果
2. **语义SLAM**: 基于语义体素信息进行更高层次的SLAM
3. **动态更新**: 实现语义信息的动态更新和管理
4. **多类别处理**: 扩展到处理更多类别的语义物体 