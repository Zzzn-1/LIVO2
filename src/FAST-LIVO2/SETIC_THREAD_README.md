# SETIC副线程语义处理系统

## 概述

本系统在原有的LIO-VIO主线程基础上，添加了一个独立的SETIC副线程来处理语义图像，确保语义处理不会影响主线程的实时性能。

## 主要特性

### 1. 独立副线程处理
- **完全异步**: SETIC处理运行在独立线程中，与LIO-VIO主线程完全并行
- **非阻塞设计**: 语义图像处理不会阻塞主要的里程计算法
- **线程安全**: 使用互斥锁和条件变量确保数据安全

### 2. 智能任务队列管理
- **队列大小限制**: 最大3个任务，防止内存占用过多
- **自动丢弃机制**: 队列满时自动丢弃最旧任务
- **时间戳检查**: 避免时间戳回退的无效数据

### 3. 鲁棒性保障
- **异常处理**: 完善的异常捕获和错误处理
- **资源管理**: 线程安全的启动和停止机制
- **调试支持**: 详细的日志输出便于调试

## 系统架构

```
主线程 (LIO-VIO)                    副线程 (SETIC)
     |                                    |
   图像采集                           语义图像处理
     |                                    |
  setic_cbk() -----> 任务队列 -----> seticThreadMain()
     |                |                   |
  继续主流程           |              processSeticAsync()
                      |                   |
                 异步传递              语义分析处理
```

## 使用方法

### 1. 启用SETIC副线程

在配置文件中设置:
```yaml
common:
  setic_en: 1                        # 启用SETIC处理
  setic_topic: "/setic/image_raw"    # 语义图像话题
```

### 2. 调整性能参数

根据需要调整以下参数:
```cpp
static const int MAX_SETIC_QUEUE_SIZE = 3;   // 最大队列大小
setic_timeout_ms = 50;                       // 处理超时时间(ms)
```

### 3. 监控线程状态

通过日志监控线程运行状态:
```bash
# 启动信息
[SETIC] Starting SETIC processing thread...
[SETIC] Thread started successfully

# 处理信息  
[SETIC-THREAD] Processed frame in 25 ms

# 停止信息
[SETIC] Stopping thread...
[SETIC] Thread stopped
```

## 关键函数说明

### 线程管理函数

#### `startSeticThread()`
- **功能**: 启动SETIC副线程
- **调用时机**: 在`initializeSubscribersAndPublishers()`中自动调用
- **线程安全**: 是

#### `stopSeticThread()`
- **功能**: 安全停止SETIC副线程
- **调用时机**: 在析构函数中自动调用
- **阻塞行为**: 等待线程完全结束

#### `seticThreadMain()`
- **功能**: 副线程主循环
- **处理逻辑**: 等待任务 → 获取任务 → 处理任务 → 发布结果
- **停止条件**: `setic_thread_should_stop_`标志

### 数据处理函数

#### `setic_cbk()`
- **功能**: 语义图像回调函数
- **行为**: 将图像数据封装为任务并添加到队列
- **非阻塞**: 不进行实际处理，立即返回

#### `processSeticAsync()`
- **功能**: 异步处理语义数据
- **输入**: 语义图像、点云数据、体素地图、时间戳
- **输出**: 处理后的语义信息

#### `publish_img_rgb_setic()`
- **功能**: 发布处理后的语义图像
- **话题**: `/setic_img`
- **格式**: BGR8编码

## 数据结构

### SeticTask
```cpp
struct SeticTask {
    cv::Mat img_setic;                                           // 语义图像
    std::vector<pointWithVar> pv_list;                          // 点云数据
    std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*> voxel_map; // 体素地图
    double timestamp;                                            // 时间戳
    bool valid;                                                  // 有效性标志
};
```

## 性能特点

### 1. 主线程保护
- **零干扰**: 语义处理完全不影响LIO-VIO性能
- **实时保障**: 主线程维持原有的处理频率
- **内存隔离**: 副线程使用独立的数据副本

### 2. 副线程优化
- **适应性处理**: 根据系统负载自动调整处理频率
- **内存控制**: 限制队列大小防止内存泄漏
- **CPU友好**: 适当休眠避免CPU占用过高

### 3. 鲁棒性设计
- **故障隔离**: 副线程异常不影响主线程
- **优雅降级**: 语义数据缺失时系统正常运行
- **资源管理**: 自动清理资源避免内存泄漏

## 调试和监控

### 1. 日志级别
- **ROS_DEBUG**: 详细处理信息
- **ROS_WARN**: 警告信息
- **ROS_ERROR**: 错误信息

### 2. 性能监控
```bash
# 监控队列状态
[SETIC] Added task to queue, queue size: 2, timestamp: 1234567890.123

# 监控处理时间
[SETIC-THREAD] Processed frame in 25 ms

# 监控队列溢出
[SETIC] Queue full, dropping oldest task
```

### 3. 故障排查

**常见问题及解决方案:**

1. **线程启动失败**
   - 检查`setic_en`参数是否启用
   - 确认`setic_manager`是否正确初始化

2. **队列频繁溢出**
   - 增加`MAX_SETIC_QUEUE_SIZE`
   - 优化语义处理算法性能

3. **图像发布失败**
   - 检查`pubImage_setic`发布器是否正确初始化
   - 确认图像数据格式是否正确

## 使用建议

### 1. 参数调优
- 根据硬件性能调整队列大小
- 根据实时性要求调整超时时间
- 根据精度要求调整处理频率

### 2. 系统集成
- 确保语义图像话题正确配置
- 验证时间戳同步性
- 测试异常情况下的系统行为

### 3. 性能优化
- 监控CPU和内存使用情况
- 根据实际需求调整处理算法
- 考虑GPU加速语义处理

## 总结

SETIC副线程系统成功实现了语义处理与主要SLAM算法的解耦，确保了系统的实时性能和鲁棒性。通过独立的线程架构，系统可以在不影响定位精度的前提下，提供丰富的语义信息。 




VIO: BuildVoxelMap() -> StateEstimation() -> UpdateVoxelMap()
SETIC: buildSeticVoxelMap() -> seticVoxelStateEstimation() -> publishSemanticVoxelMap()