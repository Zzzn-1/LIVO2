# SETIC副线程使用指南

## 快速开始

### 1. 配置启用SETIC副线程

在配置文件 `config/avia.yaml` 中确保：

```yaml
common:
  setic_en: 1                        # 启用SETIC处理
  setic_topic: "/setic/image_raw"    # 语义图像话题
```

### 2. 编译系统

```bash
cd /home/ao/jiguang/LIVO2/ws_livo2
catkin_make
```

### 3. 运行系统

启动主程序：
```bash
roslaunch fast_livo2 mapping.launch
```

### 4. 验证副线程工作

查看日志输出，应该看到：
```
[SETIC] Starting SETIC processing thread...
[SETIC] Thread started successfully
```

### 5. 测试语义处理

运行测试程序（可选）：
```bash
# 编译测试程序
./scripts/compile_setic_test.sh

# 运行测试
rosrun FAST-LIVO2 test_setic_thread
```

## 监控副线程状态

### 查看话题
```bash
# 查看语义图像输入
rostopic echo /setic/image_raw

# 查看处理结果输出  
rostopic echo /setic_img

# 查看话题列表
rostopic list | grep setic
```

### 监控性能
```bash
# 查看处理频率
rostopic hz /setic_img

# 查看系统资源使用
htop
```

## 主要优势

✅ **零干扰**：副线程完全不影响LIO-VIO主线程性能  
✅ **异步处理**：语义图像处理不阻塞实时定位  
✅ **自动管理**：任务队列自动处理溢出和异常  
✅ **线程安全**：完善的锁机制确保数据安全  
✅ **资源控制**：智能队列管理避免内存泄漏  

## 故障排除

### 常见问题

**1. 副线程未启动**
- 检查 `setic_en: 1` 是否正确配置
- 确认日志中有 "Thread started successfully" 信息

**2. 没有接收到语义图像**
- 检查话题名称是否正确：`/setic/image_raw`
- 确认有程序发布语义图像数据

**3. 处理结果异常**
- 查看错误日志：`[SETIC-THREAD]` 开头的信息
- 检查语义图像格式是否正确（BGR8）

### 调试命令

```bash
# 查看节点信息
rosnode info /fast_livo2_node

# 检查话题连接
rostopic info /setic/image_raw
rostopic info /setic_img

# 实时监控日志
rosout | grep SETIC
```

## 配置参数

可在代码中调整的关键参数：

```cpp
// 最大队列大小（防止内存占用过多）
static const int MAX_SETIC_QUEUE_SIZE = 3;

// 处理超时时间（毫秒）
setic_timeout_ms = 50;

// 跳帧阈值
max_setic_skip = 5;
```

## 性能调优建议

1. **硬件配置**
   - CPU：多核处理器（推荐4核以上）
   - 内存：至少8GB
   - 确保有足够的计算资源

2. **参数调整**
   - 根据硬件性能调整队列大小
   - 根据实时性要求调整超时时间
   - 监控CPU使用率调整处理频率

3. **系统优化**
   - 使用SSD存储提升I/O性能
   - 适当设置ROS日志级别
   - 关闭不必要的可视化

---

**注意事项：**
- SETIC副线程设计为可选组件，禁用时不影响主要SLAM功能
- 系统会自动处理语义数据缺失的情况，保证鲁棒性
- 副线程异常不会导致主线程崩溃，确保系统稳定性 