根据项目主函数/home/liu/code/LIVO2/ws_livo2/src/FAST-LIVO2/src/main.cpp
# 1.看 ROS 通信initializeSubscribersAndPublishers：
	subscribe：这个节点吃什么数据
	advertise：这个节点吐什么结果

	输入：
		LiDAR topic  -> livox_pcl_cbk / standard_pcl_cbk
		IMU topic    -> imu_cbk
		RGB image    -> img_cbk
		SETIC image  -> setic_cbk

	内部：
		callback -> buffer -> sync_packages判断数据量是否足够和时间戳是否对齐 -> run主循环


	输出：
		odometry
		path
		world pointcloud
		visual map
		semantic voxel map


# 2.run():

    ROS callbacks
        |
        v
    lid_raw_data_buffer / imu_buffer / img_buffer / img_buffer_setic
        |
        v
    **sync_packages(LidarMeasures)**   LidarMeasures为待处理的数据包
        |
        v
    handleFirstFrame() 设置时间零点（开始的时间）
        |
        v
    **processImu()**用 IMU 做状态预测 / 点云去畸变
        |
        v
    **stateEstimationAndMapping()**真正做 LIO / VIO / 建图
        |
        +--> handleLIO()
        +--> handleVIO()
        +--> handleSETIC()
        |
        v
    publish_odometry / publish_path / publish_frame_world







## sync_packages(LidarMeasures)：判断数据量是否足够和时间戳是否对齐，以下为LIVO的VIO和LIO的交换机制

### 1.上一轮是VIO/WAIT流程（则准备 LIO 数据包）：
	
    WAIT 表示系统刚开始，还没有完成过任何 LIO/VIO 更新。此时第一件事必须是：先用 LiDAR + IMU 把状态推进到第一张图像的时间点，所以这第一轮要准备并执行 LIO

    取buffer里最早图像时间 img_capture_time
    检查图像是否太旧（img_capture_time < meas.last_lio_update_time，说明这张图像的时间比系统已经处理过的 LiDAR 时间还早，直接丢掉这张图像）
    检查 LiDAR 和 IMU 是否已经覆盖到这个图像时间
    收集 last_lio_update_time 到 img_capture_time 之间的 IMU
    把 LiDAR 点按时间切开：
        小于 img_capture_time 的点 -> pcl_proc_cur
        大于 img_capture_time 的点 -> pcl_proc_next
        即：
            上次 LIO 时间        当前图像时间             LiDAR帧结束
            10.000              10.035                  10.100
            |-------------------|-----------------------|
                    当前处理              留到下次
            第 1 轮：
            LIO 处理 10.000 ~ 10.035
            VIO 使用 10.035 的图像

            第 2 轮：
            LIO 继续处理 10.035 ~ 下一个图像时间
            VIO 使用下一张图像


    meas.lio_vio_flg = LIO
    return true
		
### 2.上一轮是LIO流程（则准备 VIO 数据包）：


    取buffer里最早 RGB 图像数据（和VIO流程中的是同一张图片）
    寻找时间最接近的语义图像 setic
        dt = abs(setic_item.second - img_capture_time)，这里的setic_item.second为语义图的原始RGB图的扫描时间，
        dt为当前RGB图和语义buffer里的语义对应的原始RGB图的相差时间，取最小的dt对应的语义的原始RGB图与当前RGB图进行匹配；
        
        语义图是 RGB 图派生出来的，本应使用同一个原始时间戳；出现差异通常是因为处理延迟、队列错位、时间戳被写成发布时间，或者 RGB 端加了曝光时间补偿
        
        而python-bag5.py里面的PROCESSING_TIMEOUT_SEC = 0.15用于限制单帧语义处理允许耗时 150ms。
        脚本里如果处理超过这个时间，会返回/发布空白语义图，它影响语义内容是否有效


    弹出已经打包进当前 VIO 测量包的 RGB 图像

    meas.lio_vio_flg = VIO
    return true


### 问题：这里只弹出 RGB 图像和 RGB 时间，没有弹出 setic_frame_buffer_ 里的语义图。当前代码只是从语义 buffer 里找最近的图像拷到 m.img_setic，但没有同步删除语义缓存里的对应项。这个可能是作者为了复用/容错，也可能需要看 setic_cbk 或清理逻辑确认。

	
	9.970              10.000              10.035              10.070
	 |------------------|-------------------|-------------------|
	       LIO #1              LIO #2              LIO #3
		                ^                   ^                   ^
		              VIO #1              VIO #2              VIO #3
		            观测时间10.000       观测时间10.035       观测时间10.070
	LIO：从 last_lio_update_time 积分到 img_capture_time
	VIO 主要做的是：用当前 RGB 图像里的视觉信息，对当前时刻的位姿/状态做一次校正



## processImu()：进行状态更新和点云去畸变

    processImu()
        |
        +--> 
    Process2(*LidarMeasures*, _state, feats_undistort)
        |
        +-- IMU_init()，如果还没初始化
        |
        +-- UndistortPcl()，如果已初始化
                |
                +-- 状态传播到 prop_end_time，即state对应的传感器时刻
                +-- LIO：1.IMU对_state状态传播（用 IMU 积分更新10.000s-10.035s的姿态 rot_end，位置 pos_end，速度 vel_end，协方差 cov）  2.LiDAR点云去畸变
                    VIO：_state状态传播,和LIO的一样
        |
        v
    _state 状态更新
    feats_undistort 输出的去畸变点云更新
    voxelmap_manager 获得状态和去畸变点云

    *LidarMeasures*包括：
        1. 当前这轮要跑 LIO 还是 VIO
        2. 当前这轮对应的 IMU 数据
        3. 当前这轮对应的 LiDAR 点云或图像
        4. 当前这轮的目标时间
        5. 上一次 LIO/VIO 已经处理到的时间

## stateEstimationAndMapping()：

### handleLIO()：

    feats_undistort：processImu得到的去畸变点云
    |
    v
    降采样 -> feats_down_body
    |
    v
    按当前状态转世界系 -> feats_down_world
    |
    v
    首次：BuildVoxelMap()
    |
    v
    **StateEstimation()**
    输入：IMU预测状态 state_propagat + 当前点云 + 体素地图
    输出：优化后的 state_ 和带协方差的点列表 pv_list_
    |
    v
    **用优化后的 _state 重算点云世界坐标和协方差**
    StateEstimation() 之前的点云世界坐标可能是用预测状态算的，
    LIO 优化后 _state 变了，点的世界坐标也要跟着更新
    |
    v
    **UpdateVoxelMap()**将优化后的点云写回体素地图
    |
    v
    发布 odom/path/pointcloud/plane map


#### StateEstimation():

    IMU预测状态 state_propagat
            |
            v
    state_propagat赋值state_ 作为当前迭代状态
            |
            v
    TransformLidar()
    当前点云转世界系
            |
            v
    BuildResidualListOMP()
    找点-平面残差
            |
            v
    构造 Hsub / R_inv / meas_vec
            |
            v
    Kalman / ESIKF 更新
    state_ += solution
            |
            v
    根据平移/旋转变化判断收敛
            |
            v
    输出优化后的 state_


#### 用优化后的 _state 重算点云世界坐标和协方差

    _state = 优化后的状态
            |
            v
    transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar)
    用优化后状态_state重算 world_lidar
            |
            v
    pv_list_[i].point_w
    保存优化后的世界坐标

    body_cov_list_[i]
    点自身测量协方差

    _state.cov.block<3,3>(0,0)
    姿态协方差

    _state.cov.block<3,3>(3,3)
    位置协方差

    pv_list_[i].var
    世界系点协方差
            |
            v
    UpdateVoxelMap()
    用这些点更新地图


#### UpdateVoxelMap()

    input_points: 当前帧优化后的世界系点 + 协方差
            |
            v
    for each point
            |
            v
    根据 point_w / voxel_size 计算 VOXEL_LOCATION
            |
            +-- voxel 已存在
            |       |
            |       v
            |   *VoxelOctoTree::UpdateOctoTree(p_v)*
            |
            +-- voxel 不存在
                    |
                    v
                创建 VoxelOctoTree
                设置中心/层级/阈值
                UpdateOctoTree(p_v)
            |
            v
    *updateVoxelSemantic(input_points)*


##### UpdateOctoTree(p_v)：

    UpdateVoxelMap()
    插入当前帧点

    VoxelOctoTree::UpdateOctoTree()
    更新某个体素内部结构

    init_octo_tree() / init_plane()
    判断点是否能形成平面

    如果是平面：
    后续 StateEstimation 可以用点到平面残差

    如果不是平面：
    分裂成更小体素，点数量达到阈值后，判断点是否能形成平面



##### updateVoxelSemantic(input_points)
```
input_points
  |
  v
只看 semantic_label > 0 的点
  |
  v
用 point_w 找到 voxel
  |
  v
拿到 voxel_tree->plane_ptr_
  |
  v
addSemanticPoint(label)
  |
  v
semantic_count_ 计数增加
  |
  v
更新该平面的 dominant semantic label

```
### 问题：现在是“这个点属于哪个大 voxel，就给这个大 voxel 记一次语义”；更精细的做法是“这个点属于大 voxel 里的哪个 octree leaf，就给那个 leaf 的平面记一次语义”（已修改）/home/liu/code/LIVO2/ws_livo2/src/FAST-LIVO2/src/voxel_map.cpp line 1325-1334





### handleVIO():


    当前 VIO 包
    RGB图像 + 语义图 + 图像时间
            |
            v
    setSemanticMask()给 VIO 设置一张可用的语义遮罩
            |
            v
    vio_manager->**processFrame()**
    投影地图点 -> 图像误差 -> EKF更新 -> 维护视觉地图
            |
            v
    发布点云 / 发布图像 / 更新 latest_ekf_state




#### processFrame()：

    RGB图像 img
    |
    v
    resize / clone / semantic overlay语义叠加
    |
    v
    BGR -> Gray
    |
    v
    new_frame_ + 当前状态位姿
    |
    v
    retrieveFromVisualSparseMap()
    把当前帧对应的 3D 点（即LIO的出的LiDAR点）投影到当前图像
    检查是否在图像内()
    检查是否落在动态语义 mask 上

    从地图里找当前帧对应3D点对应的所有体素内的候选视觉点，投影到图像patch网格中，
    并控制图像每个网格上的点只有一个

    提取/比较 patch
    筛出当前帧可用的视觉点
    |
    v
    computeJacobianAndUpdateEKF()
    根据光度误差，计算雅可比 H + EKF 更新一开始IMU预测的状态
    |
    v
    generateVisualMapPoints()
    从当前图像对应的点云pg生成新的历史视觉点，加入到体素地图
    |
    v
    updateVisualMapPoints()
    告诉地图：这个老 3D 点（retrieveFromVisualSparseMap() 找到并通过筛选的那些点），
    这一帧也被相机看到了，并把当前帧的图像 patch/位姿/像素位置记录进它的观测历史
    |
    v
    updateReferencePatch()
    根据视角、光度一致性、NCC、误差等指标，决定这个地图点以后匹配时优先用哪个历史 patch 作为参考


##### retrieveFromVisualSparseMap()：
是在为 VIO 挑“可靠视觉点”：当前图像看得到、不是动态区域、没有明显遮挡、patch 匹配质量也过关的点，
才会进入后续滤波更新
```
line 480 ：if (isDynamicMaskPixel(px)) continue;点投影到动态语义区域，就跳过

pg：
当前帧 LIO 产生的点，用来确定当前应该查哪些 voxel

feat_map：
已有体素地图，用来从这些 voxel 中取历史 VisualPoint

1. 再次检查动态语义区域
2. 用 depth_img 做深度连续性检查，过滤遮挡边缘
3. 要求点的法向量已经初始化
4. 选择这个点的参考 patch
5. 计算当前帧和参考帧之间的仿射/单应变换
6. warp 参考 patch 到当前视角
7. 提取当前图像 patch
8. 计算光度误差
9. 可选 NCC 检查
10. 误差太大就剔除

参考帧来自 VisualPoint 的历史观测 pt->obs_；
当前函数从这些历史观测里选一个参考 patch，把它按当前相机视角 warp 过来，
再和当前图像 patch 比较光度一致性
```

# 3.语义部分
## 语义修改第一步：剔除动态地图点
```
语义输入：/home/liu/code/LIVO2/ws_livo2/yolov11/ultralytics/python-bag5.py
ros订阅setic_topic并回调：/home/liu/code/LIVO2/ws_livo2/src/FAST-LIVO2/src/LIVMapper.cpp中的setic_cbk()
```

### python-bag5.py

```
/camera/color/image_raw
    ↓
image_callback()
    ↓
ROS Image -> OpenCV BGR
    ↓
YOLO11 segmentation
    ↓
实例 mask + 类别名
    ↓
类别名映射到语义 ID
    ↓
生成 mono8 单通道 id_mask
    ↓
/setic/image_raw
```

### setic_cbk()
```
setic_cbk(msg):
  if SETIC 没开:
    return

  semantic_time = msg.header.stamp + offset

  if 时间戳倒退:
    return

  img = ROS Image 转 cv::Mat

  if img 为空:
    return

  if 开启有效性检查:
    if 非零像素太少 / 全背景:
      return

  缓存 latest_semantic_mask_id_

  if 没开 semantic_loop_en:
    更新时间戳
    return

以下为加了回环的效果
  加入 SETIC 帧同步 buffer
  尝试匹配 RGB + SETIC 帧

  ```

### handleLIO()中的语义：
```
filterDynamicPointsBeforeMapUpdate(
    voxelmap_manager->pv_list_,
    LidarMeasures.last_lio_update_time,
    _state,
    static_points_for_map,
    removed_dynamic_points);

voxelmap_manager->UpdateVoxelMap(static_points_for_map);


当前帧点云
  -> 根据语义 mask 过滤掉人/车等动态点
  -> 剩下 static_points_for_map
  -> 用静态点更新体素地图

```

VIO 里是 跳过动态语义区域的视觉像素/视觉点；LIO 里是 把 LiDAR 点投影到语义图，删除落在人车等动态类别上的点；体素地图层面则是 只插入过滤后的 static_points_for_map，动态点不进地图；
建图核心是 LIO/VIO 交替估计位姿，然后把当前帧 LiDAR 点插入体素平面地图voxelmap_manager->UpdateVoxelMap(static_points_for_map);






语义输出：
dynamic_mask:
  给现有 VIO/LIO 动态点过滤用
  person/car/bicycle/motorcycle/bus

static_semantic_mask:
  给语义回环用
  COCO 中相对静态、可复现的实例类




整体数据流保持：

Python YOLO 节点
  -> /setic/image_raw              动态 mask
  -> /semantic/static_mask         静态语义 mask

LIVMapper 主线程
  -> 正常 VIO/LIO
  -> 动态点过滤
  -> 体素地图更新
  -> 按关键帧策略生成 SemanticKeyframe 快照

SemanticLoopManager 副线程
  -> 接收关键帧快照
  -> 计算语义描述子
  -> 查询历史语义相似候选
  -> 对 top-K 候选做几何验证
  -> 输出 verified loop

核心原则：
主线程和回环线程不要共享可变对象。
主线程只给副线程传 深拷贝关键帧快照，副线程只输出回环结果，不直接改 _state、voxelmap_manager、_pv_list


## 语义回环修改


### 修改第二步：新增基于静态语义 mask 的回环检测


新增基于静态语义 mask 的回环检测，
1. 将有 被python-bag5.py 检测到静态语义类型的帧变为关键帧
2. 对关键帧进行语义匹配初筛
3. 对初筛后的关键帧进行体素几何筛选（多少个点云一样）
4. 对3.得到的关键帧进行ICP-lite验证
5. 可以用python3 src/FAST-LIVO2/scripts/semantic_loop_eval.py --log-dir src/FAST-LIVO2/Log --plot --plot-out semantic_loop_eval.png打印回环图片
5. 输出回环边，并且在rviz中将MarkerArray的Marker Topic选择为/semantic_loop/markers，就能在rviz看见回环边

主要改动如下：

1. Python 语义输出扩展
- 保留原有 /setic/image_raw 动态 mask 输出。
- 新增 /semantic/static_mask，用于发布静态语义类别background 0 的 mask。
- 新增 /semantic/static_vis，用于可视化静态语义结果。
- 静态类别暂选 traffic light、stop sign、parking meter、bench、chair、potted plant、fire hydrant 等少量稳定类别。
- 可通过 rostopic hz /semantic/static_mask 和 rostopic echo /semantic/static_mask/header 验证 Python 静态 mask 发布是否正常。
  STATIC_CLASS_IDS = {
      "traffic light": 10,
      "stop sign": 11,
      "parking meter": 12,
      "bench": 13,
      "chair": 14,
      "potted plant": 15,
      "fire hydrant": 16,
  }

2. 配置文件扩展
- 在 m3dgr_mid360.yaml 中新增 semantic_loop 参数组。
- 支持配置语义回环开关、静态 mask topic、关键帧触发间隔、最小静态像素数、最小静态点数、top-K 候选数量、语义相似度阈值和队列大小。
- 新增 common/verbose_console_log 与 common/keep_loop_event_log_when_quiet，用于控制终端日志输出，方便只保留关键 LOOP_EVENT 信息。

3. C++ 静态语义接收与缓存
- 在 LIVMapper 中新增 /semantic/static_mask 订阅。
- 新增静态语义 mask 缓存，包括 latest_static_semantic_mask_ 和 latest_static_semantic_time_。
- 新增 static_semantic_cbk() 和 getLatestStaticSemanticMask()。
- 收到静态 mask 后打印 nonzero 统计，用于确认 Python 到 C++ 的语义链路已经打通。

4. 语义关键帧快照
- 在 LIO 更新地图后，基于当前静态点云和静态语义 mask 创建 semantic keyframe。
- 每个关键帧记录 id、时间戳、累计路径长度、位姿、静态 mask、静态点云、静态像素数和点云数量。
- 通过日志输出 create keyframe，验证关键帧触发频率和数据规模是否合理。

5. SemanticLoopManager 副线程
- 新增 include/semantic_loop_manager.h 和 src/semantic_loop_manager.cpp。
- 使用独立线程接收语义关键帧，避免阻塞 LIO/VIO 主流程。
- 为每个关键帧计算全局语义直方图和 4x4 grid 语义直方图。
- 使用 cosine similarity 与历史关键帧计算语义相似候选。
- 支持 top-K 候选输出，用于第一阶段验证语义回环链路是否跑通。

6. 几何验证与 ICP-lite
- 在语义候选基础上增加静态点云几何重合验证。
- 使用 voxel overlap 统计几何 inliers 和 geo_ratio。
- 增加 ICP-lite 验证，使用降采样点云和粗配准一致性判断候选是否可靠。
- 增加回环冷却机制，避免同一区域重复输出过多回环事件。

7. 回环事件输出
- 当候选同时通过语义、几何和 ICP-lite 验证后，输出 LOOP_EVENT。
- LOOP_EVENT 包含 query/match id、rank、semantic score、dt、path distance、euclidean distance、geo inliers、geo ratio、ICP fitness、query/match position 和 relative distance。
- 新增 /semantic_loop/event，使用 std_msgs/String 发布回环事件，便于其他节点订阅。

8. RViz 可视化
- 新增 /semantic_loop/markers，使用 visualization_msgs/MarkerArray 发布回环边。
- 使用红色 LINE_LIST 显示 query keyframe 到 match keyframe 的回环连线。
- 使用 SPHERE_LIST 显示回环端点，便于在 RViz 中实时确认回环是否连对位置。

9. CSV 日志与离线评估
- 新增 semantic_loop_candidates.csv，记录候选帧及其语义/几何验证结果。
- 新增 semantic_loop_events.csv，记录最终接受的回环事件。
- 新增 semantic_keyframes.csv，记录语义关键帧轨迹和统计信息。
- 新增 scripts/semantic_loop_eval.py，用于统计回环数量、平均语义分数、平均几何重合率、平均 ICP fitness、平均时间间隔、平均路径距离和平均空间距离。
- 支持生成 semantic_loop_eval.png，蓝线表示语义关键帧轨迹，红线表示检测到的回环边。
- 评估脚本兼容旧 CSV header，能够正确解析 rel_dist 字段。

10. 构建系统
- 在 CMakeLists.txt 中将 src/semantic_loop_manager.cpp 加入 laser_mapping 编译目标。






### 修改第三步：构建语义关键帧为节点，语义关键帧之间的变化和回环为边，验证即将输入后端的数据是否健康
1. 将回环边（边1）的输出semantic_loop_constraints.csv设计成：平移+旋转四元数+相对平移+相对旋转+置信度+信息矩阵
2. 添加旋转约束max_relative_rotation_deg，并且将语义关键帧（边2）输出semantic_keyframes.csv 也输出四元数
3. 读取：semantic_keyframes.csv 和 semantic_loop_constraints.csv，然后构建：相邻 keyframe odom 边 和 semantic_loop 回环边 pose_graph_edges.csv作为后端输入雏形

  运行/home/liu/code/LIVO2/ws_livo2/src/FAST-LIVO2/scripts/build_pose_graph_offline.py：
  python3 scripts/build_pose_graph_offline.py --log-dir Log --check --plot

  输出文件：
    Log/pose_graph_nodes.csv 图节点
    Log/pose_graph_edges.csv 图边
    Log/pose_graph.png 轨迹blue和回环边red
    

4. 在/home/liu/code/LIVO2/ws_livo2/src/FAST-LIVO2目录下运行：进行离线优化，输出优化报告，并且画出优化前后的路径对比
python3 scripts/gtsam_pose_graph_optimize.py --log-dir Log --report --plot-loops

5. 加一个“优化结果评估脚本”，运行  python3 scripts/evaluate_pose_graph_optimization.py --log-dir Log
  读取：
  pose_graph_nodes.csv
  pose_graph_optimized_nodes.csv
  pose_graph_edges.csv

  输出：
  -1. 最大位移节点 top-k
  -2. 最大回环残差 top-k
  -3. 优化前后 loop residual 对比表
  -4. 是否存在异常移动节点
  这样以后换 bag、调参数时，可以快速判断：

  这次优化健康吗？
  有没有假回环？
  权重是不是太大？
  推荐判断阈值：

  max_position_change > 0.5m      警告
  max_rotation_change > 5deg      警告
  loop residual after 仍 > 0.5m   警告
  odom residual after 明显变大    警告


  ### 修改第四步：发布关键帧和回环边的topic，建立在线图容器：graph_nodes，odom_edges，loop_edges，连接在线前端和在线后端，对回环候选边做 ICP-lite 匹配输出真实位姿变化

  阶段 1：ROS topic replay 验证                 已完成
  阶段 2：C++ backend 订阅 + 在线构图           已完成
  阶段 3：在线导出 CSV + Python GTSAM batch     已完成
  阶段 4：C++ backend 内部 batch GTSAM          已完成
  阶段 5：C++ backend 内部 iSAM2
  阶段 6：发布 optimized_path / marker / status



1. 读取：pose_graph_nodes.csv和pose_graph_edges.csv
  按时间发布JSON类型toipc：
  /semantic_pg/keyframe
  /semantic_pg/loop_constraint

  然后把这两个 String/JSON topic 升级成正式 ROS 自定义消息（Keyframe.msg、LoopConstraint.msg）msg类型
  fast_livo/Keyframe
  fast_livo/LoopConstraint

                          检查改动成效：

                            1 启动 replay 节点
                            开一个终端：
                            roscore

                            另一个终端：
                            cd ~/code/LIVO2/ws_livo2
                            source devel/setup.bash
                            rosrun fast_livo semantic_pg_replay.py --log-dir src/FAST-LIVO2/Log --replay-rate 5.0 --keep-alive

                            2 检查 topic 类型
                            再开一个终端：
                            source ~/code/LIVO2/ws_livo2/devel/setup.bash
                            rostopic info /semantic_pg/keyframe
                            rostopic info /semantic_pg/loop_constraint

2. 写一个 C++ ROS 节点：semantic_pg_backend_node.cpp
  订阅：
  /semantic_pg/keyframe
  /semantic_pg/loop_constraint

  接着：收到 keyframe 建 node，收到相邻 keyframe 建 odom edge，收到 valid loop 建 semantic_loop edge


                            检查改动是否能接收topic：

                              1 启动 replay 节点
                              开一个终端：
                              roscore

                              另一个终端：
                              cd ~/code/LIVO2/ws_livo2
                              source devel/setup.bash
                              rosrun fast_livo semantic_pg_replay.py --log-dir src/FAST-LIVO2/Log --replay-rate 5.0 --keep-alive

                              再开一个终端：
                              source devel/setup.bash
                              rosrun fast_livo semantic_pg_backend_node

                              2 检查 topic 类型
                              再开一个终端：
                              source ~/code/LIVO2/ws_livo2/devel/setup.bash
                              rostopic info /semantic_pg/keyframe
                              rostopic info /semantic_pg/loop_constraint


  维护在线图容器：
  graph_nodes
  odom_edges（相邻 keyframe 自动生成，去重）
  loop_edges（有效 semantic loop，去重）

  输出：
  pose_graph_nodes_online.csv
  pose_graph_edges_online.csv


3. 现有 backend 里加 C++ batch GTSAM 优化 + 发布 optimized_path topic
  安装gtsam库，用gtsam库优化关键帧位姿

  发布topic
  /semantic_pg/keyframe_path
  /semantic_pg/optimized_path
  /semantic_pg/loop_markers 回环边
  并在rviz中显示


  连接在线前端和在线后端：
  LIVMapper.cpp读取fast_livo/Keyframe 与 fast_livo/LoopConstraint，
  发布/semantic_pg/keyframe 和 /semantic_pg/loop_constraint到后端

  SemanticPgBackend 后端节点
  订阅：
  /semantic_pg/keyframe             fast_livo/Keyframe
  /semantic_pg/loop_constraint      fast_livo/LoopConstraint

  发布：
  /semantic_pg/keyframe_path        nav_msgs/Path
  /semantic_pg/optimized_path       nav_msgs/Path
  /semantic_pg/loop_markers         visualization_msgs/MarkerArray


  /semantic_pg/keyframe_path
  原始关键帧 Path，优化前。

  /semantic_pg/optimized_path
  GTSAM 优化后的关键帧 Path，优化后。

  /semantic_pg/loop_markers
  绿色回环边 MarkerArray，用来显示哪些关键帧之间加了 semantic loop constraint。

  /semantic_pg/keyframe_markers
  新增的点状可视化，黄色点是原始关键帧，红色点是优化后关键帧，更适合判断优化前后节点有没有异常跳动。



对回环候选边做 ICP-lite 匹配/验证，而不是只输出当前 odom 反算位姿


### 修改第五步：C++ backend 内部 iSAM2 优化器
1. 使用iSAM2优化器进行增量式优化
2. 把 iSAM2 优化后的关键帧修正量，回写到原始连续轨迹 /path 上，生成一条“优化后的连续轨迹”
  对每个关键帧计算一个修正变换：
  T_delta_i = T_optimized_i * inverse(T_raw_i)
  也就是“这个关键帧从原始位置变到优化位置，需要怎么平移和旋转”。

  keyframe 输出时记录对应 LIO/VIO 时间
  /path 输出时记录当前位姿对应的 LIO/VIO 时间
  对 /path 里的每个连续 pose，根据时间找到它附近的两个 keyframe。
  在这两个 keyframe 的修正量之间插值：
  平移用线性插值
  旋转用四元数 slerp
  把插值得到的修正量应用到该 pose：
  p_corrected = R_delta * p_raw + t_delta
  q_corrected = q_delta * q_raw
  发布结果到：
  /semantic_pg/corrected_path



## debug

语义回环前端已经能产生候选和 LOOP_EVENT。
语义回环边已经能发布到 /semantic_pg/loop_constraint。
后端已经能接收 keyframe、odom 边、loop 边，并做 GTSAM/iSAM2 优化。
/semantic_pg/corrected_path 已经能输出，并能转成 TUM txt 跟 GT 做 evo_ape。
你已经发现：加回环后 RMSE 没有比 raw 好，甚至会变差。

将边2从语义keyframe变成了odom keyframe




1. 调回环边的权重/回环边中旋转的权重  RMSE仍然比raw差

2. 做 IPC lite 的点云和关键帧不匹配   并未出现这个问题



3. 修正坐标系

边1：loop IMU/body 坐标系
局部静态点云static_cloud_local 当前是 IMU/body 坐标系
ICP lite通过static_cloud_local之间的变换得到query和match之间的位姿变换：query local cloud -> match local cloud

边2：keyframe loop IMU/body 坐标系







4. ICP lite 计算不准确
将ICP lite改为 PCL ICP

5. RMSE不稳定

- 修改顶点/边的设计：

  odom0 -- odom1 -- odom2 -- odom3 -- odom4
            |                  |        |
        anchor_a           anchor_b  anchor_c
            \________________/
              semantic_loop

  loop 边连接 semantic anchor，而不是强行映射到最近 odom
  semantic_loop 约束 anchor，anchor 再通过 attachment 边影响 odom 主图



- 关键帧生成策略：
  启动延迟 10s 后，每 2s 才检查一次；
  平移超过 2m 或旋转超过 20deg 就创建；
  如果 8s 内一直没达到运动阈值，也强制创建一个



- 修改信息矩阵的设计：
  odom边不变，固定信息矩阵权重
  semantic_anchor边：固定信息矩阵权重
  semantic_loop 边：根据三项质量自适应score，geo_ratio，icp_fitness信息矩阵权重



- 对回环边添加鲁棒核函数 k=1.345
  根据残差对边施加惩罚，防止坏边破坏优化结果





## 新增语义类别
训练yolov11_seg：
室内： 网 net
      桌子 table
      门 door
      chair
      monitor
室外： building
      traffic_sign




names:
  1: person
  2: car
  3: bicycle
  4: motorcycle
  5: bus
  6: truck
  
  10: net
  11: table
  12: door
  13: chair
  14: tv
  15: traffic_sign
  16: building


## 修改优化器的节点和边

       semantic_loop
     ┌─────────────────┐
     │                 ▼
    x0 ── x1 ── x2 ── x3 ── x4
    ▲     semantic_odom      │
    └────────────────────────┘
          semantic_loop


## 回环检测方案修改：增加三维语义拓扑验证和回退

  语义直方图召回
      ↓
  拓扑结构可比较 → 拓扑验证
  拓扑信息不足   → 静态点云体素重叠验证
      ↓
  ICP 几何验证
      ↓
  加入位姿图

2. 三维语义拓扑节点
每个代理实例包含：
  类别 ID
  三维质心
  三维尺寸
  像素数量
  点云数量
拓扑边由两个节点的：
  类别组合
  三维质心距离
构成。
3. 遮挡碎片合并
语义节点生成流程改为：
  二维连通域过滤小噪声。
  同类别有效区域的点云统一汇集。
  在三维空间进行聚类。
  根据包围盒间距和深度差合并遮挡碎片

4. 信息矩阵改进
回环边权重现在综合考虑：
  语义直方图分数
  拓扑分数
  匹配拓扑边数量
  ICP fitness
  位姿是否来自 ICP



# 方向一：回环一致性的 3DGS 子地图校正

这是最适合与你现有“语义回环”结合的方向。

问题是什么？

你当前 FAST-LIVO2 的轨迹在长时间运行后会出现累计漂移。语义回环触发后，iSAM2 会修正关键帧位姿。

但是，如果 Gaussian 地图已经按照回环前的轨迹构建，地图不会自动完全正确地更新，可能出现：

墙面双层
门框错位
纹理撕裂
同一区域重复重建
局部 Gaussian 重影

已有 3DGS-SLAM 工作也关注这一问题。GLC-SLAM 将场景划分为 Gaussian 子地图，在回环修正后高效更新局部地图；LoopSplat 则通过注册 3D Gaussian 子地图处理全局一致性。

你可以怎么做？

将 Gaussian 地图划分为多个子地图，每个子地图绑定一组关键帧：

Submap 0 ← keyframe 0–30
Submap 1 ← keyframe 31–60
Submap 2 ← keyframe 61–90

回环优化前，第 k 个关键帧的位姿为：

T_raw(k)

iSAM2 优化后得到：

T_opt(k)

计算校正增量：

ΔT(k) = T_opt(k) · inverse(T_raw(k))

将对应子地图中的 Gaussian 中心和旋转更新：

μ_new = ΔT(k) · μ_old
R_new = ΔR(k) · R_old

然后只对受影响子地图做少量局部重优化。

输入与输出
输入
iSAM2 优化前后的关键帧位姿
Gaussian 子地图
关键帧 RGB 图像
相机内参和外参
输出
全局一致的 Gaussian 地图
更新后的 gaussians.ply
回环校正前后的 RGB / 深度渲染图
可以怎么写成创新点？

提出一种语义回环驱动的 Gaussian 子地图校正机制。根据位姿图优化后的关键帧增量变换，对受影响 Gaussian 子地图进行刚性变形和局部重优化，从而减少长序列重建中的地图重影和纹理撕裂。

与你现有语义回环的关系
创新点 1：发现并验证回环
创新点 2：利用回环结果真正修正稠密地图


# 4. 主线程 副线程 独立节点

## 主线程：laserMapping 主循环
  
  ROS 回调处理
  LiDAR / IMU / 图像缓存
  多传感器时间同步
  IMU 初值
  LIO/VIO 状态估计
  动态剔除体素地图更新
  发布 odom / path / 点云 / map
  创建 odom keyframe
  创建 semantic keyframe，并丢给回环线程

  handleLIO()：这里做降采样、体素地图匹配、动态点剔除、地图更新、发布、创建 keyframe

## SemanticLoopManager 回环线程

  计算语义描述子
  查找历史候选 keyframe
  语义相似度筛选
  几何验证
  ICP-lite 验证
  生成 LoopEvent
  通过 callback 发布 /semantic_pg/loop_constraint

## 独立节点：semantic_pg_backend_node

它订阅：

  /semantic_pg/keyframe
  /semantic_pg/loop_constraint
  /path

然后做：

  收 odom keyframe
  收 semantic loop edge
  构建 pose graph
  GTSAM batch 或 ISAM2 优化
  发布 /semantic_pg/optimized_path
  发布 /semantic_pg/corrected_path
  发布 markers