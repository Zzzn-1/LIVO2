# 进入launch命令
cd /home/ao/jiguang/LIVO2/ws_livo2
source devel/setup.bash
roslaunch fast_livo mapping_avia.launch


roslaunch fast_livo mapping_avia_m2dgr.launch

roslaunch fast_livo mapping_avia_ntu.launch


roslaunch fast_livo mapping_avia_marslvig.launch


roslaunch fast_livo mapping_avia.launch

## 不启用rviz
roslaunch fast_livo mapping_m3dgr_mid360.launch rviz:=false


### 进入硬盘命令
cd /media/ao/jiansheng/datasets/LIVO2_dataset
rosbag play Retail_Street.bag

rosbag play HKisland03.bag -s 72
rosbag play HKairport03.bag -s 62


rosbag play Retail_Street.bag

# 降低数据集播放速度到原来的0.3
rosbag play xxx.bag -r 0.3

#### 另外的数据集
cd /media/ao/T7/Dataset/M2DGR/street_02
rosbag play street_02.bag

cd /media/ao/T7/Dataset/M2DGR/street_07


cd /media/ao/jiansheng/datasets/LIVO2_dataset/NTU_VIRAL_dataset/eee_03

### 语义输入步骤
cd /home/ao/yolov11/ultralytics
python python-bag5.py

## rqt命令查看实时图片

rqt_image_view 

## 查看.pcd文件

pcl_viewer xxxx.pcd


## 处理NTU_VIRAL数据集的输出
cd /home/liu/code/LIVO2/ws_livo2/src/FAST-LIVO2/Log/result 找到输出的eee_01.txt文件

将文件通过/home/liu/code/LIVO2/ws_livo2/src/FAST-LIVO2/Log/result/liu_NTU_VIRAL 内的evaluate_viral.py 将IMU坐标系下的点eee_01.txt转成棱镜 prism 的坐标系下eee_01_prism.txt再与eee_01_gt.txt进行rmse精度分析
evo_ape tum nya_03_gt.txt nya_03_prism.txt -a -p

对比三个数据
evo_traj tum --ref=eee_01_gt.txt eee_01.txt eee_01_yolo_prism.txt -a -p


##  Run M3DGR example
```
source devel/setup.bash
roslaunch fast_livo mapping_m3dgr_avia.launch

# for mid360
roslaunch fast_livo mapping_m3dgr_mid360.launch

rosbag play Dynamic01.bag

```

## 监控输出，从 rosout 里实时筛

roslaunch fast_livo mapping_m3dgr_mid360.launch 2>&1 | stdbuf -oL grep -E --line-buffered "\[SemanticLoop\]|\[STATIC-SEM\]"


roslaunch fast_livo mapping_m3dgr_mid360.launch 2>&1 | stdbuf -oL grep -E --line-buffered "\[SemanticLoop\] LOOP_EVENT"


## 查看要输入后端的topic

liu@liu-pc:~/code/LIVO2/ws_livo2$ source devel/setup.bash 
liu@liu-pc:~/code/LIVO2/ws_livo2$ rostopic echo /semantic_loop/constraint

liu@liu-pc:~/code/LIVO2/ws_livo2$ rostopic echo /semantic_loop/backend_stats


# 有后端的运行

cd /home/liu/code/LIVO2/ws_livo2
catkin_make
source devel/setup.bash
roslaunch fast_livo mapping_m3dgr_mid360.launch 2>&1 | stdbuf -oL grep -E --line-buffered "\[SemanticLoop\] LOOP_EVENT"

可以不需要，因为roslaunch fast_livo mapping_m3dgr_mid360.launch已经包含了semantic_pg_backend_node
        另开一个终端：
        cd /home/liu/code/LIVO2/ws_livo2
        source devel/setup.bash
        rosrun fast_livo semantic_pg_backend_node

        rosrun fast_livo semantic_pg_backend_node _odom_weight_scale:=0.5 _loop_weight_scale:=0.85


再开终端：
liu@liu-pc:~/code/LIVO2/ws_livo2/yolov11/ultralytics$ python3 python-bag5.py 

再开终端：
liu@liu-pc:/media/liu/LINUX&WIN/Dataset/M3DGR/FAST-LIVO2-mid360$ rosbag play Dynamic01.bag 




# 清理ros日志：
rosclean purge -y

## 检查ros日志用了多少空间：
rosclean check


# 语义回环离线评估（统计误差+优化前后的关键帧连线图）
liu@liu-pc:~/code/LIVO2/ws_livo2/src/FAST-LIVO2$ python3 scripts/gtsam_pose_graph_optimize.py --log-dir Log --online --report --plot-loops





























