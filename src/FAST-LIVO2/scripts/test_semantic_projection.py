#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语义投影测试脚本 - 增强版
用于测试FAST-LIVO2中的语义体素投影功能
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import time

class SemanticProjectionTester:
    def __init__(self):
        rospy.init_node('semantic_projection_tester', anonymous=True)
        
        # ROS Publishers and Subscribers
        self.semantic_pub = rospy.Publisher('/semantic_image', Image, queue_size=1)
        self.voxel_sub = rospy.Subscriber('/semantic_voxel_map', MarkerArray, self.voxel_callback)
        
        self.bridge = CvBridge()
        
        # 测试参数
        self.test_image_width = 640
        self.test_image_height = 480
        self.current_pattern = 'mixed'
        
        # 统计信息
        self.semantic_voxel_stats = {
            'total_voxels': 0,
            'semantic_voxels': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'label_distribution': {},
            'update_count': 0,
            'last_update_time': time.time()
        }
        
        # 语义标签定义（对应配置文件中的颜色映射）
        self.semantic_labels = {
            1: "建筑物",    # 红色
            2: "植被",      # 绿色  
            3: "道路",      # 蓝色
            4: "车辆",      # 黄色
            5: "行人",      # 洋红色
            6: "天空",      # 青色
            7: "地面",      # 橙色
            8: "标志牌",    # 紫色
            9: "其他物体",  # 粉色
            10: "树干",     # 棕色
        }
        
        rospy.loginfo("语义投影测试器已启动，支持的测试模式：")
        rospy.loginfo("  - regions: 区域分块模式")
        rospy.loginfo("  - stripes: 条纹模式") 
        rospy.loginfo("  - checkerboard: 棋盘模式")
        rospy.loginfo("  - mixed: 混合模式（推荐）")
        rospy.loginfo("  - dense: 密集语义模式")
        rospy.loginfo("  - sparse: 稀疏语义模式")

    def create_test_semantic_image(self, width=640, height=480, pattern='mixed'):
        """创建测试用的语义图像，支持多种模式"""
        semantic_img = np.zeros((height, width), dtype=np.uint8)
        
        if pattern == 'regions':
            # 区域分块模式
            block_w, block_h = width // 3, height // 3
            labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            for i in range(3):
                for j in range(3):
                    x1, x2 = j * block_w, (j + 1) * block_w
                    y1, y2 = i * block_h, (i + 1) * block_h
                    semantic_img[y1:y2, x1:x2] = labels[i * 3 + j]
                    
        elif pattern == 'stripes':
            # 条纹模式
            stripe_height = height // len(self.semantic_labels)
            for i, label in enumerate(self.semantic_labels.keys()):
                y1 = i * stripe_height
                y2 = min((i + 1) * stripe_height, height)
                semantic_img[y1:y2, :] = label
                
        elif pattern == 'checkerboard':
            # 棋盘模式
            block_size = 40
            labels = list(self.semantic_labels.keys())
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    label_idx = ((x // block_size) + (y // block_size)) % len(labels)
                    x2 = min(x + block_size, width)
                    y2 = min(y + block_size, height)
                    semantic_img[y:y2, x:x2] = labels[label_idx]
                    
        elif pattern == 'mixed':
            # 混合模式 - 更真实的场景
            # 天空区域
            semantic_img[0:height//4, :] = 6
            # 建筑物区域
            semantic_img[height//4:height//2, 0:width//2] = 1
            # 植被区域
            semantic_img[height//4:height//2, width//2:] = 2
            # 道路区域
            semantic_img[height//2:3*height//4, :] = 3
            # 地面和其他
            semantic_img[3*height//4:, 0:width//3] = 7
            semantic_img[3*height//4:, width//3:2*width//3] = 4  # 车辆
            semantic_img[3*height//4:, 2*width//3:] = 9  # 其他
            
        elif pattern == 'dense':
            # 密集语义模式 - 每个像素都有语义标签
            labels = list(self.semantic_labels.keys())
            for y in range(height):
                for x in range(width):
                    # 基于位置的伪随机标签分配
                    label_idx = (x + y * 3 + x * y // 10) % len(labels)
                    semantic_img[y, x] = labels[label_idx]
                    
        elif pattern == 'sparse':
            # 稀疏语义模式 - 只有部分像素有语义
            labels = list(self.semantic_labels.keys())
            for y in range(0, height, 20):
                for x in range(0, width, 20):
                    label_idx = ((x // 20) + (y // 20)) % len(labels)
                    # 创建小的语义区域
                    x2 = min(x + 10, width)
                    y2 = min(y + 10, height)
                    semantic_img[y:y2, x:x2] = labels[label_idx]
        
        return semantic_img

    def publish_test_semantic_image(self, pattern='mixed'):
        """发布测试语义图像"""
        semantic_img = self.create_test_semantic_image(
            self.test_image_width, 
            self.test_image_height, 
            pattern
        )
        
        # 统计语义图像信息
        unique_labels, counts = np.unique(semantic_img[semantic_img > 0], return_counts=True)
        total_semantic_pixels = np.sum(counts)
        total_pixels = self.test_image_width * self.test_image_height
        
        rospy.loginfo(f"发布{pattern}模式语义图像:")
        rospy.loginfo(f"  图像尺寸: {self.test_image_width}x{self.test_image_height}")
        rospy.loginfo(f"  语义像素: {total_semantic_pixels}/{total_pixels} ({100*total_semantic_pixels/total_pixels:.1f}%)")
        rospy.loginfo(f"  语义标签数: {len(unique_labels)}")
        
        for label, count in zip(unique_labels, counts):
            label_name = self.semantic_labels.get(label, f"未知({label})")
            percentage = 100 * count / total_pixels
            rospy.loginfo(f"    标签{label}({label_name}): {count}像素 ({percentage:.1f}%)")
        
        # 转换为ROS图像消息并发布
        try:
            image_msg = self.bridge.cv2_to_imgmsg(semantic_img, encoding='mono8')
            image_msg.header.stamp = rospy.Time.now()
            image_msg.header.frame_id = 'camera_init'
            
            self.semantic_pub.publish(image_msg)
            rospy.loginfo(f"成功发布{pattern}模式语义图像")
            
        except Exception as e:
            rospy.logerr(f"发布语义图像失败: {e}")

    def voxel_callback(self, msg):
        """处理语义体素地图消息"""
        current_time = time.time()
        self.semantic_voxel_stats['update_count'] += 1
        self.semantic_voxel_stats['last_update_time'] = current_time
        
        # 解析体素信息
        total_markers = len(msg.markers)
        semantic_markers = 0
        high_conf_markers = 0
        medium_conf_markers = 0
        low_conf_markers = 0
        
        label_count = {}
        
        for marker in msg.markers:
            # 根据透明度和颜色判断语义类型
            alpha = marker.color.a
            r, g, b = marker.color.r, marker.color.g, marker.color.b
            
            # 检查是否为语义体素（基于颜色特征）
            is_semantic = False
            detected_label = 0
            
            # 简单的颜色到标签映射（基于预定义的语义颜色）
            color_to_label = {
                (1.0, 0.0, 0.0): 1,  # 红色 - 建筑物
                (0.0, 1.0, 0.0): 2,  # 绿色 - 植被
                (0.0, 0.0, 1.0): 3,  # 蓝色 - 道路
                (1.0, 1.0, 0.0): 4,  # 黄色 - 车辆
                (1.0, 0.0, 1.0): 5,  # 洋红 - 行人
                (0.0, 1.0, 1.0): 6,  # 青色 - 天空
                (1.0, 0.5, 0.0): 7,  # 橙色 - 地面
            }
            
            # 查找最接近的颜色
            min_dist = float('inf')
            for color, label in color_to_label.items():
                dist = abs(r - color[0]) + abs(g - color[1]) + abs(b - color[2])
                if dist < min_dist and dist < 0.3:  # 颜色容差
                    min_dist = dist
                    detected_label = label
                    is_semantic = True
            
            if is_semantic:
                semantic_markers += 1
                label_count[detected_label] = label_count.get(detected_label, 0) + 1
                
                # 根据透明度判断置信度级别
                if alpha > 0.85:
                    high_conf_markers += 1
                elif alpha > 0.65:
                    medium_conf_markers += 1
                else:
                    low_conf_markers += 1
        
        # 更新统计信息
        self.semantic_voxel_stats.update({
            'total_voxels': total_markers,
            'semantic_voxels': semantic_markers,
            'high_confidence': high_conf_markers,
            'medium_confidence': medium_conf_markers,
            'low_confidence': low_conf_markers,
            'label_distribution': label_count
        })
        
        # 输出详细统计
        semantic_ratio = (semantic_markers / max(1, total_markers)) * 100
        rospy.loginfo(f"语义体素统计 (更新#{self.semantic_voxel_stats['update_count']}):")
        rospy.loginfo(f"  总体素数: {total_markers}")
        rospy.loginfo(f"  语义体素: {semantic_markers} ({semantic_ratio:.1f}%)")
        rospy.loginfo(f"    高置信度: {high_conf_markers}")
        rospy.loginfo(f"    中置信度: {medium_conf_markers}")
        rospy.loginfo(f"    低置信度: {low_conf_markers}")
        
        if label_count:
            rospy.loginfo("  标签分布:")
            for label, count in sorted(label_count.items()):
                label_name = self.semantic_labels.get(label, f"未知({label})")
                rospy.loginfo(f"    标签{label}({label_name}): {count}个体素")

    def run_test_sequence(self):
        """运行测试序列"""
        test_patterns = ['mixed', 'regions', 'stripes', 'checkerboard', 'dense', 'sparse']
        
        rospy.loginfo("开始语义投影测试序列...")
        
        for i, pattern in enumerate(test_patterns):
            rospy.loginfo(f"\n========== 测试{i+1}/{len(test_patterns)}: {pattern}模式 ==========")
            
            # 发布测试图像
            self.current_pattern = pattern
            self.publish_test_semantic_image(pattern)
            
            # 等待语义投影处理
            rospy.loginfo("等待语义投影处理...")
            time.sleep(3.0)
            
            # 输出当前统计
            self.print_current_stats()
            
            rospy.loginfo(f"{pattern}模式测试完成\n")
            time.sleep(2.0)
        
        rospy.loginfo("测试序列完成！")

    def print_current_stats(self):
        """打印当前统计信息"""
        stats = self.semantic_voxel_stats
        rospy.loginfo("当前累积统计:")
        rospy.loginfo(f"  更新次数: {stats['update_count']}")
        rospy.loginfo(f"  最新总体素: {stats['total_voxels']}")
        rospy.loginfo(f"  最新语义体素: {stats['semantic_voxels']}")
        if stats['total_voxels'] > 0:
            ratio = (stats['semantic_voxels'] / stats['total_voxels']) * 100
            rospy.loginfo(f"  语义覆盖率: {ratio:.1f}%")

    def monitor_continuous(self):
        """连续监控模式"""
        rospy.loginfo("进入连续监控模式，每10秒发布一次测试图像...")
        patterns = ['mixed', 'dense', 'regions', 'sparse']
        pattern_idx = 0
        
        rate = rospy.Rate(0.1)  # 10秒间隔
        
        while not rospy.is_shutdown():
            pattern = patterns[pattern_idx % len(patterns)]
            rospy.loginfo(f"发布{pattern}模式测试图像...")
            
            self.publish_test_semantic_image(pattern)
            pattern_idx += 1
            
            rate.sleep()

def main():
    try:
        tester = SemanticProjectionTester()
        
        # 等待ROS初始化
        rospy.sleep(2.0)
        
        rospy.loginfo("语义投影测试选项:")
        rospy.loginfo("1. 运行完整测试序列")
        rospy.loginfo("2. 连续监控模式")
        rospy.loginfo("3. 单次测试特定模式")
        
        # 这里可以根据需要选择测试模式
        # 默认运行完整测试序列
        choice = rospy.get_param('~test_mode', 'sequence')
        
        if choice == 'sequence':
            tester.run_test_sequence()
        elif choice == 'monitor':
            tester.monitor_continuous()
        else:
            # 单次测试
            pattern = rospy.get_param('~pattern', 'mixed')
            tester.publish_test_semantic_image(pattern)
            rospy.loginfo(f"发布{pattern}模式测试图像完成")
            
        # 保持节点运行以接收回调
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("测试被中断")
    except Exception as e:
        rospy.logerr(f"测试过程中出错: {e}")

if __name__ == '__main__':
    main() 