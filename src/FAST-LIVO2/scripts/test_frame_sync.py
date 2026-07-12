#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
帧同步测试脚本
用于测试语义图像帧与RGB图像帧的同步效果
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import threading

class FrameSyncTester:
    def __init__(self):
        rospy.init_node('frame_sync_tester', anonymous=True)
        
        # ROS Publishers
        self.rgb_pub = rospy.Publisher('/left_camera/image', Image, queue_size=1)
        self.setic_pub = rospy.Publisher('/setic/image_raw', Image, queue_size=1)
        
        self.bridge = CvBridge()
        
        # 测试参数
        self.image_width = 640
        self.image_height = 480
        self.publish_rate = 30  # Hz
        self.time_offset_setic = 0.0  # 语义图像的时间偏移
        
        # 统计信息
        self.rgb_frame_count = 0
        self.setic_frame_count = 0
        self.start_time = None
        
        # 同步测试模式
        self.sync_modes = {
            'perfect': 0.0,      # 完美同步
            'slight_delay': 0.01, # 10ms延迟
            'medium_delay': 0.02, # 20ms延迟
            'large_delay': 0.05,  # 50ms延迟
            'random': 'random'    # 随机延迟
        }
        self.current_mode = 'perfect'
        
    def create_test_rgb_image(self, frame_id):
        """创建测试用的RGB图像"""
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # 创建带有帧ID的彩色图像
        color_value = (frame_id * 10) % 255
        img[:, :] = [color_value, 255 - color_value, (color_value * 2) % 255]
        
        # 添加文本标识
        text = f"RGB Frame {frame_id}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 添加时间戳
        timestamp_text = f"T: {rospy.Time.now().to_sec():.6f}"
        cv2.putText(img, timestamp_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
        
    def create_test_setic_image(self, frame_id):
        """创建测试用的语义图像"""
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # 创建语义分割样式的图像
        # 区域1：建筑物（红色）
        img[0:150, 0:200] = [0, 0, 255]
        
        # 区域2：植被（绿色）
        img[150:300, 0:200] = [0, 255, 0]
        
        # 区域3：道路（蓝色）
        img[300:480, 0:200] = [255, 0, 0]
        
        # 区域4：车辆（黄色）
        img[0:150, 200:400] = [0, 255, 255]
        
        # 区域5：天空（青色）
        img[150:300, 200:400] = [255, 255, 0]
        
        # 区域6：其他（洋红）
        img[300:480, 200:400] = [255, 0, 255]
        
        # 右侧区域渐变效果
        for i in range(400, 640):
            for j in range(480):
                intensity = int((i - 400) / 240.0 * 255)
                label = (frame_id + j // 60) % 10 + 1
                if label == 1:
                    img[j, i] = [0, 0, intensity]      # 建筑物
                elif label == 2:
                    img[j, i] = [0, intensity, 0]      # 植被
                elif label == 3:
                    img[j, i] = [intensity, 0, 0]      # 道路
                else:
                    img[j, i] = [intensity, intensity, 0] # 其他
        
        # 添加文本标识
        text = f"SETIC Frame {frame_id}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 添加时间戳
        timestamp_text = f"T: {rospy.Time.now().to_sec():.6f}"
        cv2.putText(img, timestamp_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def get_sync_delay(self):
        """根据当前模式获取同步延迟"""
        if self.current_mode == 'random':
            return np.random.uniform(-0.03, 0.03)  # -30ms到+30ms的随机延迟
        else:
            return self.sync_modes.get(self.current_mode, 0.0)
    
    def publish_rgb_frame(self, frame_id):
        """发布RGB图像帧"""
        img = self.create_test_rgb_image(frame_id)
        
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = "camera_init"
            
            self.rgb_pub.publish(img_msg)
            self.rgb_frame_count += 1
            
            rospy.loginfo(f"[SYNC-TEST] Published RGB frame {frame_id} at {img_msg.header.stamp.to_sec():.6f}")
            
        except Exception as e:
            rospy.logerr(f"[SYNC-TEST] Error publishing RGB frame: {e}")
    
    def publish_setic_frame(self, frame_id, delay=0.0):
        """发布语义图像帧（可能带延迟）"""
        if delay > 0:
            threading.Timer(delay, self._publish_setic_frame_delayed, [frame_id]).start()
        else:
            self._publish_setic_frame_delayed(frame_id)
    
    def _publish_setic_frame_delayed(self, frame_id):
        """延迟发布语义图像帧的实际实现"""
        img = self.create_test_setic_image(frame_id)
        
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = "camera_init"
            
            self.setic_pub.publish(img_msg)
            self.setic_frame_count += 1
            
            rospy.loginfo(f"[SYNC-TEST] Published SETIC frame {frame_id} at {img_msg.header.stamp.to_sec():.6f}")
            
        except Exception as e:
            rospy.logerr(f"[SYNC-TEST] Error publishing SETIC frame: {e}")
    
    def run_sync_test(self, duration=60, mode='perfect'):
        """运行同步测试"""
        self.current_mode = mode
        self.start_time = rospy.Time.now()
        
        rospy.loginfo(f"[SYNC-TEST] Starting frame sync test in '{mode}' mode for {duration} seconds")
        rospy.loginfo(f"[SYNC-TEST] Expected delay: {self.sync_modes.get(mode, 'variable')}")
        
        rate = rospy.Rate(self.publish_rate)
        frame_id = 0
        
        try:
            while not rospy.is_shutdown() and frame_id < duration * self.publish_rate:
                # 发布RGB帧
                self.publish_rgb_frame(frame_id)
                
                # 根据模式发布语义帧
                delay = self.get_sync_delay()
                self.publish_setic_frame(frame_id, max(0, delay))
                
                frame_id += 1
                rate.sleep()
                
                # 每100帧打印一次统计
                if frame_id % 100 == 0:
                    self.print_statistics()
                    
        except KeyboardInterrupt:
            rospy.loginfo("[SYNC-TEST] Test interrupted by user")
        
        self.print_final_statistics()
    
    def print_statistics(self):
        """打印统计信息"""
        if self.start_time is None:
            return
            
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        rgb_rate = self.rgb_frame_count / elapsed if elapsed > 0 else 0
        setic_rate = self.setic_frame_count / elapsed if elapsed > 0 else 0
        
        rospy.loginfo(f"[SYNC-TEST] Statistics after {elapsed:.1f}s:")
        rospy.loginfo(f"  RGB frames: {self.rgb_frame_count} ({rgb_rate:.1f} Hz)")
        rospy.loginfo(f"  SETIC frames: {self.setic_frame_count} ({setic_rate:.1f} Hz)")
        rospy.loginfo(f"  Mode: {self.current_mode}")
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        rospy.loginfo("[SYNC-TEST] Final Test Results:")
        rospy.loginfo("="*50)
        self.print_statistics()
        rospy.loginfo(f"  Frame pairs generated: {min(self.rgb_frame_count, self.setic_frame_count)}")
        rospy.loginfo(f"  Frame difference: {abs(self.rgb_frame_count - self.setic_frame_count)}")
        rospy.loginfo("="*50)

def main():
    tester = FrameSyncTester()
    
    try:
        # 可以运行不同的测试模式
        test_modes = ['perfect', 'slight_delay', 'medium_delay', 'large_delay', 'random']
        
        rospy.loginfo("[SYNC-TEST] Available test modes:")
        for i, mode in enumerate(test_modes):
            rospy.loginfo(f"  {i+1}. {mode}")
        
        # 运行完美同步测试
        tester.run_sync_test(duration=30, mode='perfect')
        
        rospy.sleep(2)
        
        # 运行轻微延迟测试
        tester.run_sync_test(duration=30, mode='slight_delay')
        
    except rospy.ROSInterruptException:
        rospy.loginfo("[SYNC-TEST] Test terminated")
    except Exception as e:
        rospy.logerr(f"[SYNC-TEST] Test failed: {e}")

if __name__ == '__main__':
    main() 