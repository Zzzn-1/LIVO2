#!/usr/bin/env python3
import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import xxhash
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

# --------------------- 类别颜色映射 ---------------------
CLASS_COLORS = {
    "chair":         (0, 0, 255),       # 红色
    "sofa":          (0, 165, 255),     # 橙色
    "table":         (204, 204, 0),     # 金色
    "bed":           (147, 20, 255),    # 深粉色
    "refrigerator":  (255, 0, 0),       # 蓝色
    "microwave":     (255, 255, 0),     # 青色
    "oven":          (0, 255, 255),     # 黄色
    "sink":          (128, 0, 128),     # 紫色
    # ---------- 电子设备类 ----------
    "tv":            (0, 255, 0),       # 绿色
    "laptop":        (192, 192, 192),   # 银色
    "cell phone":    (255, 0, 255),     # 品红
    "keyboard":      (30, 144, 255),    # 道奇蓝
    # ---------- 厨房用品类 ----------
    "bottle":        (0, 128, 128),     # 橄榄绿
    "cup":           (64, 224, 208),    # 青绿色
    "kettle":        (0, 255, 127),     # 春绿色
    "knife":         (255, 105, 180),   # 热粉色
    # ---------- 交通工具类 ----------
    "car":           (255, 215, 0),     # 金橙色
    "bicycle":       (34, 139, 34),     # 森林绿
    "motorcycle":    (130, 0, 75),      # 深紫色
    "bus":           (0, 69, 255),      # 深橙色
    # ---------- 动植物类 ----------
    "person":        (240, 230, 140),   # 卡其色
    "dog":           (139, 69, 19),     # 马鞍棕
    "cat":           (255, 192, 203),   # 粉红色
    "plant":         (34, 139, 34),     # 森林绿
    "sheep":          (0, 100, 0),       # 深绿色
    # ---------- 其他常用类 ----------
    "bench":         (210, 105, 30),    # 巧克力色
    "dining table":  (240, 128, 128),   # 亮珊瑚色
    "potted plant":  (138, 43, 226),    # 蓝紫色
    "stop sign":      (255, 140, 0),     # 深橙色
    "sandwich":       (75, 0, 130),       # 靛蓝色
    # ---------- 室外物体识别 ----------
    "lamp":       (43, 54, 130),
    "tree":       (123, 255, 130)
}

class DualHashColor:
    def __init__(self):
        self.cache = {}

    def get_color(self, track_id):
        if track_id not in self.cache:
            track_str = str(track_id).encode('utf-8')
            hash1 = xxhash.xxh64(track_str).intdigest()
            hash2 = hash(track_id) & 0xFFFFFFFF
            mixed_hash = (hash1 ^ (hash2 << 32)) % 0xFFFFFF
            b, g, r = (mixed_hash >> 16) & 0xFF, (mixed_hash >> 8) & 0xFF, mixed_hash & 0xFF
            self.cache[track_id] = (b, g, r)
        return self.cache[track_id]

class ImageProcessorNode:
    def __init__(self, topic, model_path):
        # 初始化ROS订阅者和发布者
        self.image_sub = rospy.Subscriber(topic, Image, self.image_callback)
        self.image_pub = rospy.Publisher("/setic/image_raw", Image, queue_size=10)

        # 初始化模型和其他组件
        self.model = YOLO(model_path).cuda()
        self.tracker = DeepSort(max_age=30)
        self.color_allocator = DualHashColor()
        self.bridge = CvBridge()

        # 跟踪 ID 和类别的映射表
        self.track_id_to_class = {}

    def calculate_iou(self, box1, box2):
        """计算两个矩形框的 IOU"""
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    
if __name__ == "__main__":
    rospy.init_node('python_bag5')

    # rospy.init_node('semantic_image_processor')

    # 配置参数（根据实际情况修改）
    TOPIC = "/left_camera/image"
    MODEL_PATH = "/home/ao/yolov11/models/yolo11x-seg.pt"

    processor_node = ImageProcessorNode(TOPIC, MODEL_PATH)

    rospy.loginfo("Semantic Image Processor Node is running...")
    rospy.spin()