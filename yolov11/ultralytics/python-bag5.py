#!/usr/bin/env python3
import cv2
import numpy as np
import time
from ultralytics import YOLO
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from concurrent.futures import ThreadPoolExecutor

PROCESSING_TIMEOUT_SEC = 0.15  # 处理单帧图像的最大允许时间（秒）
LOG_DETECTIONS = True  # 是否在终端低频打印识别到的语义类别

# 动态类别（用于LIO动态剔除）
DYNAMIC_CLASS_IDS = {
    # 掩码值 0 保留给背景；person 不能使用模型中的类别 ID 0。
    "person": 1,
    "car": 2,
    "bicycle": 3,
    "motorcycle": 4,
    "bus": 5,
    "truck": 6,
}

# 静态类别（用于语义回环验证）
STATIC_CLASS_IDS = {
    "table": 10,
    "net": 11,
    "monitor": 12,
    "chair": 13,
    "door": 14,
    "traffic_sign": 15,
    "building": 16,

}

STATIC_VIS_COLORS = {
    10: (0, 255, 255),   # table
    11: (0, 140, 255),   # net
    12: (255, 0, 0),     # monitor
    13: (0, 255, 0),     # chair
    14: (255, 255, 0),   # door
    15: (0, 0, 255),     # traffic_sign
    16: (255, 0, 255),   # building
}

class ImageProcessorNode:
    def __init__(self, topic, model_path):
        # 初始化ROS订阅者和发布者
        self.image_sub = rospy.Subscriber(topic, Image, self.image_callback, queue_size=1, buff_size=2**24)# 收到图像后自动调用image_callback函数处理图像
        self.image_pub = rospy.Publisher("/setic/image_raw", Image, queue_size=10)
        self.static_mask_pub = rospy.Publisher("/semantic/static_mask", Image, queue_size=10)
        self.static_vis_pub = rospy.Publisher("/semantic/static_vis", Image, queue_size=10)

        # 初始化模型和其他组件
        self.model = YOLO(model_path).cuda()
        self.bridge = CvBridge()

        # 类别统计
        self.class_detection_count = {}

        # 使用线程池加速处理
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 统计处理时间
        self.total_frames = 0
        self.total_time = 0.0

    def process_frame(self, frame):
        """处理单帧图像"""
        start_time = time.time()
        # 创建空白语义图
        dynamic_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        static_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        frame_detections = []

        try:
            # 进行检测和分割。conf=0.65：置信度低于 0.65 的目标不要，iou=0.4：NMS 去重阈值
            results = self.model.predict(frame, conf=0.65, iou=0.4, verbose=False)

            # 检查是否超时
            if time.time() - start_time > PROCESSING_TIMEOUT_SEC:
                raise TimeoutError("Processing timeout")

            result = results[0]
            # 把 YOLO 输出转成 NumPy
            if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
                # detections为每个目标的框，包含 [x1, y1, x2, y2,  confidence, class_id]，masks 是对应的分割掩码
                detections = result.boxes.data.cpu().numpy()
                # masks：每个目标的分割区域
                masks = result.masks.data.cpu().numpy()

                # 按置信度升序排列，高置信度结果最后覆盖低置信度，减少冲突。比如人和车 mask 重叠时，高置信度那个会赢
                sorted_indices = np.argsort(detections[:, 4])
                names = result.names
                for idx in sorted_indices:
                    if time.time() - start_time > PROCESSING_TIMEOUT_SEC:
                        raise TimeoutError("Processing timeout")
                    if idx >= len(masks):
                        continue

                    det = detections[idx]
                    # det[4]  confidence, det[5]  class_id
                    conf = float(det[4])
                    if conf < 0.65:
                        continue

                    # 把 YOLO 类别转成动态/静态语义 ID
                    cls_id = int(det[5])
                    if isinstance(names, dict):
                        cls_name = names.get(cls_id, str(cls_id))
                    else:
                        cls_name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)

                    # 保留模型类别 ID，仅将 COCO 的 tv 显示并记录为 monitor。
                    if cls_name == "tv":
                        cls_name = "monitor"

                    resized_mask = cv2.resize(
                        masks[idx],
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    selected = resized_mask > 0.5

                    dynamic_id = DYNAMIC_CLASS_IDS.get(cls_name, 0)
                    static_id = STATIC_CLASS_IDS.get(cls_name, 0)
                    if dynamic_id > 0:
                        dynamic_mask[selected] = dynamic_id
                        self.class_detection_count[cls_name] = self.class_detection_count.get(cls_name, 0) + 1
                        frame_detections.append((cls_name, dynamic_id, conf))
                    elif static_id > 0:
                        static_mask[selected] = static_id
                        self.class_detection_count[cls_name] = self.class_detection_count.get(cls_name, 0) + 1
                        frame_detections.append((cls_name, static_id, conf))

            if LOG_DETECTIONS and frame_detections:
                detection_text = ", ".join(
                    f"{name}(ID={semantic_id}, conf={conf:.2f})"
                    for name, semantic_id, conf in frame_detections
                )
                rospy.loginfo_throttle(1.0, f"当前识别: {detection_text}")

        except TimeoutError as e:
            # rospy.logwarn(f"Processing interrupted: {str(e)}")
            dynamic_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            static_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        except Exception as e:
            # rospy.logerr(f"Error processing frame: {str(e)}")
            dynamic_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            static_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # 检查处理时间
        processing_time = time.time() - start_time
        self.total_frames += 1
        self.total_time += processing_time

        if processing_time > PROCESSING_TIMEOUT_SEC:
            # rospy.logwarn(f"帧处理时间 {processing_time:.2f}秒 超过 {PROCESSING_TIMEOUT_SEC}秒")
            dynamic_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            static_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        else:
            pass

        return dynamic_mask, static_mask, processing_time

    def build_static_vis(self, static_mask):
        vis = np.zeros((static_mask.shape[0], static_mask.shape[1], 3), dtype=np.uint8)
        for cls_id, color in STATIC_VIS_COLORS.items():
            vis[static_mask == cls_id] = color
        return vis

    def image_callback(self, msg):
        """ROS图像回调函数"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            # rospy.logerr(f"Error converting image: {str(e)}")
            return

        # 使用线程池处理图像，调用process_frame函数，并获取处理结果和时间
        future = self.executor.submit(self.process_frame, frame)
        dynamic_mask, static_mask, processing_time = future.result()# 等待处理完成并获取结果

        # 如果处理时间超过阈值，输出空白帧
        if processing_time > PROCESSING_TIMEOUT_SEC:
            # rospy.logwarn(f"帧处理时间 {processing_time:.2f}秒 超过 {PROCESSING_TIMEOUT_SEC}秒。输出空白帧。")
            dynamic_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            static_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # 发布处理后的图像
        try:
            dynamic_msg = self.bridge.cv2_to_imgmsg(dynamic_mask, encoding="mono8")
            dynamic_msg.header.stamp = msg.header.stamp
            self.image_pub.publish(dynamic_msg)

            static_mask_msg = self.bridge.cv2_to_imgmsg(static_mask, encoding="mono8")
            static_mask_msg.header.stamp = msg.header.stamp
            self.static_mask_pub.publish(static_mask_msg)

            static_vis = self.build_static_vis(static_mask)
            static_vis_msg = self.bridge.cv2_to_imgmsg(static_vis, encoding="bgr8")
            static_vis_msg.header.stamp = msg.header.stamp
            self.static_vis_pub.publish(static_vis_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing image: {str(e)}")

    def print_statistics(self):
        """打印统计信息"""
        if self.total_frames > 0:
            avg_time = self.total_time / self.total_frames
            rospy.loginfo(f"处理了 {self.total_frames} 帧。平均处理时间：{avg_time:.2f}秒")
            
            # 打印类别检测统计
            rospy.loginfo("类别检测统计：")
            for cls_name, count in sorted(self.class_detection_count.items(), key=lambda x: x[1], reverse=True):
                semantic_id = DYNAMIC_CLASS_IDS.get(cls_name, STATIC_CLASS_IDS.get(cls_name, 0))
                rospy.loginfo(f"  {cls_name}(ID={semantic_id}): {count} 次检测")
        else:
            rospy.loginfo("未处理任何帧")

if __name__ == "__main__":
    # 初始化ROS节点，节点名称为'semantic_image_processor'
    rospy.init_node('semantic_image_processor')

    TOPIC = "/camera/color/image_raw"
    # MODEL_PATH = "/home/liu/code/LIVO2/ws_livo2/yolov11/models/yolo11x-seg.pt"
    MODEL_PATH = "/home/liu/code/LIVO2/ws_livo2/yolov11/models/Dynamic03_nomal_yolo11m-seg_best.pt"

    processor_node = ImageProcessorNode(TOPIC, MODEL_PATH)
    rospy.on_shutdown(processor_node.print_statistics)

    rospy.loginfo("Semantic Image Processor Node is running...")
    # 序就一直等 ROS 消息。只要 /camera/color/image_raw 来一帧图像，就会自动调用 image_callback 函数处理图像并发布结果。
    rospy.spin()
