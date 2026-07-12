#!/usr/bin/env python3
"""
行人车辆检测程序
基于 YOLOv11 模型，专门用于检测和跟踪行人、车辆等交通元素
模仿 python-bag5 的结构设计
"""
import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import xxhash
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path

# --------------------- 车辆行人类别颜色映射 ---------------------
# 专注于车辆和行人检测
VEHICLE_PERSON_COLORS = {
    "person": (0, 255, 0),          # 绿色 - 行人
    "car": (0, 0, 255),             # 红色 - 汽车
    "truck": (128, 0, 128),         # 紫色 - 卡车
    "bus": (0, 165, 255),           # 橙色 - 公交车
    "motorcycle": (255, 0, 255),    # 品红 - 摩托车
    "bicycle": (255, 255, 0),       # 青色 - 自行车
    "default": (128, 128, 128)      # 默认灰色
}

class TrafficHashColor:
    """交通场景颜色分配器"""
    def __init__(self):
        self.cache = {}

    def get_color_by_class(self, class_name):
        """根据类别名称获取颜色，确保同一类别显示相同颜色"""
        if class_name in VEHICLE_PERSON_COLORS:
            return VEHICLE_PERSON_COLORS[class_name]
        else:
            # 如果类别不在预定义中，返回默认颜色并记录警告
            print(f"警告: 未定义的类别: {class_name}，使用默认颜色")
            return VEHICLE_PERSON_COLORS["default"]

    def get_color(self, track_id):
        """保留原有功能用于兼容性"""
        if track_id not in self.cache:
            track_str = str(track_id).encode('utf-8')
            hash1 = xxhash.xxh64(track_str).intdigest()
            hash2 = hash(track_id) & 0xFFFFFFFF
            mixed_hash = (hash1 ^ (hash2 << 32)) % 0xFFFFFF
            b, g, r = (mixed_hash >> 16) & 0xFF, (mixed_hash >> 8) & 0xFF, mixed_hash & 0xFF
            self.cache[track_id] = (b, g, r)
        return self.cache[track_id]

class PedestrianVehicleDetector:
    """行人车辆检测器主类"""
    
    def __init__(self, model_path, conf_threshold=0.6, iou_threshold=0.4):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        print("正在初始化行人车辆检测器...")
        
        # 初始化模型
        self.model = YOLO(model_path)
        if hasattr(self.model.model, 'cuda'):
            self.model = self.model.cuda()
        
        # 初始化跟踪器
        self.tracker = DeepSort(max_age=30, n_init=2)
        
        # 颜色分配器
        self.color_allocator = TrafficHashColor()
        
        # 跟踪 ID 和类别的映射表
        self.track_id_to_class = {}
        
        # 类别统计
        self.class_detection_count = {}
        
        # 检测参数
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 使用线程池加速处理
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 统计处理时间
        self.total_frames = 0
        self.total_time = 0.0
        
        print(f"检测器初始化完成")
        print(f"支持的交通类别数量: {len(VEHICLE_PERSON_COLORS)-1}")
        print(f"置信度阈值: {conf_threshold}")
        print(f"IoU阈值: {iou_threshold}")

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

    def process_frame(self, frame):
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            annotated_frame: 标注后的图像帧
            processing_time: 处理时间
            detection_info: 检测信息
        """
        start_time = time.time()
        annotated_frame = frame.copy()
        detection_info = {
            'total_detections': 0,
            'detected_classes': {},
            'tracked_objects': 0
        }

        try:
            # YOLO检测
            results = self.model.predict(
                frame, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold, 
                verbose=False
            )
            
            if not results or not results[0].boxes:
                return annotated_frame, time.time() - start_time, detection_info
            
            detections = results[0].boxes.data.cpu().numpy()
            cls_names = [results[0].names[int(cls_id)] for cls_id in detections[:, 5]]

            # 过滤只保留交通相关类别
            filtered_detections = []
            filtered_cls_names = []
            
            for det, cls_name in zip(detections, cls_names):
                if cls_name in VEHICLE_PERSON_COLORS and det[4] >= self.conf_threshold:
                    filtered_detections.append(det)
                    filtered_cls_names.append(cls_name)
                    
                    # 更新检测统计
                    if cls_name not in detection_info['detected_classes']:
                        detection_info['detected_classes'][cls_name] = 0
                    detection_info['detected_classes'][cls_name] += 1
            
            detections = np.array(filtered_detections) if filtered_detections else np.array([])
            cls_names = filtered_cls_names
            detection_info['total_detections'] = len(detections)
            
            print(f"检测到 {len(detections)} 个交通相关目标")
            
            # DeepSORT跟踪
            tracks_data = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = det
                tracks_data.append(([x1, y1, x2-x1, y2-y1], conf, cls_names[i]))

            tracks = self.tracker.update_tracks(tracks_data, frame=frame)
            detection_info['tracked_objects'] = len([t for t in tracks if t.is_confirmed()])

            # 绘制检测结果
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                cls_name = track.det_class if hasattr(track, 'det_class') else "unknown"

                # 检查并更新类别一致性
                if track_id in self.track_id_to_class:
                    if cls_name != self.track_id_to_class[track_id]:
                        print(f"警告: Track ID {track_id} 类别从 {self.track_id_to_class[track_id]} 变为 {cls_name}")
                    cls_name = self.track_id_to_class[track_id]
                else:
                    self.track_id_to_class[track_id] = cls_name

                # 更新类别统计
                if cls_name not in self.class_detection_count:
                    self.class_detection_count[cls_name] = 0
                self.class_detection_count[cls_name] += 1

                # 获取颜色
                color = self.color_allocator.get_color_by_class(cls_name)
                
                # 获取边界框
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # 找到对应的检测结果获取置信度
                conf = 0.0
                for i, det in enumerate(detections):
                    det_box = det[:4]
                    iou = self.calculate_iou(ltrb, det_box)
                    if iou > 0.3:  # IoU阈值
                        conf = det[4]
                        break

                # 绘制边界框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{cls_name} ID:{track_id} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 背景矩形
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # 文字
                cv2.putText(annotated_frame, label, 
                          (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                          (255, 255, 255), 2)

        except Exception as e:
            print(f"处理帧时出错: {str(e)}")

        # 统计处理时间
        processing_time = time.time() - start_time
        self.total_frames += 1
        self.total_time += processing_time

        return annotated_frame, processing_time, detection_info

    def detect_image(self, image_path, output_path=None):
        """
        检测单张图像
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）
        """
        print(f"正在处理图像: {image_path}")
        
        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"错误: 无法读取图像 {image_path}")
            return
        
        # 处理图像
        annotated_frame, processing_time, detection_info = self.process_frame(frame)
        
        # 显示结果信息
        print(f"处理完成，耗时: {processing_time:.2f}秒")
        print(f"检测到 {detection_info['total_detections']} 个目标")
        print(f"跟踪到 {detection_info['tracked_objects']} 个目标")
        
        if detection_info['detected_classes']:
            print("检测到的类别:")
            for cls_name, count in detection_info['detected_classes'].items():
                color = self.color_allocator.get_color_by_class(cls_name)
                print(f"  {cls_name}: {count} 个 (颜色: RGB{color})")
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"结果已保存到: {output_path}")
        
        # 显示图像
        cv2.imshow('行人车辆检测结果', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_video(self, video_path, output_path=None):
        """
        检测视频
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
        """
        print(f"正在处理视频: {video_path}")
        
        # 打开视频
        if video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))  # 摄像头
        else:
            cap = cv2.VideoCapture(video_path)  # 视频文件
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}FPS, 总帧数: {total_frames}")
        
        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 处理帧
                annotated_frame, processing_time, detection_info = self.process_frame(frame)
                
                # 在图像上显示统计信息
                info_text = f"Frame: {frame_count}/{total_frames} | Detections: {detection_info['total_detections']} | Tracked: {detection_info['tracked_objects']} | Time: {processing_time:.2f}s"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 保存帧
                if writer:
                    writer.write(annotated_frame)
                
                # 显示帧
                cv2.imshow('行人车辆检测', annotated_frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 每100帧打印一次进度
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except KeyboardInterrupt:
            print("用户中断处理")
        
        finally:
            # 清理资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"视频处理完成，共处理 {frame_count} 帧")
            self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        if self.total_frames > 0:
            avg_time = self.total_time / self.total_frames
            print(f"\n=== 处理统计 ===")
            print(f"总处理帧数: {self.total_frames}")
            print(f"总处理时间: {self.total_time:.2f}秒")
            print(f"平均处理时间: {avg_time:.2f}秒/帧")
            print(f"处理速度: {1/avg_time:.1f} FPS")
            
            # 打印类别检测统计
            if self.class_detection_count:
                print(f"\n=== 类别检测统计 ===")
                for cls_name, count in sorted(self.class_detection_count.items(), 
                                            key=lambda x: x[1], reverse=True):
                    color = self.color_allocator.get_color_by_class(cls_name)
                    print(f"  {cls_name}: {count} 次检测 (颜色: RGB{color})")
        else:
            print("未处理任何帧")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='行人车辆检测程序')
    parser.add_argument('--model', '-m', type=str, 
                       default='/home/ao/yolov11/models/yolo11x.pt',
                       help='YOLO模型路径')
    parser.add_argument('--source', '-s', type=str, required=True,
                       help='输入源（图像/视频路径或摄像头编号）')
    parser.add_argument('--output', '-o', type=str,
                       help='输出路径')
    parser.add_argument('--conf', type=float, default=0.6,
                       help='置信度阈值 (默认: 0.6)')
    parser.add_argument('--iou', type=float, default=0.4,
                       help='IoU阈值 (默认: 0.4)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    print("=== 行人车辆检测程序 ===")
    print(f"模型路径: {args.model}")
    print(f"输入源: {args.source}")
    print(f"置信度阈值: {args.conf}")
    print(f"IoU阈值: {args.iou}")
    
    # 显示支持的类别
    print(f"\n支持的交通类别 ({len(VEHICLE_PERSON_COLORS)-1} 种):")
    priority_classes = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
    other_classes = [cls for cls in VEHICLE_PERSON_COLORS.keys() 
                    if cls not in priority_classes and cls != "default"]
    
    print("  主要交通类别:")
    for cls_name in priority_classes:
        if cls_name in VEHICLE_PERSON_COLORS:
            color = VEHICLE_PERSON_COLORS[cls_name]
            print(f"    {cls_name}: RGB{color}")
    
    print("  其他支持类别:")
    for cls_name in other_classes:
        color = VEHICLE_PERSON_COLORS[cls_name]
        print(f"    {cls_name}: RGB{color}")
    
    # 初始化检测器
    detector = PedestrianVehicleDetector(
        args.model, 
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 判断输入类型并处理
    source = args.source
    
    if source.isdigit():
        # 摄像头输入
        print(f"\n开始摄像头检测 (摄像头 {source})")
        detector.detect_video(source, args.output)
    elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # 图像文件
        print(f"\n开始图像检测")
        detector.detect_image(source, args.output)
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        # 视频文件
        print(f"\n开始视频检测")
        detector.detect_video(source, args.output)
    else:
        print(f"错误: 不支持的输入格式: {source}")
        print("支持的格式:")
        print("  图像: .jpg, .jpeg, .png, .bmp, .tiff")
        print("  视频: .mp4, .avi, .mov, .mkv, .flv") 
        print("  摄像头: 数字 (例如: 0)")

if __name__ == "__main__":
    main() 