#!/usr/bin/env python3
"""
Ultralytics YOLOv11 行人车辆检测跟踪器
基于官方 ultralytics 框架的行人车辆检测+跟踪+计数实现
集成ByteTrack跟踪算法，实现目标计数功能

Example:
    ```python
    from detect_pedestrian_vehicle_tracker import PedestrianVehicleTracker
    
    tracker = PedestrianVehicleTracker("yolo11n.pt")
    results = tracker.track_video("video.mp4", save=True)
    ```

Dependencies:
    - ultralytics
    - opencv-python
    - numpy
    - lap (用于线性分配)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict, deque
import time

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER, ROOT, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.trackers.utils import TrackState


class CountingLine:
    """计数线类 - 用于定义车辆行人通过的计数线"""
    
    def __init__(self, 
                 start_point: Tuple[int, int], 
                 end_point: Tuple[int, int],
                 name: str = "default",
                 direction: str = "both"):
        """
        初始化计数线
        
        Args:
            start_point: 起始点坐标 (x, y)
            end_point: 结束点坐标 (x, y) 
            name: 计数线名称
            direction: 计数方向 ("up", "down", "left", "right", "both")
        """
        self.start = start_point
        self.end = end_point
        self.name = name
        self.direction = direction
        
        # 计算线的方向向量
        self.vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        self.length = np.sqrt(self.vector[0]**2 + self.vector[1]**2)
        
        # 统计计数
        self.counts = defaultdict(int)  # 按类别统计
        self.total_count = 0
        
    def is_crossed(self, prev_center: Tuple[int, int], curr_center: Tuple[int, int]) -> bool:
        """
        检测目标是否穿越计数线
        
        Args:
            prev_center: 上一帧中心点
            curr_center: 当前帧中心点
            
        Returns:
            bool: 是否穿越计数线
        """
        # 使用线段相交算法
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        return intersect(prev_center, curr_center, self.start, self.end)
    
    def get_crossing_direction(self, prev_center: Tuple[int, int], curr_center: Tuple[int, int]) -> str:
        """获取穿越方向"""
        if not self.is_crossed(prev_center, curr_center):
            return "none"
            
        # 计算垂直向量
        perp_vector = (-self.vector[1], self.vector[0])
        
        # 计算移动向量
        move_vector = (curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
        
        # 计算点积判断方向
        dot_product = move_vector[0] * perp_vector[0] + move_vector[1] * perp_vector[1]
        
        return "forward" if dot_product > 0 else "backward"
    
    def draw(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """在图像上绘制计数线"""
        cv2.line(image, self.start, self.end, color, 3)
        cv2.putText(image, f"{self.name}: {self.total_count}", 
                   (self.start[0], self.start[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image


class PedestrianVehicleTracker:
    """
    行人车辆检测跟踪器类，基于 YOLOv11 + ByteTracker
    
    专门用于检测、跟踪和计数行人和车辆目标的检测器，使用 ultralytics 框架标准接口。
    支持图像和视频输入，提供灵活的配置选项和计数功能。
    
    Attributes:
        model (YOLO): YOLOv11 检测模型实例
        tracker: ByteTracker跟踪器实例
        names (Dict): 类别名称映射
        device (str): 推理设备 ('cpu' 或 'cuda')
        target_classes (List[str]): 目标检测类别列表
        class_colors (Dict[str, tuple]): 类别颜色映射
        counting_lines (List[CountingLine]): 计数线列表
        track_history (Dict): 跟踪历史记录
        
    Examples:
        初始化跟踪器:
        >>> tracker = PedestrianVehicleTracker("yolo11n.pt")
        
        添加计数线:
        >>> tracker.add_counting_line((100, 300), (500, 300), "main_line")
        
        跟踪视频:
        >>> results = tracker.track_video("video.mp4", output="output.mp4")
    """
    
    # 目标检测类别 - 专注于行人和车辆
    TARGET_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    
    # 类别颜色映射 (BGR格式，适配OpenCV)
    CLASS_COLORS = {
        'person': (0, 255, 0),      # 绿色
        'bicycle': (255, 255, 0),   # 青色  
        'car': (0, 0, 255),         # 红色
        'motorcycle': (255, 0, 255), # 品红
        'bus': (0, 165, 255),       # 橙色
        'truck': (128, 0, 128),     # 紫色
    }
    
    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        tracker_type: str = "bytetrack",
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化行人车辆检测跟踪器
        
        Args:
            model (str | Path): 模型路径或模型名称
            tracker_type (str): 跟踪器类型 ("bytetrack" 或 "botsort")
            device (str, optional): 推理设备，自动检测如果为None
            verbose (bool): 是否显示详细信息
            
        Raises:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型加载失败
        """
        self.verbose = verbose
        if verbose:
            LOGGER.info(f"{colorstr('Ultralytics YOLOv11')} 🚀 行人车辆跟踪器初始化中...")
        
        try:
            # 加载模型
            self.model = YOLO(model, task='detect', verbose=verbose)
            self.names = self.model.names
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 初始化跟踪器
            self.tracker_type = tracker_type
            self.tracker = None  # 将在第一次跟踪时初始化
            
            # 获取目标类别的索引
            self.target_class_indices = []
            self.target_class_names = []
            
            for cls_name in self.TARGET_CLASSES:
                for idx, name in self.names.items():
                    if name == cls_name:
                        self.target_class_indices.append(idx)
                        self.target_class_names.append(cls_name)
                        break
            
            # 初始化计数相关
            self.counting_lines = []
            self.track_history = defaultdict(lambda: deque(maxlen=50))  # 保存最近50帧的轨迹
            self.tracked_objects = defaultdict(dict)  # 跟踪的对象信息
            self.crossing_records = set()  # 记录已经计数的穿越事件
            
            # 统计信息
            self.detection_stats = defaultdict(int)
            self.tracking_stats = defaultdict(int)
            
            if verbose:
                LOGGER.info(f"跟踪器初始化成功")
                LOGGER.info(f"设备: {self.device}")
                LOGGER.info(f"模型: {model}")
                LOGGER.info(f"跟踪器: {tracker_type}")
                LOGGER.info(f"目标类别: {self.target_class_names}")
                
        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise RuntimeError(f"无法初始化跟踪器: {e}")
    
    def add_counting_line(
        self,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        name: str = None,
        direction: str = "both"
    ) -> None:
        """
        添加计数线
        
        Args:
            start_point: 起始点坐标 (x, y)
            end_point: 结束点坐标 (x, y)
            name: 计数线名称，如果为None则自动生成
            direction: 计数方向
        """
        if name is None:
            name = f"line_{len(self.counting_lines) + 1}"
            
        counting_line = CountingLine(start_point, end_point, name, direction)
        self.counting_lines.append(counting_line)
        
        if self.verbose:
            LOGGER.info(f"添加计数线: {name} from {start_point} to {end_point}")
    
    def track(
        self,
        source: Union[str, Path, np.ndarray, List],
        conf: float = 0.5,
        iou: float = 0.4,
        imgsz: Union[int, List[int]] = 640,
        persist: bool = True,
        tracker: str = None,
        **kwargs
    ):
        """
        对输入源进行跟踪
        
        Args:
            source: 输入源 (图像路径、视频路径、numpy数组等)
            conf (float): 置信度阈值
            iou (float): NMS IoU阈值  
            imgsz (int | List[int]): 推理图像尺寸
            persist (bool): 是否持久化跟踪器
            tracker (str): 跟踪器配置文件
            **kwargs: 其他参数
            
        Returns:
            results: 跟踪结果列表
            
        Examples:
            >>> results = tracker.track("video.mp4", conf=0.6)
            >>> for result in results:
            ...     print(f"跟踪到 {len(result.boxes)} 个目标")
        """
        if self.verbose:
            LOGGER.info(f"开始跟踪: {source}")
        
        # 使用指定的跟踪器类型
        if tracker is None:
            tracker = f"{self.tracker_type}.yaml"
        
        # 执行跟踪，只检测目标类别
        results = self.model.track(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=self.target_class_indices,  # 只检测目标类别
            tracker=tracker,
            persist=persist,
            verbose=self.verbose,
            **kwargs
        )
        
        return results
    
    def track_video(
        self,
        source: Union[str, Path],
        output: Optional[Union[str, Path]] = None,
        conf: float = 0.5,
        iou: float = 0.4,
        imgsz: int = 640,
        show_progress: bool = True,
        save_frames: bool = False,
        **kwargs
    ) -> List:
        """
        跟踪视频文件并添加计数功能
        
        Args:
            source (str | Path): 输入视频路径
            output (str | Path, optional): 输出视频路径
            conf (float): 置信度阈值
            iou (float): NMS IoU阈值
            imgsz (int): 推理图像尺寸
            show_progress (bool): 是否显示进度
            save_frames (bool): 是否保存处理帧
            **kwargs: 其他参数
            
        Returns:
            List: 跟踪结果列表
            
        Examples:
            >>> results = tracker.track_video("input.mp4", "output.mp4", conf=0.6)
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"视频文件不存在: {source}")
            
        if self.verbose:
            LOGGER.info(f"处理视频: {source}")
            
        # 设置输出路径
        if output is None:
            output = source.parent / f"{source.stem}_tracked{source.suffix}"
        else:
            output = Path(output)
            
        # 确保输出目录存在
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化视频读写器
        cap = cv2.VideoCapture(str(source))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果没有设置计数线，默认添加一条水平线
        if not self.counting_lines:
            center_y = height // 2
            self.add_counting_line((0, center_y), (width, center_y), "default_line")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        all_results = []
        
        if self.verbose:
            LOGGER.info(f"开始处理 {total_frames} 帧，分辨率: {width}x{height}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 执行跟踪
            results = self.model.track(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=self.target_class_indices,
                tracker=f"{self.tracker_type}.yaml",
                persist=True,
                verbose=False
            )
            
            if results and len(results) > 0:
                result = results[0]
                annotated_frame = self._annotate_frame(frame, result, frame_idx)
                all_results.append(result)
            else:
                annotated_frame = frame
            
            # 绘制计数线
            for line in self.counting_lines:
                annotated_frame = line.draw(annotated_frame)
            
            # 添加统计信息
            annotated_frame = self._draw_statistics(annotated_frame)
            
            # 写入输出视频
            out.write(annotated_frame)
            
            # 保存单帧（如果需要）
            if save_frames:
                frame_output_dir = output.parent / f"{output.stem}_frames"
                frame_output_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(frame_output_dir / f"frame_{frame_idx:06d}.jpg"), annotated_frame)
            
            frame_idx += 1
            
            # 显示进度
            if show_progress and frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / total_frames) * 100
                LOGGER.info(f"进度: {progress:.1f}% ({frame_idx}/{total_frames}), "
                          f"速度: {fps_current:.1f} FPS")
        
        # 释放资源
        cap.release()
        out.release()
        
        # 输出最终统计
        if self.verbose:
            elapsed = time.time() - start_time
            avg_fps = frame_idx / elapsed if elapsed > 0 else 0
            LOGGER.info(f"视频处理完成！")
            LOGGER.info(f"输出文件: {output}")
            LOGGER.info(f"处理时间: {elapsed:.2f}s, 平均速度: {avg_fps:.1f} FPS")
            self._print_final_statistics()
        
        return all_results
    
    def _annotate_frame(self, frame: np.ndarray, result, frame_idx: int) -> np.ndarray:
        """标注单帧图像"""
        annotated_frame = frame.copy()
        
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            # 处理每个检测结果
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                if cls in self.target_class_indices:
                    class_name = self.names[cls]
                    color = self.CLASS_COLORS.get(class_name, (128, 128, 128))
                    
                    # 计算中心点
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    center = (center_x, center_y)
                    
                    # 更新跟踪历史
                    self.track_history[track_id].append(center)
                    
                    # 检查计数线穿越
                    self._check_line_crossing(track_id, class_name, frame_idx)
                    
                    # 更新跟踪对象信息
                    self.tracked_objects[track_id] = {
                        'class_name': class_name,
                        'center': center,
                        'box': box,
                        'confidence': conf,
                        'frame': frame_idx
                    }
                    
                    # 绘制边框和标签
                    label = f"{class_name} ID:{track_id} {conf:.2f}"
                    cv2.rectangle(annotated_frame, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                color, 2)
                    
                    # 绘制标签背景
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame,
                                (int(box[0]), int(box[1]) - label_size[1] - 10),
                                (int(box[0]) + label_size[0], int(box[1])),
                                color, -1)
                    
                    # 绘制标签文字
                    cv2.putText(annotated_frame, label,
                              (int(box[0]), int(box[1]) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # 绘制轨迹
                    if len(self.track_history[track_id]) > 1:
                        points = list(self.track_history[track_id])
                        for i in range(1, len(points)):
                            cv2.line(annotated_frame, points[i-1], points[i], color, 2)
                    
                    # 绘制中心点
                    cv2.circle(annotated_frame, center, 3, color, -1)
                    
                    # 更新统计
                    self.detection_stats[class_name] += 1
                    self.tracking_stats[track_id] = class_name
        
        return annotated_frame
    
    def _check_line_crossing(self, track_id: int, class_name: str, frame_idx: int) -> None:
        """检查跟踪对象是否穿越计数线"""
        if len(self.track_history[track_id]) < 2:
            return
        
        current_center = self.track_history[track_id][-1]
        previous_center = self.track_history[track_id][-2]
        
        for line in self.counting_lines:
            # 检查是否穿越
            if line.is_crossed(previous_center, current_center):
                # 创建唯一的穿越记录ID
                crossing_id = f"{track_id}_{line.name}_{frame_idx//10}"  # 防止同一目标短时间内重复计数
                
                if crossing_id not in self.crossing_records:
                    # 记录穿越事件
                    self.crossing_records.add(crossing_id)
                    line.counts[class_name] += 1
                    line.total_count += 1
                    
                    if self.verbose:
                        direction = line.get_crossing_direction(previous_center, current_center)
                        LOGGER.info(f"穿越检测: {class_name} ID:{track_id} 穿越 {line.name} "
                                  f"方向:{direction} 帧:{frame_idx}")
    
    def _draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制统计信息"""
        height, width = frame.shape[:2]
        
        # 统计面板背景
        panel_height = 150
        panel_width = 300
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 标题
        cv2.putText(frame, "Detection & Tracking Stats", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 活跃跟踪数
        active_tracks = len([tid for tid, info in self.tracked_objects.items() 
                           if info.get('frame', 0) > len(self.track_history) - 50])
        cv2.putText(frame, f"Active Tracks: {active_tracks}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 各类别统计
        y_offset = 75
        for class_name in self.TARGET_CLASSES:
            count = sum(line.counts[class_name] for line in self.counting_lines)
            if count > 0:
                color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
                cv2.putText(frame, f"{class_name}: {count}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 20
        
        return frame
    
    def _print_final_statistics(self) -> None:
        """打印最终统计信息"""
        LOGGER.info("=" * 50)
        LOGGER.info("最终统计结果")
        LOGGER.info("=" * 50)
        
        # 计数线统计
        for line in self.counting_lines:
            LOGGER.info(f"计数线 '{line.name}': 总计 {line.total_count}")
            for class_name, count in line.counts.items():
                if count > 0:
                    LOGGER.info(f"  {class_name}: {count}")
        
        # 跟踪统计
        total_tracked = len(self.tracking_stats)
        LOGGER.info(f"总跟踪对象数: {total_tracked}")
        
        # 按类别统计跟踪对象
        class_tracking_count = defaultdict(int)
        for track_id, class_name in self.tracking_stats.items():
            class_tracking_count[class_name] += 1
        
        for class_name, count in class_tracking_count.items():
            LOGGER.info(f"  {class_name} 跟踪: {count}")
    
    def get_counting_stats(self) -> Dict:
        """获取计数统计信息"""
        stats = {
            'lines': {},
            'total': defaultdict(int),
            'active_tracks': len(self.tracked_objects)
        }
        
        for line in self.counting_lines:
            stats['lines'][line.name] = {
                'total': line.total_count,
                'by_class': dict(line.counts)
            }
            
            # 累计总数
            for class_name, count in line.counts.items():
                stats['total'][class_name] += count
        
        return stats
    
    def reset_counters(self) -> None:
        """重置所有计数器"""
        for line in self.counting_lines:
            line.counts.clear()
            line.total_count = 0
        
        self.crossing_records.clear()
        self.detection_stats.clear()
        self.tracking_stats.clear()
        
        if self.verbose:
            LOGGER.info("所有计数器已重置")
    
    def save_results(self, output_dir: Union[str, Path], format: str = 'json') -> None:
        """保存跟踪和计数结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_counting_stats()
        
        if format == 'json':
            import json
            with open(output_dir / 'tracking_results.json', 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        
        elif format == 'csv':
            import csv
            with open(output_dir / 'counting_results.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Line', 'Class', 'Count'])
                
                for line_name, line_data in stats['lines'].items():
                    for class_name, count in line_data['by_class'].items():
                        writer.writerow([line_name, class_name, count])
        
        if self.verbose:
            LOGGER.info(f"结果已保存到: {output_dir}")
    
    def __repr__(self) -> str:
        """返回跟踪器的字符串表示"""
        return (f"{self.__class__.__name__}("
                f"model={self.model.ckpt_path}, "
                f"tracker={self.tracker_type}, "
                f"device={self.device}, "
                f"lines={len(self.counting_lines)}, "
                f"classes={len(self.target_class_names)})")


def main():
    """
    主函数 - 命令行接口示例
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv11 行人车辆检测跟踪器")
    parser.add_argument("--model", type=str, default="yolo11n.pt", 
                       help="模型路径")
    parser.add_argument("--source", type=str, required=True,
                       help="输入源 (视频路径)")
    parser.add_argument("--output", type=str,
                       help="输出路径")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.4,
                       help="IoU阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="推理图像尺寸")
    parser.add_argument("--tracker", type=str, default="bytetrack",
                       choices=["bytetrack", "botsort"],
                       help="跟踪器类型")
    parser.add_argument("--device", type=str,
                       help="推理设备")
    parser.add_argument("--line", nargs=4, metavar=('x1', 'y1', 'x2', 'y2'),
                       type=int, action='append',
                       help="添加计数线坐标 (可多次使用)")
    parser.add_argument("--save-frames", action="store_true",
                       help="保存处理帧")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="详细输出")
    
    args = parser.parse_args()
    
    try:
        # 初始化跟踪器
        tracker = PedestrianVehicleTracker(
            model=args.model,
            tracker_type=args.tracker,
            device=args.device,
            verbose=args.verbose
        )
        
        # 添加计数线
        if args.line:
            for i, line_coords in enumerate(args.line):
                x1, y1, x2, y2 = line_coords
                tracker.add_counting_line((x1, y1), (x2, y2), f"line_{i+1}")
        
        # 执行跟踪
        results = tracker.track_video(
            source=args.source,
            output=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save_frames=args.save_frames
        )
        
        # 保存结果
        if args.output:
            output_dir = Path(args.output).parent / "results"
            tracker.save_results(output_dir)
        
    except Exception as e:
        LOGGER.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 