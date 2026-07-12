#!/usr/bin/env python3
"""
Ultralytics YOLOv11 行人车辆检测器
基于官方 ultralytics 框架的行人车辆检测实现
专门用于检测视频中的行人和车辆目标

Example:
    ```python
    from detect_pedestrian_vehicle import PedestrianVehicleDetector
    
    detector = PedestrianVehicleDetector("yolo11n.pt")
    results = detector.predict("video.mp4", save=True)
    ```

Dependencies:
    - ultralytics
    - opencv-python
    - numpy
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER, ROOT, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors


class PedestrianVehicleDetector:
    """
    行人车辆检测器类，基于 YOLOv11 检测模型
    
    专门用于检测行人和车辆类别的目标检测器，使用 ultralytics 框架标准接口。
    支持图像和视频输入，提供灵活的配置选项。
    
    Attributes:
        model (YOLO): YOLOv11 检测模型实例
        names (Dict): 类别名称映射
        device (str): 推理设备 ('cpu' 或 'cuda')
        target_classes (List[str]): 目标检测类别列表
        class_colors (Dict[str, tuple]): 类别颜色映射
        
    Examples:
        初始化检测器:
        >>> detector = PedestrianVehicleDetector("yolo11n.pt")
        
        检测单张图像:
        >>> results = detector.predict("image.jpg", conf=0.5)
        
        检测视频:
        >>> results = detector.predict_video("video.mp4", output="output.mp4")
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
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化行人车辆检测器
        
        Args:
            model (str | Path): 模型路径或模型名称
            device (str, optional): 推理设备，自动检测如果为None
            verbose (bool): 是否显示详细信息
            
        Raises:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型加载失败
        """
        self.verbose = verbose
        if verbose:
            LOGGER.info(f"{colorstr('Ultralytics YOLOv11')} 🚀 行人车辆检测器初始化中...")
        
        try:
            # 加载模型
            self.model = YOLO(model, task='detect', verbose=verbose)
            self.names = self.model.names
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 获取目标类别的索引
            self.target_class_indices = []
            self.target_class_names = []
            
            for cls_name in self.TARGET_CLASSES:
                for idx, name in self.names.items():
                    if name == cls_name:
                        self.target_class_indices.append(idx)
                        self.target_class_names.append(cls_name)
                        break
            
            if verbose:
                LOGGER.info(f"检测器初始化成功")
                LOGGER.info(f"设备: {self.device}")
                LOGGER.info(f"模型: {model}")
                LOGGER.info(f"目标类别: {self.target_class_names}")
                
        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise RuntimeError(f"无法初始化检测器: {e}")
    
    def predict(
        self,
        source: Union[str, Path, np.ndarray, List],
        conf: float = 0.5,
        iou: float = 0.4,
        imgsz: Union[int, List[int]] = 640,
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
        stream: bool = False,
        **kwargs
    ):
        """
        对输入源进行预测
        
        Args:
            source: 输入源 (图像路径、视频路径、numpy数组等)
            conf (float): 置信度阈值
            iou (float): NMS IoU阈值  
            imgsz (int | List[int]): 推理图像尺寸
            save (bool): 是否保存结果
            save_dir (str | Path, optional): 保存目录
            show (bool): 是否显示结果
            stream (bool): 是否流式处理
            **kwargs: 其他参数
            
        Returns:
            results: 预测结果列表
            
        Examples:
            >>> results = detector.predict("image.jpg", conf=0.6, save=True)
            >>> for result in results:
            ...     print(f"检测到 {len(result.boxes)} 个目标")
        """
        if self.verbose:
            LOGGER.info(f"开始预测: {source}")
        
        # 设置保存目录
        if save and save_dir is None:
            save_dir = increment_path(Path("runs/detect/exp"))
            
        # 执行预测，只检测目标类别
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=self.target_class_indices,  # 只检测目标类别
            save=save,
            save_dir=save_dir,
            show=show,
            stream=stream,
            verbose=self.verbose,
            **kwargs
        )
        
        return results
    
    def predict_video(
        self,
        source: Union[str, Path],
        output: Optional[Union[str, Path]] = None,
        conf: float = 0.5,
        iou: float = 0.4,
        imgsz: int = 640,
        show_progress: bool = True,
        **kwargs
    ) -> List:
        """
        预测视频文件
        
        Args:
            source (str | Path): 输入视频路径
            output (str | Path, optional): 输出视频路径
            conf (float): 置信度阈值
            iou (float): NMS IoU阈值
            imgsz (int): 推理图像尺寸
            show_progress (bool): 是否显示进度
            **kwargs: 其他参数
            
        Returns:
            List: 预测结果列表
            
        Examples:
            >>> results = detector.predict_video("input.mp4", "output.mp4", conf=0.6)
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"视频文件不存在: {source}")
            
        if self.verbose:
            LOGGER.info(f"处理视频: {source}")
            
        # 设置输出路径
        if output is None:
            output = source.parent / f"{source.stem}_detected{source.suffix}"
        else:
            output = Path(output)
            
        # 确保输出目录存在
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # 执行预测
        results = self.predict(
            source=str(source),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=True,
            save_dir=output.parent,
            stream=True,
            **kwargs
        )
        
        return list(results) if hasattr(results, '__iter__') else [results]
    
    def annotate_results(
        self,
        image: np.ndarray,
        results,
        line_width: Optional[int] = None,
        font_size: Optional[float] = None,
        font: str = 'Arial.ttf',
        show_conf: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        在图像上标注检测结果
        
        Args:
            image (np.ndarray): 输入图像
            results: 检测结果
            line_width (int, optional): 边框线宽
            font_size (float, optional): 字体大小
            font (str): 字体类型
            show_conf (bool): 是否显示置信度
            show_labels (bool): 是否显示标签
            
        Returns:
            np.ndarray: 标注后的图像
        """
        annotator = Annotator(
            image,
            line_width=line_width,
            font_size=font_size,
            font=font,
            pil=False
        )
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                if cls in [list(self.names.keys())[i] for i in range(len(self.names)) 
                          if self.names[i] in self.TARGET_CLASSES]:
                    
                    class_name = self.names[cls]
                    color = self.CLASS_COLORS.get(class_name, (128, 128, 128))
                    
                    # 构建标签
                    label = class_name if show_labels else ""
                    if show_conf:
                        label += f" {conf:.2f}" if label else f"{conf:.2f}"
                    
                    # 绘制边框和标签
                    annotator.box_label(box, label, color=color)
        
        return annotator.result()
    
    def get_detection_stats(self, results) -> Dict[str, int]:
        """
        获取检测统计信息
        
        Args:
            results: 检测结果或结果列表
            
        Returns:
            Dict[str, int]: 各类别检测数量统计
        """
        stats = {cls: 0 for cls in self.TARGET_CLASSES}
        
        # 处理单个结果或结果列表
        if not isinstance(results, list):
            results = [results]
            
        for result in results:
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for cls in classes:
                    class_name = self.names[cls]
                    if class_name in stats:
                        stats[class_name] += 1
        
        return stats
    
    def save_detection_results(
        self,
        results,
        save_dir: Union[str, Path],
        format: str = 'txt'
    ) -> None:
        """
        保存检测结果到文件
        
        Args:
            results: 检测结果
            save_dir (str | Path): 保存目录
            format (str): 保存格式 ('txt', 'json', 'csv')
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'txt':
            # 保存为YOLO格式
            for i, result in enumerate(results if isinstance(results, list) else [results]):
                if result.boxes is not None:
                    save_path = save_dir / f"result_{i:06d}.txt"
                    result.save_txt(save_path)
        
        elif format == 'json':
            # 保存为JSON格式 (需要实现)
            pass
            
        elif format == 'csv':
            # 保存为CSV格式 (需要实现)
            pass
    
    def __repr__(self) -> str:
        """返回检测器的字符串表示"""
        return (f"{self.__class__.__name__}("
                f"model={self.model.ckpt_path}, "
                f"device={self.device}, "
                f"classes={len(self.target_class_names)})")


def main():
    """
    主函数 - 命令行接口示例
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv11 行人车辆检测器")
    parser.add_argument("--model", type=str, default="yolo11n.pt", 
                       help="模型路径")
    parser.add_argument("--source", type=str, required=True,
                       help="输入源 (图像/视频路径)")
    parser.add_argument("--output", type=str,
                       help="输出路径")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.4,
                       help="IoU阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="推理图像尺寸")
    parser.add_argument("--device", type=str,
                       help="推理设备")
    parser.add_argument("--save", action="store_true",
                       help="保存结果")
    parser.add_argument("--show", action="store_true",
                       help="显示结果")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="详细输出")
    
    args = parser.parse_args()
    
    try:
        # 初始化检测器
        detector = PedestrianVehicleDetector(
            model=args.model,
            device=args.device,
            verbose=args.verbose
        )
        
        # 执行预测
        results = detector.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save=args.save,
            show=args.show
        )
        
        # 显示统计信息
        if args.verbose:
            stats = detector.get_detection_stats(results)
            LOGGER.info("检测统计:")
            for cls_name, count in stats.items():
                if count > 0:
                    LOGGER.info(f"  {cls_name}: {count}")
        
    except Exception as e:
        LOGGER.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 