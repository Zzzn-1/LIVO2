#include "seticmanager.h"
#include <opencv2/opencv.hpp>

void seticmanager::processSemanticFrame(const cv::Mat& setic_img, 
                                      std::vector<pointWithVar>& semantic_points,
                                      const StatesGroup& state,
                                      double timestamp) {
    if(setic_img.empty()) {
        return;
    }

    // 保存处理后的图像
    processed_image_ = setic_img.clone();

    // 处理语义分割图像
    for(int y = 0; y < setic_img.rows; y++) {
        for(int x = 0; x < setic_img.cols; x++) {
            // 获取像素的语义标签
            int label = setic_img.at<uchar>(y, x);
            
            // 如果是有效的语义标签
            if(label > 0) {
                // 计算3D点坐标
                cv::Point3f point3d;
                // TODO: 根据相机参数将2D点投影到3D空间
                
                // 处理语义点
                processSemanticPoint(point3d, label, semantic_points);
            }
        }
    }

    // 可视化处理
    cv::Mat colored_semantic = cv::Mat::zeros(setic_img.size(), CV_8UC3);
    for(int y = 0; y < setic_img.rows; y++) {
        for(int x = 0; x < setic_img.cols; x++) {
            int label = setic_img.at<uchar>(y, x);
            if(label < label_colors_.size()) {
                colored_semantic.at<cv::Vec3b>(y, x) = label_colors_[label];
            }
        }
    }
    processed_image_ = colored_semantic;
}

void seticmanager::processSemanticPoint(const cv::Point3f& point, 
                                      int semantic_label,
                                      std::vector<pointWithVar>& semantic_points) {
    // 创建新的语义点
    pointWithVar pv;
    pv.point_w << point.x, point.y, point.z;
    pv.semantic_label = semantic_label;
    
    // 计算点的协方差
    // TODO: 根据深度和语义标签的不确定性计算协方差
    pv.var = Eigen::Matrix3d::Identity() * 0.01; // 临时使用固定协方差
    
    semantic_points.push_back(pv);
}

cv::Mat seticmanager::getProcessedImage() const {
    return processed_image_;
} 