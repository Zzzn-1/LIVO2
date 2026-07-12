class seticmanager {
public:
    seticmanager() = default;
    ~seticmanager() = default;

    // 处理语义图像帧
    void processSemanticFrame(const cv::Mat& setic_img, 
                            std::vector<pointWithVar>& semantic_points,
                            const StatesGroup& state,
                            double timestamp);

    // 获取处理后的图像
    cv::Mat getProcessedImage() const;

    // 处理单个语义点
    void processSemanticPoint(const cv::Point3f& point, 
                            int semantic_label,
                            std::vector<pointWithVar>& semantic_points);

private:
    cv::Mat processed_image_;
    std::vector<int> semantic_labels_;
    std::vector<cv::Vec3b> label_colors_;
}; 