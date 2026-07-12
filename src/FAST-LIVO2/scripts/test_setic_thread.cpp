#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>

/**
 * SETIC副线程测试程序
 * 
 * 功能：
 * 1. 发布模拟的语义图像到 /setic/image_raw 话题
 * 2. 验证SETIC副线程是否正确接收和处理数据
 * 3. 监控处理结果的发布
 */

class SeticThreadTester
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher semantic_pub_;
    image_transport::Subscriber result_sub_;
    ros::Timer timer_;
    
    int frame_count_;
    
public:
    SeticThreadTester() : it_(nh_), frame_count_(0)
    {
        // 发布模拟语义图像
        semantic_pub_ = it_.advertise("/setic/image_raw", 1);
        
        // 订阅处理结果
        result_sub_ = it_.subscribe("/setic_img", 1, &SeticThreadTester::resultCallback, this);
        
        // 定时发布测试图像 (10Hz)
        timer_ = nh_.createTimer(ros::Duration(0.1), &SeticThreadTester::publishTestImage, this);
        
        ROS_INFO("[SETIC-TEST] Started SETIC thread tester");
        ROS_INFO("[SETIC-TEST] Publishing semantic images to: /setic/image_raw");
        ROS_INFO("[SETIC-TEST] Listening for results on: /setic_img");
    }
    
    void publishTestImage(const ros::TimerEvent& event)
    {
        // 创建模拟语义图像 (640x480, 3通道)
        cv::Mat semantic_image = generateSemanticImage();
        
        // 转换为ROS消息
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
            std_msgs::Header(), "bgr8", semantic_image).toImageMsg();
        
        msg->header.stamp = ros::Time::now();
        msg->header.frame_id = "camera";
        
        semantic_pub_.publish(msg);
        
        frame_count_++;
        
        if (frame_count_ % 50 == 0) {
            ROS_INFO("[SETIC-TEST] Published %d test frames", frame_count_);
        }
    }
    
    cv::Mat generateSemanticImage()
    {
        cv::Mat image(480, 640, CV_8UC3);
        
        // 创建不同颜色的区域模拟语义分割结果
        
        // 背景 - 灰色
        image.setTo(cv::Scalar(128, 128, 128));
        
        // 建筑物 - 红色区域
        cv::rectangle(image, cv::Point(50, 50), cv::Point(250, 200), cv::Scalar(0, 0, 255), -1);
        
        // 植被 - 绿色区域  
        cv::circle(image, cv::Point(400, 150), 80, cv::Scalar(0, 255, 0), -1);
        
        // 道路 - 蓝色区域
        cv::rectangle(image, cv::Point(0, 350), cv::Point(640, 480), cv::Scalar(255, 0, 0), -1);
        
        // 车辆 - 黄色区域 (移动的)
        int x_offset = (frame_count_ * 2) % 300;
        cv::rectangle(image, cv::Point(200 + x_offset, 300), 
                     cv::Point(280 + x_offset, 350), cv::Scalar(0, 255, 255), -1);
        
        // 添加一些噪声
        cv::Mat noise(image.size(), CV_8UC3);
        cv::randu(noise, cv::Scalar(0, 0, 0), cv::Scalar(30, 30, 30));
        cv::add(image, noise, image);
        
        return image;
    }
    
    void resultCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        static int result_count = 0;
        result_count++;
        
        ROS_INFO("[SETIC-TEST] Received processed result #%d at timestamp %.6f", 
                 result_count, msg->header.stamp.toSec());
        
        try {
            cv::Mat result_image = cv_bridge::toCvShare(msg, "bgr8")->image;
            
            // 显示处理结果 (可选)
            if (!result_image.empty()) {
                cv::imshow("SETIC Processing Result", result_image);
                cv::waitKey(1);
                
                ROS_DEBUG("[SETIC-TEST] Result image size: %dx%d", 
                         result_image.cols, result_image.rows);
            }
            
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("[SETIC-TEST] Failed to convert result image: %s", e.what());
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "setic_thread_tester");
    
    ROS_INFO("=== SETIC副线程测试程序 ===");
    ROS_INFO("此程序将：");
    ROS_INFO("1. 发布模拟语义图像到 /setic/image_raw");
    ROS_INFO("2. 监控SETIC副线程的处理结果");
    ROS_INFO("3. 验证线程间通信是否正常");
    ROS_INFO("按 Ctrl+C 停止测试");
    ROS_INFO("=============================");
    
    SeticThreadTester tester;
    
    ros::spin();
    
    cv::destroyAllWindows();
    return 0;
} 