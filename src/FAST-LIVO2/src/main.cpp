#include "LIVMapper.h"

int main(int argc, char **argv)
{//初始化 ROS 节点
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  //创建图像传输对象
  image_transport::ImageTransport it(nh);
  //创建 LIVMapper 对象
  LIVMapper mapper(nh); 
  //初始化订阅和发布
  mapper.initializeSubscribersAndPublishers(nh, it);
  //运行主功能
  mapper.run();
  return 0;
}