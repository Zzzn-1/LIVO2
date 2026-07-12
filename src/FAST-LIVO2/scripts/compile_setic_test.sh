#!/bin/bash

# SETIC副线程测试程序编译脚本

echo "编译SETIC副线程测试程序..."

# 检查是否在正确的工作空间
if [ ! -f "CMakeLists.txt" ]; then
    echo "错误：请在FAST-LIVO2项目根目录下运行此脚本"
    exit 1
fi

# 添加测试程序到CMakeLists.txt (如果还没有添加)
grep -q "test_setic_thread" CMakeLists.txt
if [ $? -ne 0 ]; then
    echo "添加测试程序到CMakeLists.txt..."
    cat >> CMakeLists.txt << 'EOF'

# SETIC副线程测试程序
add_executable(test_setic_thread scripts/test_setic_thread.cpp)
target_link_libraries(test_setic_thread
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
)
EOF
    echo "已添加测试程序到构建配置"
fi

# 编译
echo "开始编译..."
cd ../../..  # 回到工作空间根目录
catkin_make --only-pkg-with-deps FAST-LIVO2

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo ""
    echo "运行测试："
    echo "1. 启动主程序：roslaunch fast_livo2 mapping.launch"
    echo "2. 运行测试：rosrun FAST-LIVO2 test_setic_thread"
    echo "3. 查看结果：rostopic echo /setic_img"
else
    echo "编译失败，请检查错误信息"
    exit 1
fi 