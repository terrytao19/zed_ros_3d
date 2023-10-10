cmake_minimum_required(VERSION 3.0.2)
project(zed_ros_3d)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)


find_package(ZED 4 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_libraries(${catkin_LIBRARIES})
link_libraries(${OpenCV_LIBRARIES})
link_libraries(${ZED_LIBRARIES})
link_libraries(${CUDA_LIBRARIES})

SET(TRT_LIBS nvinfer nvonnxparser)

link_libraries( ${TRT_LIBS}
                ${SPECIAL_OS_LIBS}
                ${ZED_LIBS}
                ${OPENGL_LIBRARIES}
                ${GLUT_LIBRARIES}
                ${GLEW_LIBRARIES}
                ${OpenCV_LIBRARIES})

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
catkin_package()
add_executable(zed_ros_3d_node src/zed_ros_3d_node.cpp src/yolo.cpp)
