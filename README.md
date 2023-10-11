# zed_ros_3d
ROS wrapper for TensorRT 3D object detection using ZED SDK

To use:

clone into src folder

```
git clone https://github.com/terrytao19/zed_ros_3d.git
```
├── ros_ws
│   ├── src
│   │   ├── package_1
│   │   ├── package_2
│   │   ├── zed_ros_3d
 
Make sure you have [zed_sdk](https://github.com/stereolabs/zed-sdk) installed

Build package
```
catkin build zed_ros_3d
source devel/setup.bash
```

Follow this [guide](https://github.com/stereolabs/zed-sdk/tree/master/object%20detection/custom%20detector/cpp/tensorrt_yolov5-v6-v8_onnx) to convert a .onnx or .pt model into .engine for TensorRT (You can also use rosrun zed_ros_3d_node in this package with the same arguments and the -s flag to convert it):

If your .engine model is stored somewhere else, move it to the zed_ros_3d directory (optional)

To run the detector:

Make sure you have roscore running

```
roscore
```

Then start the node:

```
rosrun zed_ros_3d zed_ros_3d_node [your model].engine 0
```

Assuming 0 is the camera ID, it will be 0 if you only have one zed camera connected.

[visualization_msgs/MarkerArray](https://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html) messages should now be publishing in the topic /zed_detections_raw

To view the markers, open rviz:

```
rosrun rviz rviz
```

Change the fixed frame to 'base_link', then add the MarkerArray topic

