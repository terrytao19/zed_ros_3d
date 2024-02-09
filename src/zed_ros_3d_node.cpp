#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "ros/ros.h"

#include "yolo.hpp"

#include <sl/Camera.hpp>
#include <NvInfer.h>

#include <zed_interfaces/Object.h>
#include <zed_interfaces/ObjectsStamped.h>

// #include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
// #include <image_transport/image_transport.h>
// #include <sensor_msgs/image_encodings.h>

using namespace nvinfer1;
#define NMS_THRESH 0.4
#define CONF_THRESH 0.3

bool isSunInFrame(const cv::Mat &frame)
{

    cv::Mat hsvFrame;
    cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

    // Define a range for the sun color in HSV space
    cv::Scalar lowerSunColor(20, 100, 100); // Example values for yellow sun
    cv::Scalar upperSunColor(30, 255, 255);

    // Create a mask to extract the sun color
    cv::Mat sunMask;
    cv::inRange(hsvFrame, lowerSunColor, upperSunColor, sunMask);

    // Find contours in the sun mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(sunMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Iterate through contours and check for conditions indicating the sun
    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area > 2000)
        { // Adjust the threshold based on your video characteristics
            // Sun detected
            return true;
        }
    }

    // Sun not detected
    return false;
}

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix)
{
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
    {
        // std::cout << " | " << toString(err_code) << " : ";
        // std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

cv::Rect get_rect(BBox box)
{
    return cv::Rect(round(box.x1), round(box.y1), round(box.x2 - box.x1), round(box.y2 - box.y1));
}

std::vector<sl::uint2> cvt(const BBox &bbox_in)
{
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
    bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
    bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
    bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
    return bbox_out;
}

int main(int argc, char **argv)
{

    if (argc == 1)
    {
        std::cout << "Usage: \n 1. ./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine\n 2. ./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine images:1x3x512x512\n 3. ./yolo_onnx_zed yolov8s.engine <SVO path>" << std::endl;
        return 0;
    }

    // Initialize ros publisher
    ros::init(argc, argv, "zed_ros_3d_node");

    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ros::NodeHandle n;

    /**
     * The advertise() function is how you tell ROS that you want to
     * publish on a given topic name. This invokes a call to the ROS
     * master node, which keeps a registry of who is publishing and who
     * is subscribing. After this advertise() call is made, the master
     * node will notify anyone who iconst cv::Mat &fs trying to subscribe to this topic name,
     * and they will in turn negotiate a peer-to-peer connection with this
     * node.  advertise() returns a Publisher object which allows you to
     * publish messages on that topic through a call to publish().  Once
     * all copies of the returned Publisher object are destroyed, the topic
     * will be automatically unadvertised.
     *
     * The second parameter to advertise() is the size of the message queue
     * used for publishing messages.  If messages are published more quickly
     * than we can send them, the number here specifies how many messages to
     * buffer up before throwing some away.
     */
    std::string object_det_topic_root = "obj_det";
    std::string object_det_topic = object_det_topic_root + "/objects";

    ros::Publisher detections_pub = n.advertise<zed_interfaces::ObjectsStamped>(object_det_topic, 1);
    // image_transport::ImageTransport it_(n);
    // image_transport::Publisher image_pub_ = it_.advertise("/zed_detections_img", 1);
    // cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

    // Check Optim engine first
    if (std::string(argv[1]) == "-s" && (argc >= 4))
    {
        std::string onnx_path = std::string(argv[2]);
        std::string engine_path = std::string(argv[3]);
        OptimDim dyn_dim_profile;

        if (argc == 5)
        {
            std::string optim_profile = std::string(argv[4]);
            bool error = dyn_dim_profile.setFromString(optim_profile);
            if (error)
            {
                std::cerr << "Invalid dynamic dimension argument, expecting something like 'images:1x3x512x512'" << std::endl;
                return EXIT_FAILURE;
            }
        }

        Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
        return 0;
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = 60;

    if (argc > 1)
    {
        std::string zed_opt = argv[2];
        if (zed_opt.find(".svo") != std::string::npos)
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
    }
    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS)
    {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    zed.enablePositionalTracking();
    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = false; // designed to give person pixel mask
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS)
    {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }
    auto camera_config = zed.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(std::min((int)camera_config.resolution.width, 720), std::min((int)camera_config.resolution.height, 404));
    auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;
    // Creating the inference engine class
    std::string engine_name = "";
    Yolo detector;

    if (argc > 0)
        engine_name = argv[1];
    else
    {
        std::cout << "Error: missing engine name as argument" << std::endl;
        return EXIT_FAILURE;
    }

    if (detector.init(engine_name))
    {
        std::cerr << "Detector init failed!" << std::endl;
        return EXIT_FAILURE;
    }
    auto display_resolution = zed.getCameraInformation().camera_configuration.resolution;
    sl::Mat left_sl, point_cloud;
    cv::Mat left_cv;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Pose cam_w_pose;
    cam_w_pose.pose_data.setIdentity();

    int count = 0;
    int exposure = 0;

    ros::Time t;
    while (ros::ok())
    {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS)
        {

            t = ros::Time::now();

            zed.setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, true);

            // Get image for inference
            zed.retrieveImage(left_sl, sl::VIEW::LEFT);

            // Running inference
            auto detections = detector.run(left_sl, display_resolut            {
                std::cerr << "Invalid dynamic dimension argument, expecting something like 'images:1x3x512x512'" << std::endl;
                return EXIT_FAILURE;
            }
        }

        Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
        return 0;
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = 60;

    if (argc > 1)
    {
        std::string zed_opt = argv[2];
        if (zed_opt.find(".svo") != std::string::npos)
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
    }
    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS)
    {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    zed.enablePositionalTracking();
    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = false; // designed to give person pixel mask
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS)
    {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }
    auto camera_config = zed.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(stdion.height, display_resolution.width, CONF_THRESH);
            // Get image for display
            left_cv = slMat2cvMat(left_sl);

                        auto start = std::chrono::high_resolution_clock::now();

            if (!isSunInFrame(left_cv))
            {
                ++exposure;
            }
            else
            {
                --exposure;
            }

            if (exposure < 0)
            {
                exposure = 0;
            }
            else if (exposure > 100)
            {
                exposure = 100;
            }

            zed.setCameraSettings(sl::VIDEO_SETTINGS::BRIGHTNESS, exposure);

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

            std::cout << "Sun in Frame? " << isSunInFrame(left_cv) << " | "
                      << "Time: " << duration.count() << " ms"
                      << " | Exposure: " << exposure << " %\n";


            zed.getCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, exposure);

            // std::cout << "Exposure: " << exposure << " %\n";

            // Preparing for ZED SDK ingesting
            std::vector<sl::CustomBoxObjectData> objects_in;
            for (auto &it : detections)
            {
                sl::CustomBoxObjectData tmp;
                // Fill the detections into the correct format
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = it.prob;
                tmp.label = (int)it.label;
                tmp.bounding_box_2d = cvt(it.box);
                tmp.is_grounded = ((int)it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                // others are tracked in full 3D space
                objects_in.push_back(tmp);
            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);

            // Displaying 'raw' objects
            for (size_t j = 0; j < detections.size(); j++)
            {
                cv::Rect r = get_rect(detections[j].box);
                cv::rectangle(left_cv, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(left_cv, std::to_string((int)detections[j].label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }

            cv::imshow("Objects", left_cv);
            cv::waitKey(10);

            // cv_ptr->encoding = "bgr8";
            // cv_ptr->header.stamp = t;
            // cv_ptr->header.frame_id = "/zed_detections_img";
            // cv_ptr->image = left_cv;

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, objectTracker_parameters_rt);

            size_t objCount = objects.object_list.size();

            zed_interfaces::ObjectsStampedPtr objMsg = boost::make_shared<zed_interfaces::ObjectsStamped>();
            objMsg->header.stamp = t;
            objMsg->header.frame_id = "base_link";

            objMsg->objects.resize(objCount);

            size_t idx = 0;
            for (auto data : objects.object_list)
            {
                objMsg->objects[idx].label = sl::toString(data.label).c_str();
                objMsg->objects[idx].sublabel = sl::toString(data.sublabel).c_str();
                objMsg->objects[idx].label_id = data.raw_label;
                objMsg->objects[idx].instance_id = data.id;
                objMsg->objects[idx].confidence = data.confidence;

                memcpy(&(objMsg->objects[idx].position[0]), &(data.position[0]), 3 * sizeof(float));
                memcpy(&(objMsg->objects[idx].position_covariance[0]), &(data.position_covariance[0]), 6 * sizeof(float));
                memcpy(&(objMsg->objects[idx].velocity[0]), &(data.velocity[0]), 3 * sizeof(float));

                objMsg->objects[idx].tracking_available = 1;
                objMsg->objects[idx].tracking_state = static_cast<int8_t>(data.tracking_state);
                // NODELET_INFO_STREAM( "[" << idx << "] Tracking: " <<
                // sl::toString(static_cast<sl::OBJECT_TRACKING_STATE>(data.tracking_state)));
                objMsg->objects[idx].action_state = static_cast<int8_t>(data.action_state);

                if (data.bounding_box_2d.size() == 4)
                {
                    memcpy(&(objMsg->objects[idx].bounding_box_2d.corners[0]), &(data.bounding_box_2d[0]), 8 * sizeof(unsigned int));
                }
                if (data.bounding_box.size() == 8)
                {
                    memcpy(&(objMsg->objects[idx].bounding_box_3d.corners[0]), &(data.bounding_box[0]), 24 * sizeof(float));
                }

                memcpy(&(objMsg->objects[idx].dimensions_3d[0]), &(data.dimensions[0]), 3 * sizeof(float));

                // Body Detection is in a separate module in ZED SDK v4
                objMsg->objects[idx].skeleton_available = false;

                // at the end of the loop
                idx++;
            }

            // image_pub_.publish(cv_ptr->toImageMsg());

            detections_pub.publish(objMsg);

            ros::spinOnce();
        }
        ++count;
    }

    return 0;
}
