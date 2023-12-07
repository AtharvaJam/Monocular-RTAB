#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

def publish_video(video_path):
    rospy.init_node('video_publisher', anonymous=True)
    rate = rospy.Rate(10)  # Set the publishing rate (10 Hz in this example)

    cap = cv2.VideoCapture(video_path)  # Use the path to your video file

    bridge = CvBridge()
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    depth_pub = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=10)
    info_pub = rospy.Publisher('/camera/depth_image', Image, queue_size=10)

    cam_info = CameraInfo()

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            # Convert OpenCV image to ROS image message
            ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            # Publish the ROS image message
            image_pub.publish(ros_image)
        else:
            rospy.logwarn("End of video file reached. Exiting...")
            break

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        # Provide the path to your video file as an argument
        video_path = "/home/anatharv1/test_ws/room.MOV"
        publish_video(video_path)
    except rospy.ROSInterruptException:
        pass
