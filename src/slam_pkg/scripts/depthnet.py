import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import rospy 
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import yaml

def depthnet(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
    model.to(device)

    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Get the dimensions of the original image
    original_height, original_width, _ = image.shape

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_height, original_width),
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")

    return formatted


def publish_camera_info():
    # Load camera calibration parameters from the YAML file
    with open('camera_calibration.yaml', 'r') as yaml_file:
        calibration_data = yaml.safe_load(yaml_file)

    # Extract camera matrix and distortion coefficients
    mtx = np.array(calibration_data['camera_matrix'])
    dist = np.array(calibration_data['distortion_coefficients'])
    shape = np.array(calibration_data['shape'])

    # Create a CameraInfo message
    camera_info_msg = CameraInfo()
    camera_info_msg.header.frame_id = 'camera_link'  # Change the frame_id as needed
    camera_info_msg.distortion_model = 'plumb_bob'
    camera_info_msg.width = shape[1]  # Set the image width
    camera_info_msg.height = shape[0]  # Set the image height
    camera_info_msg.K = mtx.flatten().tolist()
    camera_info_msg.D = dist.flatten().tolist()

    return camera_info_msg

def publish_video(video_path):
    rospy.init_node('depthnet_publisher', anonymous=True)
    cap = cv2.VideoCapture(video_path)  # Use the path to your video file

    bridge = CvBridge()
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    depth_pub = rospy.Publisher('/camera/image_depth', Image, queue_size=10)
    info_pub = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=10)
    cam_info = publish_camera_info()
    seq=0
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            # Convert OpenCV image to ROS image message
            ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            # Publish the ROS image message
            cam_info.header.stamp = rospy.Time.now()
            cam_info.header.seq = seq
            depth_img = depthnet(frame)
            ros_depth = bridge.cv2_to_imgmsg(depth_img, encoding="mono8")
            ros_depth.header = cam_info.header
            ros_image.header = cam_info.header
            image_pub.publish(ros_image)
            depth_pub.publish(ros_depth)
            info_pub.publish(cam_info)
            seq+=1

        else:
            rospy.logwarn("End of video file reached. Exiting...")
            break
    cap.release()


if __name__ == '__main__':
    try:
        # Provide the path to your video file as an argument
        video_path ='/home/anatharv1/test_ws/data/room.MOV'
        publish_video(video_path)
    except rospy.ROSInterruptException:
        pass
