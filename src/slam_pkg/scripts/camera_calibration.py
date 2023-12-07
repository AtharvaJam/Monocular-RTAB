import numpy as np
import cv2
import glob
import yaml

# Chessboard size (number of inner corners)
chessboard_size = (7, 5)  # Updated for a 6x8 chessboard

# Termination criteria for the corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(5,7,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane.

# Load images from the specified directory
images = glob.glob('/home/anatharv1/test_ws/data/chessboard/*.jpg')  # Change the path accordingly

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If corners are found, add object points, image points
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
# (h,w) = img.shape[:2]
# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

calibration_data = {
    'camera_matrix': mtx.tolist(),
    'distortion_coefficients': dist.tolist(),
    'shape': list(img.shape)
}

with open('camera_calibration.yaml', 'w') as yaml_file:
    yaml.dump(calibration_data, yaml_file)


