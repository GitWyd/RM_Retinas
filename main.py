# calculating reprojection error 
# camera frame to global frame transition (using the world tags)

import os
import cv2
from dt_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt

# Detector setup
detector = Detector

import os

import cv2
from dt_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt

# Detector setup
detector = Detector(searchpath=['apriltags'],
                    families='tag36h11',
                    nthreads=8,
                    quad_decimate=1.0,
                    quad_sigma=0.2,
                    refine_edges=True,
                    decode_sharpening=0.25,
                    debug=False)

# Video capture setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # Height

# Loading Camera Calibration Results
calib_file = os.path.join("assets/calibration", 'calibration_data.npz')
if os.path.exists(calib_file):
    with np.load(calib_file) as X:
        mtx, dist, rvecs, tvecs = X['mtx'], X['dist'], X['rvecs'], X['tvecs']
else:
    print("Calibration data does not exist. Please run calibration.py first")
    exit()

# Visualization and data storage setup
# display_width = 1920
# display_height = 1080
global_poses = []
reprojection_errors = []

# Functions
def draw(img, rvec, tvec, mtx, dist):
    axis_length = 5
    origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
    axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)
    imgpts_origin, _ = cv2.projectPoints(origin, rvec, tvec, mtx, dist)
    imgpts_axis, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)
    img = cv2.line(img, tuple(imgpts_origin[0].ravel().astype(int)), tuple(imgpts_axis[0].ravel().astype(int)), (0, 0, 255), 5)
    img = cv2.line(img, tuple(imgpts_origin[0].ravel().astype(int)), tuple(imgpts_axis[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv2.line(img, tuple(imgpts_origin[0].ravel().astype(int)), tuple(imgpts_axis[2].ravel().astype(int)), (255, 0, 0), 5)
    return img

def compute_reprojection_error(observed_corners, reprojected_corners):
    error = np.sum(np.linalg.norm(observed_corners - reprojected_corners, axis=2))
    return error / 4


# Function to calculate the transformation matrix from rvec and tvec
def transformation_matrix(rvec, tvec):
    rot_mat, _ = cv2.Rodrigues(rvec)
    trans_mat = np.column_stack((rot_mat, tvec))
    return np.vstack((trans_mat, [0, 0, 0, 1]))

# Dictionary to store the predefined spatial relationships of world tags
world_tag_positions = {
    575: np.array([0, 0, 0, 1]),
    579: np.array([50, 0, 0, 1]),
    582: np.array([0, 50, 0, 1]),
    585: np.array([50, 50, 0, 1])
}


world_to_camera_transform = None  # Transformation matrix from world to camera frame

# Main loop
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = detector.detect(gray)

    world_to_camera_transform = None

    # Step 1: Detect AprilTags in the current frame
    for tag in result:
        # Step 2: For each detected tag, estimate its pose relative to the camera
        corners = tag.corners.astype(int)
        cv2.polylines(frame, [corners], True, (255,0,0), 2)
        cv2.putText(frame, str(tag.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Determine tag size
        if 575 <= tag.tag_id <= 586:  # World tags of size 55mm x 55mm
            model_points = np.float32([[-2.75, -2.75, 0], [2.75, -2.75, 0], [2.75, 2.75, 0], [-2.75, 2.75, 0]])
        else:  # Other tags of size 18mm x 18mm
            model_points = np.float32([[-0.9, -0.9, 0], [0.9, -0.9, 0], [0.9, 0.9, 0], [-0.9, 0.9, 0]])
        
        # Pose estimation
        ret, rvec, tvec = cv2.solvePnP(model_points, tag.corners.astype(np.float32).reshape(-1, 2), mtx, dist)
        tag_transform = transformation_matrix(rvec, tvec)
        
        #### world tag transformation ####

        # Step 2: If the detected tag is one of the "world tags", compute the world-to-camera transformation
        if tag.tag_id in world_tag_positions:
            # Camera-to-Tag Transformation (already calculated)
            camera_to_tag = tag_transform
            
            # World-to-Tag Transformation
            world_to_tag_translation = world_tag_positions[tag.tag_id]
            world_to_tag = np.eye(4)
            world_to_tag[:3, 3] = world_to_tag_translation[:3]

            # Compute World-to-Camera Transformation
            try:
                world_to_camera_transform = np.linalg.inv(camera_to_tag) @ world_to_tag
            except np.linalg.LinAlgError:
                print(f"Error: Singular matrix encountered for tag_id {tag.tag_id}")
                continue  # Skip this tag and move to the next one.

        # Step 3: Transform all other detected tag poses into the world frame
        if world_to_camera_transform is not None:
            global_pose = world_to_camera_transform @ tag_transform @ np.array([0, 0, 0, 1])  # Homogeneous coordinates
            pose_text = f"Global Pose: T=({global_pose[0]:.2f}, {global_pose[1]:.2f}, {global_pose[2]:.2f})"
            cv2.putText(frame, pose_text, tuple(corners[0] + [0, -30]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Compute and display reprojection error
        reprojected_corners, _ = cv2.projectPoints(model_points, rvec, tvec, mtx, dist)
        error = compute_reprojection_error(tag.corners, reprojected_corners)
        error_text = f"Error: {error:.2f} px"
        cv2.putText(frame, error_text, tuple(corners[0] + [0, -60]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Store data
        global_poses.append((tag.tag_id, rvec, tvec))
        reprojection_errors.append(error)

    # Visualization
    # frame_resized = cv2.resize(frame, (display_width, display_height))
    # cv2.imshow('AprilTag Detection', frame_resized)
    cv2.imshow('AprilTag Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot reprojection errors
plt.figure()
plt.bar(range(len(reprojection_errors)), reprojection_errors, align='center')
plt.xlabel('Tag ID')
plt.ylabel('Reprojection Error (px)')
plt.title('Reprojection Errors for Detected Tags')
plt.show()

# wayland to x11