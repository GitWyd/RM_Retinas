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

# Camera calibration
mtx = np.array([
    [9.53e2, 0, 975],
    [0, 9.53e2, 550],
    [0, 0, 1]
])
dist = np.zeros(5)

# Visualization and data storage setup
display_width = 1920
display_height = 1080
global_poses = {}
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


# Main loop
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = detector.detect(gray)

    for tag in result:
        corners = tag.corners.astype(int)
        cv2.polylines(frame, [corners], True, (255,0,0), 2)
        cv2.putText(frame, str(tag.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Tag size
        if 575 <= tag.tag_id <= 586:  # World tags of size 55mm x 55mm
            model_points = np.float32([[-2.75, -2.75, 0], [2.75, -2.75, 0], [2.75, 2.75, 0], [-2.75, 2.75, 0]])
        else:  # Other tags of size 18mm x 18mm
            model_points = np.float32([[-0.9, -0.9, 0], [0.9, -0.9, 0], [0.9, 0.9, 0], [-0.9, 0.9, 0]])

        # Pose estimation
        ret, rvec, tvec = cv2.solvePnP(model_points, tag.corners.astype(np.float32).reshape(-1, 2), mtx, dist)

        # Display global pose
        pose_text = f"Global Pose: T=({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})"
        cv2.putText(frame, pose_text, tuple(corners[0] + [0, -30]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Compute and display reprojection error
        reprojected_corners, _ = cv2.projectPoints(model_points, rvec, tvec, mtx, dist)
        error = compute_reprojection_error(tag.corners, reprojected_corners)
        error_text = f"Error: {error:.2f} px"
        cv2.putText(frame, error_text, tuple(corners[0] + [0, -60]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Store data
        global_poses[tag.tag_id] = tvec.ravel()  
        reprojection_errors.append(error)


    # Visualization
    frame_resized = cv2.resize(frame, (display_width, display_height))
    cv2.imshow('AprilTag Detection', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot global poses
tag_ids = list(global_poses.keys())
x_positions = [global_poses[tag_id][0] for tag_id in tag_ids]
y_positions = [global_poses[tag_id][1] for tag_id in tag_ids]
z_positions = [global_poses[tag_id][2] for tag_id in tag_ids]

plt.figure()
plt.bar(tag_ids, x_positions, align='center', label='X Position')
plt.bar(tag_ids, y_positions, align='center', bottom=x_positions, label='Y Position')
plt.bar(tag_ids, z_positions, align='center', bottom=np.array(x_positions)+np.array(y_positions), label='Z Position')
plt.xlabel('Tag ID')
plt.ylabel('Global Position (mm)')
plt.title('Global Positions for Detected Tags')
plt.legend()
plt.show()

