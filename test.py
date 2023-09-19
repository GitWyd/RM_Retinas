import os
import cv2
import numpy as np
import dt_apriltags


######################
## GLOBAL CONSTANTS ##
######################


# Define world tags and their locations in world frame
WORLD_TAGS = {
    575: np.array([0, 0, 0]),
    578: np.array([0.3, 0, 0]),
    581: np.array([0, 0.3, 0]),
    586: np.array([0.3, 0.3, 0])
}

tag_size = 0.055 # 55mm


#############
## CLASSES ##
#############


class TagBoundary:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
    
    def draw_boundary(self, frame, corners, color = (255, 0, 0), thickness = 4):
        corners_int = corners.astype(int)
        cv2.polylines(frame, [corners,int], True, color, thickness)
    
    def project_points(self, obj_pts, R, t):
        img_pts, _ = cv2.projectPoints(obj_pts, R, t, self.mtx, self.dist)
        return img_pts.reshape(-1, 2).astype(int)
    
    def draw_axes(self, frame, R, t, axis_length = 0.05):
        obj_pts_axes = np.array([
            [0, 0, 0],         
            [axis_length, 0, 0],  
            [0, axis_length, 0],  
            [0, 0, axis_length]   
        ])
        # project axes to camera frame
        img_pts_axes = self.project_points(obj_pts_axes, R, t)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range (1, 4):
            cv2.line (frame, tuple(img_pts_axes[0]), tuple(img_pts_axes[i]), colors[i-1], 4)

    def draw_text(self, frame, text, position, color=(255, 255, 255), scale=0.7):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


####################
## INITIALIZATION ##
####################


# Video capture setup
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # Height

# Loading Camera Calibration Results
calib_file = os.path.join("assets/calibration", 'calibration_data.npz')
if os.path.exists(calib_file):
    with np.load(calib_file) as X:
        mtx, dist, rvecs, tvecs = X['mtx'], X['dist'], X['rvecs'], X['tvecs']
        print(f"mtx: {mtx}")
        print(f"dist: {dist}")
        print(f"rvecs: {rvecs}")
        print(f"tvecs: {tvecs}")
else:
    print("Calibration data does not exist. Please run calibration.py first")
    exit()

# Extract camera_params, which are focal lengths and optical center each in x, y direction
fx = mtx[0,0]
fy = mtx[1,1]
cx = mtx[0,2]
cy = mtx[1,2]
camera_params = (fx, fy, cx, cy)

# Create an AprilTag detector
detector = dt_apriltags.Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

visualizer = TagBoundary(mtx, dist)


#######################
## UTILITY FUNCTIONS ##
#######################


def draw_pose(frame, tag, R_avg_camera_to_world, t_avg_camera_to_world):
    """Draw the tag pose estimation, XYZ coordinate axes, and pose coordinates in the frame."""
    corners = tag.corners.astype(int)
    cv2.polylines(frame, [corners], True, (255,0,0), 2)
    corners = tag.corners
    tag_center = np.mean(corners, axis=0).astype(int)

    # Define the square's 3D points
    # Tag Frame
    obj_pts_square = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ])

    img_pts_square, _ = cv2.projectPoints(obj_pts_square, tag.pose_R, tag.pose_t, mtx, dist)
    img_pts_square = img_pts_square.reshape(-1, 2).astype(int)

    for i in range(4):
        cv2.line(frame, tuple(img_pts_square[i]), tuple(img_pts_square[(i+1)%4]), (0, 255, 0), 2)

    # Define the 3D points for XYZ axes
    axis_length = 0.05  
    obj_pts_axes = np.array([
        [0, 0, 0],         
        [axis_length, 0, 0],  
        [0, axis_length, 0],  
        [0, 0, axis_length]   
    ])

    img_pts_axes, _ = cv2.projectPoints(obj_pts_axes, tag.pose_R, tag.pose_t, mtx, dist)
    img_pts_axes = img_pts_axes.reshape(-1, 2).astype(int)

    # Draw the XYZ axes
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  
    for i in range(1, 4):
        cv2.line(frame, tuple(img_pts_axes[0]), tuple(img_pts_axes[i]), colors[i-1], 2)

    # Project tag pose to world frame
    if R_avg_camera_to_world is not None and t_avg_camera_to_world is not None:
        # Compute tag's pose in world frame using the average camera-to-world transform
        R_tag_to_world = np.dot(R_avg_camera_to_world, tag.pose_R)
        t_tag_to_world = np.dot(R_avg_camera_to_world, tag.pose_t) + t_avg_camera_to_world

        # Object point in world frame
        obj_pts_world = np.array([tag.pose_t.flatten()])

        # Project the tag's pose in the world frame onto the image
        img_pts_world, _ = cv2.projectPoints(obj_pts_world, R_tag_to_world, t_tag_to_world, mtx, dist)
        img_pts_world = img_pts_world.reshape(-1, 2).astype(int)

        # Visualize the projected world pose with a circle in the image
        # cv2.circle(frame, tuple(img_pts_world[0]), 5, (0, 255, 255), -1)
        cv2.circle(frame, (int(img_pts_world[0][0]), int(img_pts_world[0][1])), 5, (0, 255, 255), -1)

        # Calculate dynamic offset for the text based on tag's Z position in the world frame (further away = bigger offset)
        world_text_offset = int(20 / (0.1 + t_tag_to_world[2][0]))

        # Display the pose estimation coordinates relative to the world frame ABOVE the tag
        world_pose_str = f"X: {t_tag_to_world[0][0]:.2f}, Y: {t_tag_to_world[1][0]:.2f}, Z: {t_tag_to_world[2][0]:.2f}"
        cv2.putText(frame, world_pose_str, (tag_center[0] - 60, tag_center[1] - world_text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

def get_camera_to_world_transform(tag):
    # Invert the transformation, since we need camera -> world
    R_inv = np.transpose(tag.pose_R)
    t_inv = -np.dot(R_inv, tag.pose_t)
    return R_inv, t_inv

def average_transforms(transforms):
    average_R = np.mean([R for R, t in transforms], axis=0)
    average_t = np.mean([t for R, t in transforms], axis=0)
    return average_R, average_t

def validate_world_position(tag, t_tag_to_world):
    # Extract the tag's estimated position in the world frame
    estimated_position = t_tag_to_world.flatten()
    
    # Get the tag's true position from the WORLD_TAGS dictionary
    true_position = WORLD_TAGS.get(tag.tag_id)
    
    if true_position is not None:
        # Compute the difference between estimated and true position
        difference = true_position - estimated_position

        # Log the difference
        print(f"Tag {tag.tag_id} Difference: {difference}")

        # Optionally, compute the Euclidean distance (L2 norm) between the estimated and true position
        error_distance = np.linalg.norm(difference)
        print(f"Tag {tag.tag_id} Error Distance: {error_distance:.4f} meters")
        
        return difference
    return None

def compute_reprojection_error(tag):
    # Reproject the tag corners using the estimated pose
    obj_pts_square = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ])

    img_pts_reprojected, _ = cv2.projectPoints(obj_pts_square, tag.pose_R, tag.pose_t, mtx, dist)
    img_pts_reprojected = img_pts_reprojected.reshape(-1, 2)

    # Compute the distance between reprojected corners and detected corners
    error = np.linalg.norm(img_pts_reprojected - tag.corners, axis=1)
    avg_error = np.mean(error)
    
    return avg_error

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags in the frame
        result = detector.detect(gray, True, camera_params, tag_size)

        R_camera_to_world = None
        t_camera_to_world = None

        for tag in result:
            # If the detected tag is the origin (tag ID 575), use its transformation as the world frame
            if tag.tag_id == 575:
                R_camera_to_world, t_camera_to_world = get_camera_to_world_transform(tag)

        # If we found the origin tag, use its transformation for all tags. 
        # Otherwise, skip world frame computations for this frame.
        if R_camera_to_world is not None and t_camera_to_world is not None:
            for tag in result:
                draw_pose(frame, tag, R_camera_to_world, t_camera_to_world)

        cv2.imshow("AprilTags Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
