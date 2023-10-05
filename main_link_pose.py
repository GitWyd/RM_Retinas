import os
from math import sqrt
import numpy as np
import cv2
import dt_apriltags


######################
## GLOBAL CONSTANTS ##
######################


# Define world tags and their locations in world frame
WORLD_TAGS = {
    575: np.array([0, 0, 0]),
    576: None,
    577: None,
    578: np.array([0.3, 0, 0]),
    579: None,
    580: None,
    581: np.array([0, 0.3, 0]),
    582: None,
    583: None,
    584: None,
    585: None,
    586: np.array([0.3, 0.3, 0]),
}

# Define link and the attached 12 apriltags in link frame
H = 0.010 # 10cm
R = 0.02 # 2cm
R_prime = sqrt(3) * R

# Define link tags and their lcoations in link frame
# LINK_TAGS = {
#     # UPPER EDGE TAGS
#     1: [R, 0, H/2],
#     2: [R/2, R_prime/2, H/2],
#     3: [-R/2, R_prime/2, H/2],
#     4: [-R, 0, H/2],
#     5: [-R/2, -R_prime/2, H/2],
#     6: [R/2, -R_prime/2, H/2],
#     # BOTTOM EDGE TAGS
#     7: [R, 0, -H/2],
#     8: [R/2, R_prime/2, -H/2],
#     9: [-R/2, R_prime/2, -H/2],
#     10: [-R, 0, -H/2],
#     11: [-R/2, -R_prime/2, -H/2],
#     12: [R/2, -R_prime/2, -H/2],
# }



world_tag_size = 0.055 # 55mm
april_tag_size = 0.017

#############
## CLASSES ##
#############


class TagBoundary:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
    
    @staticmethod
    def draw_boundary(self, frame, corners, color = (255, 0, 0), thickness = 4):
        corners_int = corners.astype(int)
        cv2.polylines(frame, [corners,int], True, color, thickness)
    
    @staticmethod
    def project_points(self, obj_pts, R, t):
        img_pts, _ = cv2.projectPoints(obj_pts, R, t, self.mtx, self.dist)
        return img_pts.reshape(-1, 2).astype(int)
    
    @staticmethod
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
   

class LinkTags:
    def __init__(self):
        self.tags = {}
    
    def append(self, tag):
        # link_tag_id is from 1 to 12
        link_tag_id = tag.tag_id % 12
        # link_id : 14
        # tag_id : link_id*12 + 11

        if link_tag_id not in self.tags:
            self.tags[link_tag_id] = {}
        
        self.tags[link_tag_id] = {
            'pose_t': tag.pose_t,
            'pose_R': tag.pose_R,
            'center': tag.center,
            'pose_err' : tag.pose_err # used for confidence
        }
    
    def get_link_tag_id(self, link_tag_id):
        self.tags.get(link_tag_id, {})

    # this function should calculate the link's pose from the detected apriltags. position of the link is the centroid of the link and the orientation is the orientation of the link
    def pose_estimation(self):

        for link_tag_id, link_tag in self.tags.items():
            # Get link frame tag position
            link_frame_tag_position = LINK_TAGS[link_tag_id]
            
            # Get pose of the tag in camera frame
            t_link_to_camera = link_tag['pose_t']
            R_link_to_camera = link_tag['pose_R']
            
            # Transform the tag's position from link frame to camera frame
            t_link = np.dot(R_link_to_camera, link_frame_tag_position) + t_link_to_camera
            
            
            center = link_tag['center']
            pose_err = link_tag['pose_err']
        



            # first perform some operation to obtain the centroid of the link from the detected tags, also use LINK_TAGS
            # Apply transformation to move from link frame to the world frame
            # values[pose_t] = (,,)
            
#     for link in links.values():
#         for link_tag_id, values in link.tags.items():
            
#         pos = link.tags[link_tag_id]['pose_t']
#         link.tags 

    def draw_pose(link_pose):


        return None




####################
## INITIALIZATION ##
####################


# Video capture setup
display_width = 1920
display_height = 1080
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

# link number and related TagPose instances
# data collection and organization for later compute
# DATA STRUCTURE
# links  = { link_num : LinkFrameTags instances for the link_num }
# these instances have related tags as values{ 'link_num'.tags[link_frame_tag_id] }
# 17.tags[value between 1 ~ 12] returns a dictionary for that tag
# 17.tags[1] = {
#     'pose_t': tag.pose_t,
#     'pose_R': tag.pose_R,
#     'center': tag.center
# }

links = {}


#######################
## UTILITY FUNCTIONS ##
#######################


def centroid_estimation()

def draw_pose(frame, tag, R_avg_camera_to_world, t_avg_camera_to_world, tag_size):
    """Draw the tag pose estimation, XYZ coordinate axes, and pose coordinates in the frame."""
    corners = tag.corners.astype(int)
    cv2.polylines(frame, [corners], True, (255,0,0), 2)
    corners = tag.corners
    tag_center = np.mean(corners, axis=0).astype(int)

    # Tag Corner Coordinates in Tag Frame
    obj_pts_square = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0],

    ])

    # Tag Frame to Camera Frame Transformation for Corners
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

    # Tag Frame to Camera Frame Transformation for Axes
    img_pts_axes, _ = cv2.projectPoints(obj_pts_axes, tag.pose_R, tag.pose_t, mtx, dist)
    img_pts_axes = img_pts_axes.reshape(-1, 2).astype(int)

    # Draw the transformed XYZ axes on to the Camera Frame
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(1, 4):
        cv2.line(frame, tuple(img_pts_axes[0]), tuple(img_pts_axes[i]), colors[i-1], 2)

    # Project tag pose to world frame
    if R_avg_camera_to_world is not None and t_avg_camera_to_world is not None:
        # Compute tag's pose in world frame using the average Camera to World transform
        # Obtain rotation and transition matrices
        R_tag_to_world = np.dot(R_avg_camera_to_world, tag.pose_R)
        t_tag_to_world = np.dot(R_avg_camera_to_world, tag.pose_t) + t_avg_camera_to_world

        print("R_tag_to_world:", R_tag_to_world)
        print("t_tag_to_world :", t_tag_to_world)

        if np.any(np.isnan(R_tag_to_world)) or np.any(np.isnan(t_tag_to_world)):
            print("Warning: Transformation matrices contain NaN values!")

        # Project the tag's pose in the world frame onto the image
        img_pts_world, _ = cv2.projectPoints(obj_pts_square, R_tag_to_world, t_tag_to_world, mtx, dist)
        img_pts_world = img_pts_world.reshape(-1, 2).astype(int)

        print("img_pts_world_projected:", img_pts_world)

        if np.any(np.isnan(img_pts_world)):
            print("Warning: img_pts_world contains NaN values!")

        print(type(img_pts_world[0][0]), img_pts_world[0][0])
        print(type(img_pts_world[0][1]), img_pts_world[0][1])


        # Visualize the projected world pose with a circle in the image
        # cv2.circle(frame, tuple(img_pts_world[0]), 5, (0, 255, 255), -1)
        # cv2.circle(frame, (int(img_pts_world[0][0]), int(img_pts_world[0][1])), 5, (0, 255, 255), -1)

        # Set a static offset for the text, e.g., 10 pixels above the tag center.
        world_text_offset = 10

        # Display the pose estimation coordinates relative to the world frame ABOVE the tag
        world_pose_str = f"X: {(t_tag_to_world[0][0])*100:.1f}, Y: {(t_tag_to_world[1][0])*100:.1f}, Z: {(t_tag_to_world[2][0])*100:.1f}"
        cv2.putText(frame, world_pose_str, (tag_center[0] - 60, tag_center[1] - world_text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    return t_tag_to_world


def get_camera_to_world_transform(tag):
    # Invert the transformation, since we need camera -> world
    R_inv = np.transpose(tag.pose_R)
    t_inv = -np.dot(R_inv, tag.pose_t)
    return R_inv, t_inv

def average_transforms(transforms):
    average_R = np.mean([R for R, t in transforms], axis=0)
    average_t = np.mean([t for R, t in transforms], axis=0)
    return average_R, average_t

def validate_world_position(tag, t_tag_to_world, frame, tag_index):
    # Extract the tag's estimated position in the world frame
    estimated_position = t_tag_to_world.flatten()
    
    # Get the tag's true position from the WORLD_TAGS dictionary
    true_position = WORLD_TAGS.get(tag.tag_id)
    
    if true_position is not None:
        # Compute the difference between estimated and true position
        difference = true_position - estimated_position
        error_distance = np.linalg.norm(difference)

        display_text = f"Tag {tag.tag_id} Diff: {difference*100} Euclidean Distance: {(error_distance*100):.4f} cm"

        vertical_position = 30 + tag_index * 30
        position = (frame.shape[1] - 600, vertical_position)
        cv2.putText(frame, display_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
           
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
        result_world_tags = detector.detect(gray, True, camera_params, world_tag_size)

        result_april_tags = detector.detect(gray, True, camera_params, april_tag_size)

        R_camera_to_world = None
        t_camera_to_world = None

        for tag in result_world_tags:
            # If the detected tag is the origin (tag ID 575), use its transformation as the world frame  
            if tag.tag_id == 575:
                    R_camera_to_world, t_camera_to_world = get_camera_to_world_transform(tag)

        # If we found the origin tag, use its transformation for all tags. 
        # Otherwise, skip world frame computations for this frame.
        if R_camera_to_world is not None and t_camera_to_world is not None:
            world_tag_counter = 0
            for tag in result_world_tags:
                if tag.tag_id in WORLD_TAGS: 
                    t_tag_to_world = draw_pose(frame, tag, R_camera_to_world, t_camera_to_world, world_tag_size)
                    if tag.tag_id is not None:
                        validate_world_position(tag, t_tag_to_world, frame, world_tag_counter)
                        world_tag_counter += 1
            
            for tag in result_april_tags:
                if tag.tag_id not in WORLD_TAGS:
                    draw_pose(frame, tag, R_camera_to_world, t_camera_to_world, april_tag_size)

                    # tag.pose_t # translation of the pose estimate
                    # tag.center # center of the detection in image coordinates
                    # tag.pose_R # rotation matrix of the pose estimate
                    # tag.pose_err # object-space error of the estimation
                    # logic for storing tag pose data to LinkPose instances
                    
                    link_num = tag.tag_id // 12 + 1
                    link_tag_id = tag.tag_id % 12
                
                    if link_num not in links:
                        links[link_num] = LinkTags()
                    
                    links[link_num].append(tag)
                    # links[link_num].get_link_tag_id(link_tag_id)

            for link in links.values():
                link_pose = link.pose_estimation()
                link.draw_pose(link_pose)

            # now we have all data stored in links
            # for LinkTags Instance in links, we must calculate the center and the axes
            # using the predefined LINK_TAGS
            # links_pose = link_pose_estimation(links)

            # After all the calculations, draw link pose [pos and orientation and highlight as neon green]
            # draw_link_pose()

        frame_resized = cv2.resize(frame, (display_width, display_height))
        cv2.imshow("AprilTags Pose Estimation", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
