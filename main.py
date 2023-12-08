import os
import sys
sys.path.append('/home/simonhwk/RobotMetabolism/particleTrussServer')
from math import sqrt
import numpy as np
import cv2
import dt_apriltags
import pandas as pd
import time
# from shared_resources import retinas_data_lock

######################
## GLOBAL CONSTANTS ##
######################

DEBUG = False

def debug_print(*args):
    if DEBUG:
        print(*args)

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
    586: np.array([0.3, 0.3, 0])
}


world_tag_size = 0.055 # 55mm
april_tag_size = 0.017

# Tag Corner Coordinates in Tag Frame
obj_pts_square = np.array([
    [-april_tag_size/2, -april_tag_size/2, 0],
    [ april_tag_size/2, -april_tag_size/2, 0],
    [-april_tag_size/2,  april_tag_size/2, 0],
    [ april_tag_size/2,  april_tag_size/2, 0],
])

columns = ['link_num', 'link_tag_id', 'centroid_x', 'centroid_y', 'centroid_z',
                   'upper_tip_x', 'upper_tip_y', 'upper_tip_z',
                   'bottom_tip_x', 'bottom_tip_y', 'bottom_tip_z']

#############
## CLASSES ##
#############


class Tag():
    def __init__(self, frame, tag, R_camera_to_world, t_camera_to_world, tag_size):
        # self.mtx = mtx
        # self.dist = dist
        self.frame = frame
        self.tag = tag
        self.tag_id = tag.tag_id
        self.R_camera_to_world = R_camera_to_world
        self.t_camera_to_world = t_camera_to_world
        self.tag_size = tag_size
        self.tf_tag_corners = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
        ])
        self.cf_linkbody_pts = np.array([])
        
        # related to link

        #frame, tag, R_avg_camera_to_world, t_avg_camera_to_world, tag_size


    # def draw_original_tag_boundary(self):
    #     # Draw detected corners without transformation for comparison
    #     corners = self.tag.corners.astype(int)
    #     cv2.polylines(self.frame, [corners], True, (0,255,0), 2)
    #     return None
    
    def draw_tag_boundary(self):

        # Tag Frame to Camera Frame Transformation for Corners
        cf_tag_corners, _ = cv2.projectPoints(self.tf_tag_corners, self.tag.pose_R, self.tag.pose_t, mtx, dist)
        cf_tag_corners = cf_tag_corners.reshape(-1, 2).astype(int)

        # Draw the tag boundary in green
        for i in range(4):
            cv2.line(self.frame, tuple(cf_tag_corners[i]), tuple(cf_tag_corners[(i+1)%4]), (255, 0, 0), 2)

        return cf_tag_corners # tag corners in camera frame

    def draw_axes(self):

         # Define the 3D points for XYZ axes
        axis_length = 0.03  
        obj_pts_axes = np.array([
            [0, 0, 0],         
            [axis_length, 0, 0],  
            [0, axis_length, 0],  
            [0, 0, axis_length]   
        ])

        # Tag Frame to Camera Frame Transformation for Axes
        img_pts_axes, _ = cv2.projectPoints(obj_pts_axes, self.tag.pose_R, self.tag.pose_t, mtx, dist)
        img_pts_axes = img_pts_axes.reshape(-1, 2).astype(int)



        # Draw the transformed XYZ axes on to the Camera Frame
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(1, 4):
            cv2.line(self.frame, tuple(img_pts_axes[0]), tuple(img_pts_axes[i]), colors[i-1], 2)

        return None

    # def draw_centroid_axes(self, link_tag_id):
    #     debug_print(link_tag_id)
    #     # debug_print(type(tag.tag_id), tag.tag_id)
    #     debug_print(type(link_tag_id), link_tag_id)

    #     axis_length = 0.05

    #     if link_tag_id is not None:

    #         # if link frame id is from 0 to 5 -> UPPER
    #         if (0 <= link_tag_id <= 5):
    #             tf_axes_pts = np.array([
    #                 [0,0,0],         
    #                 [axis_length, 0, 0],  
    #                 [0, axis_length, 0],  
    #                 [0, 0, axis_length]   
    #             ])+ [0.047, 0, 0.016]
    #         # if link frame id is from 6 to 11 -> BOTTOM
    #         elif(6 <= link_tag_id <= 11):
    #             tf_axes_pts = np.array([
    #                 [0,0,0],         
    #                 [axis_length, 0, 0],  
    #                 [0, axis_length, 0],  
    #                 [0, 0, axis_length]   
    #             ]) + [-0.047, 0, 0.016]

    #          # Tag Frame to Camera Frame Transformation for Axes
    #         cf_axes_pts, _ = cv2.projectPoints(tf_axes_pts, self.tag.pose_R, self.tag.pose_t, mtx, dist)
    #         cf_axes_pts = cf_axes_pts.reshape(-1, 2).astype(int)

    #         # Draw the transformed XYZ axes on to the Camera Frame
    #         colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    #         for i in range(1, 4):
    #             cv2.line(self.frame, tuple(cf_axes_pts[0]), tuple(cf_axes_pts[i]), colors[i-1], 2)

    #         return None
   

    def calculate_linkbody(self, link_tag_id, servo_extension = 0):

        # if link frame id is from 0 to 5 -> UPPER
        if (0 <= link_tag_id <= 5):
            linkbody_pts = np.array([
            [0.047, 0, 0.016], # centroid
            [-(0.085+servo_extension), 0, 0.016] # upper tip (servo unextended) 
            ])
        # if link frame id is from 6 to 11 -> BOTTOM
        elif(6 <= link_tag_id <= 11):
            linkbody_pts = np.array([
            [-0.047, 0, 0.016], # centroid
            [0.085+servo_extension, 0, 0.016] # bottom tip (servo unextended) 
            ])

        self.cf_linkbody_pts, _ = cv2.projectPoints(linkbody_pts, self.tag.pose_R, self.tag.pose_t, mtx, dist)
        self.cf_linkbody_pts = self.cf_linkbody_pts.reshape(-1, 2).astype(int)

        return None

    def draw_linkbody(self, link_tag_id, servo_extension = 0):
        
        
        debug_print("Shape of cf_linkbody_pts:", self.cf_linkbody_pts.shape)

        for pt in self.cf_linkbody_pts:
            neon_green = (57, 255, 20)
            # Draw the transformed centroid and tips! as a circle
            cv2.circle(self.frame, tuple(pt), 5, neon_green, -1)

        return None

    def compute_tranformation(self):

        # initialization
        R_tag_to_world = None
        t_tag_to_world = None

        if self.R_camera_to_world is not None and self.t_camera_to_world is not None:
            # Will obtain tag_to_world transformation
            # To compute the tag's pose in world frame

            R_tag_to_world = np.dot(self.R_camera_to_world, self.tag.pose_R)
            t_tag_to_world = np.dot(self.R_camera_to_world, self.tag.pose_t) + self.t_camera_to_world

            debug_print("R_tag_to_world:", R_tag_to_world)
            debug_print("t_tag_to_world:", t_tag_to_world)

            if np.any(np.isnan(R_tag_to_world)) or np.any(np.isnan(t_tag_to_world)):
                debug_print("Warning: Transformation matrices contain NaN values!")

        return R_tag_to_world, t_tag_to_world


    def project_to_world(self, R_tag_to_world, t_tag_to_world, cf_tag_corners, cf_linkbody_pts = None, link_tag_id = None, servo_extension = 0):
        
        centroid_world = 0
        tip_world = 0
        # Only execute linkbody related code if both camera_linkbody_pts and link_tag_id are provided
        if cf_linkbody_pts is not None and link_tag_id is not None:

            # if link frame id is from 0 to 5 -> UPPER
            if (0 <= link_tag_id <= 5):
                linkbody_pts = np.array([
                [0.047, 0, 0.016], # centroid
                [-(0.085+servo_extension), 0, 0.016] # upper tip (servo unextended) 
                ])
            # if link frame id is from 6 to 11 -> BOTTOM
            elif(6 <= link_tag_id <= 11):
                linkbody_pts = np.array([
                [-0.047, 0, 0.016], # centroid
                [0.085+servo_extension, 0, 0.016] # bottom tip (servo unextended) 
                ])

            linkbody_pts_world = np.dot(R_tag_to_world, linkbody_pts.T).T + t_tag_to_world.T

            centroid_world = linkbody_pts_world[0]
            tip_world = linkbody_pts_world[1]

            centroid_str = f"Centroid: X: {centroid_world[0]*100:.1f}, Y: {centroid_world[1]*100:.1f}, Z: {centroid_world[2]*100:.1f}"
            tip_str = f"Tip: X: {tip_world[0]*100:.1f}, Y: {tip_world[1]*100:.1f}, Z: {tip_world[2]*100:.1f}"

            # centroid and tip points in camera frame
            cf_centroid = cf_linkbody_pts[0]
            cf_tip = cf_linkbody_pts[1]

            # # display world coordinates on the camera frame cordinates (since our view is in camera frame)
            # offset_y = 20  # vertical offset for text placement
            # cv2.putText(self.frame, centroid_str, (cf_centroid[0] - 60, cf_centroid[1] - offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(self.frame, tip_str, (cf_tip[0] - 60, cf_tip[1] - offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        
        # Set a static offset for the text, e.g., 10 pixels above the tag center.
        cf_tag_center = np.mean(cf_tag_corners, axis=0).astype(int)
        text_offset = 10

        ############ DISPLAY WORLD POS OF THE TAGS ################### CAN BE OFF FOR NOW
        #Display the pose estimation coordinates relative to the world frame ABOVE the tag
        if self.tag_id == 575:
            world_pos_str = f"X: {(t_tag_to_world[0][0])*100:.1f}, Y: {(t_tag_to_world[1][0])*100:.1f}, Z: {(t_tag_to_world[2][0])*100:.1f}"
            cv2.putText(self.frame, world_pos_str, (cf_tag_center[0] - 60, cf_tag_center[1] - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        #####################################################
        # HAVE TO ADD PART FOR TAG TO WORLD FOR THE CENTROID AND TIP POINTS AS WELL ALSO FOR X - AXES####
        #####################################################
        return centroid_world, tip_world # will be adding in the the link instance


class LinkBody:
    def __init__(self, frame):
        self.tags = {}
        self.frame = frame
        # self.tags[link_tag_id][centroid] = #centroid value
        
        # initialize dataframe to collect link centroid, tip, axes data
        self.data = pd.DataFrame(columns=columns)

    def update_data(self, link_num, link_tag_id, centroid, tip):
        # determine tip type
        axis_length = 0.05

        if 0 <= link_tag_id <= 5:
            tip_type = 'upper_tip'
            tf_axes_pts = np.array([
                [0,0,0],         
                [axis_length, 0, 0],  
                [0, axis_length, 0],  
                [0, 0, axis_length]   
            ])+ [0.047, 0, 0.016]
        else:
            tip_type = 'bottom_tip'
            tf_axes_pts = np.array([
                [0,0,0],         
                [axis_length, 0, 0],  
                [0, axis_length, 0],  
                [0, 0, axis_length]   
            ]) + [-0.047, 0, 0.016]
        
        # Tag Frame to Camera Frame Transformation for Axes
        cf_axes_pts, _ = cv2.projectPoints(tf_axes_pts, self.tags[link_tag_id].tag.pose_R, self.tags[link_tag_id].tag.pose_t, mtx, dist)
        cf_axes_pts = cf_axes_pts.reshape(-1, 2).astype(int)

        new_data = {
            'link_num': link_num,
            'link_tag_id': link_tag_id,
            'centroid_x': centroid[0], 'centroid_y': centroid[1], 'centroid_z': centroid[2],
            tip_type+'_x': tip[0], tip_type+'_y': tip[1], tip_type+'_z': tip[2],
            'cf_axes_pts_o': cf_axes_pts[0], 'cf_axes_pts_x': cf_axes_pts[1], 'cf_axes_pts_y': cf_axes_pts[2], 'cf_axes_pts_z': cf_axes_pts[3]
        }
        # debug_print(type(self.data))
        debug_print(self.data)
        # self.data = self.data.append(new_data, ignore_index = True)
        self.data = pd.concat([self.data, pd.DataFrame([new_data])], ignore_index=True)

    def compute_mean(self):

        unique_link_nums = self.data['link_num'].unique()

        for link_num in unique_link_nums:
            filtered_data = self.data[self.data['link_num'] == link_num]
            mean_values = {
                'link_num' : link_num,
                'link_tag_id' : 'Mean',
                'centroid_x': filtered_data['centroid_x'].mean(),
                'centroid_y': filtered_data['centroid_y'].mean(),
                'centroid_z': filtered_data['centroid_z'].mean(),
                'upper_tip_x': filtered_data['upper_tip_x'].mean(),
                'upper_tip_y': filtered_data['upper_tip_y'].mean(),
                'upper_tip_z': filtered_data['upper_tip_z'].mean(),
                'bottom_tip_x': filtered_data['bottom_tip_x'].mean(),
                'bottom_tip_y': filtered_data['bottom_tip_y'].mean(),
                'bottom_tip_z': filtered_data['bottom_tip_z'].mean(),
                'cf_axes_pts_o': filtered_data['cf_axes_pts_o'].mean(),
                'cf_axes_pts_x': filtered_data['cf_axes_pts_x'].mean(),
                'cf_axes_pts_y': filtered_data['cf_axes_pts_y'].mean(),
                'cf_axes_pts_z': filtered_data['cf_axes_pts_z'].mean()
            }
            # self.data = self.data.append(mean_values, ignore_index = True)
            self.data = pd.concat([self.data, pd.DataFrame([mean_values])], ignore_index=True)

        return None

    def append(self, tag, link_tag_id):
        # link_tag_id is from 1 to 12
        # link_id : 14
        # tag_id : link_id*12 + 11

        if link_tag_id not in self.tags:
            self.tags[link_tag_id] = tag
        
        return None
    
    def get_link_tag_id(self, link_tag_id):
        self.tags.get(link_tag_id, {})

    def display_linkbody(self, R_world_to_camera, t_world_to_camera):

        # TO DISPLAY THE MEAN LINKBODY PTS IN THE CF
        mean_row = self.data[self.data['link_tag_id'] == "Mean"].iloc[0]
        wf_centroid = (mean_row['centroid_x'], mean_row['centroid_y'], mean_row['centroid_z'])
        wf_upper_tip = (mean_row['upper_tip_x'], mean_row['upper_tip_y'], mean_row['upper_tip_z'])
        wf_bottom_tip = (mean_row['bottom_tip_x'], mean_row['bottom_tip_y'], mean_row['bottom_tip_z'])
        
        # TO DISPLAY AXES IN CF
        cf_axes_pts = (mean_row['cf_axes_pts_o'], mean_row['cf_axes_pts_x'], mean_row['cf_axes_pts_y'], mean_row['cf_axes_pts_z'])
        
        debug_print("CF_AXES_PTS: ", cf_axes_pts)
         # Draw the transformed XYZ axes on to the Camera Frame
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(1, 4):
            start_point = tuple(map(int, cf_axes_pts[0]))
            end_point = tuple(map(int, cf_axes_pts[i]))
            cv2.line(self.frame, start_point, end_point, colors[i-1], 2)

        wf_linkbody_mean = np.array([wf_centroid, wf_upper_tip, wf_bottom_tip], dtype = np.float32)
        # cf_linkbody_mean = np.dot(R_world_to_camera, a) + t_world_to_camera
        cf_linkbody_mean, _ = cv2.projectPoints(wf_linkbody_mean, R_world_to_camera, t_world_to_camera, mtx, dist)
        debug_print(f"cf_linkbody_mean{cf_linkbody_mean}")
        # cf_linkbody_mean = cf_linkbody_mean.reshape(-1, 2).astype(int)
        cf_linkbody_reshaped = cf_linkbody_mean.reshape(-1, 2)
        mask = ~np.isnan(cf_linkbody_reshaped).any(axis=1)
        cf_linkbody_reshaped[mask] = cf_linkbody_reshaped[mask].astype(int) # convert to integer
        debug_print(f"cf_linkbody_mean{cf_linkbody_reshaped}")


        centroid_str = f"Centroid: X: {wf_centroid[0]*100:.1f}, Y: {wf_centroid[1]*100:.1f}, Z: {wf_centroid[2]*100:.1f}"
        upper_tip_str = f"Upper Tip: X: {wf_upper_tip[0]*100:.1f}, Y: {wf_upper_tip[1]*100:.1f}, Z: {wf_upper_tip[2]*100:.1f}"
        bottom_tip_str = f"Bottom Tip: X: {wf_bottom_tip[0]*100:.1f}, Y: {wf_bottom_tip[1]*100:.1f}, Z: {wf_bottom_tip[2]*100:.1f}"

        neon_green = (57, 255, 20)

        counter = 0

        for pt in cf_linkbody_reshaped[mask]:
            # debug_print(pt)
            if not np.isnan(np.array(pt)).any(): # skip
                # debug_print(tuple(pt))
                # cv2.circle(self.frame, tuple(pt), 5, neon_green, -1)
                cv2.circle(self.frame, center=(int(pt[0]), int(pt[1])), radius=5, color=neon_green, thickness=-1)
                
                # display world coordinates on the camera frame cordinates (since our view is in camera frame)
                offset_y = 20  # vertical offset for text placement
                if counter == 0:
                    cv2.putText(self.frame, centroid_str, (int(pt[0]) - 60, int(pt[1]) - offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                elif counter == 1:
                    cv2.putText(self.frame, upper_tip_str, (int(pt[0]) - 60, int(pt[1]) - offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                elif counter == 2:
                    cv2.putText(self.frame, bottom_tip_str, (int(pt[0]) - 60, int(pt[1]) - offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            counter +=1 


####################
## INITIALIZATION ##
####################


# Video capture setup
display_width = 1920
display_height = 1080
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # Height


# Loading calibration results
calib_file = os.path.join("particleTrussServer/RM_Retinas/assets/calibration", "calibration_data.npz")

# The path when pwd is RM_Retinas
# calib_file = os.path.join("assets/calibration", 'calibration_data.npz')

if os.path.exists(calib_file):
    with np.load(calib_file) as X:
        mtx, dist, rvecs, tvecs = X['mtx'], X['dist'], X['rvecs'], X['tvecs']
        debug_print(f"mtx: {mtx}")
        debug_print(f"dist: {dist}")
        debug_print(f"rvecs: {rvecs}")
        debug_print(f"tvecs: {tvecs}")
else:
    debug_print("Calibration data does not exist. Please run calibration.py first")
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

# link number and related TagPose instances
# data collection and organization for later compute
# DATA STRUCTURE
# links  = { link_num : LinkBody instances for the link_num }
# these instances have related tags as values{ 'link_num'.tags[link_frame_tag_id] }
# 17.tags[value between 1 ~ 12] returns a dictionary for that tag
# 17.tags[1] = {
#     'pose_t': tag.pose_t,
#     'pose_R': tag.pose_R,
#     'center': tag.center
# }

# THREADING
retinas_data = {}

#######################
## UTILITY FUNCTIONS ##
#######################

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
    img_pts_reprojected, _ = cv2.projectPoints(obj_pts_square, tag.pose_R, tag.pose_t, mtx, dist)
    img_pts_reprojected = img_pts_reprojected.reshape(-1, 2)

    # Compute the distance between reprojected corners and detected corners
    error = np.linalg.norm(img_pts_reprojected - tag.corners, axis=1)
    avg_error = np.mean(error)
    
    return avg_error


def main():

    global retinas_data
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # Height

    # failed_reads = 0
    # max_retries = 5

    try:

        while True:
            ret, frame = cap.read()
            # if not ret:
            #     failed_reads += 1
            #     debug_print(f"Failed to read frame. Attempt: {failed_reads}")
            #     if failed_reads >= max_retries:
            #         debug_print("Exceeded maximum retries. Reinitializing camera.")
            #         # Reinitialize video capture
            #         cap.release()
            #         time.sleep(1)  # Short delay
            #         cap = cv2.VideoCapture(0)
            #         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            #         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            #         failed_reads = 0
            #         continue
            #     if failed_reads >= max_retries * 2:
            #         debug_print("Persistent failure in reading frames. Exiting.")
            #         # Handle persistent failure
            #         break
            if not ret:
                print("Failed to capture frame. Releasing camera.")
                cap.release()
                time.sleep(1)  # Short delay before reinitializing the camera
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                continue
            
            links = {}

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
                        R_world_to_camera, t_world_to_camera = tag.pose_R, tag.pose_t

            # If we found the world tag, use its transformation for all other tags. 
            # Otherwise, skip world frame computations for this frame.
            if R_camera_to_world is not None and t_camera_to_world is not None:
                world_tag_counter = 0

                for tag in result_world_tags:
                    if tag.tag_id in WORLD_TAGS:
                        detected_world_tag = Tag(frame, tag, R_camera_to_world, t_camera_to_world, world_tag_size)
                        # t_tag_to_world = draw_pose(frame, tag, R_camera_to_world, t_camera_to_world, world_tag_size)
                        cf_tag_corners = detected_world_tag.draw_tag_boundary()
                        detected_world_tag.draw_axes()
                        R_tag_to_world, t_tag_to_world = detected_world_tag.compute_tranformation()
                        detected_world_tag.project_to_world(R_tag_to_world, t_tag_to_world, cf_tag_corners)

                        if tag.tag_id is not None:
                            validate_world_position(tag, t_tag_to_world, frame, world_tag_counter)
                            world_tag_counter += 1


                for tag in result_april_tags:
                    if tag.tag_id not in WORLD_TAGS:
                        
                        link_num = tag.tag_id // 12 # Link Number starts with P0
                        link_tag_id = tag.tag_id % 12
                        
                        # logic for storing tag pose data to LinkPose instances
                        if link_num not in links:
                            links[link_num] = LinkBody(frame)
                        
                        # links[link_num].get_link_tag_id(link_tag_id)

                            
                        detected_tag = Tag(frame, tag, R_camera_to_world, t_camera_to_world, april_tag_size)
                        links[link_num].append(detected_tag, link_tag_id) # custom append!
                        # detected_tag.draw_original_boundary()
                        cf_tag_corners = detected_tag.draw_tag_boundary()
                        detected_tag.draw_axes()
                        # detected_tag.draw_centroid_axes(link_tag_id)
                        

                        detected_tag.calculate_linkbody(link_tag_id)
                        # detected_tag.draw_linkbody(link_tag_id)

                        R_tag_to_world, t_tag_to_world = detected_tag.compute_tranformation()
                        wf_centroid, wf_tip = detected_tag.project_to_world(R_tag_to_world, t_tag_to_world, cf_tag_corners, detected_tag.cf_linkbody_pts, link_tag_id)
                        
                        

                        # update data
                        links[link_num].update_data(link_num, link_tag_id, wf_centroid, wf_tip)

        
                for link_num, link_body in links.items():
                    
                    # first_tag_link_tag_id = next(iter(link_body.tags.keys()))
                    # first_tag = next(iter(link_body.tags.values()))
                    # first_tag.draw_centroid_axes(first_tag_link_tag_id)
                    link_body.compute_mean()
                    link_body.display_linkbody(R_world_to_camera, t_world_to_camera)
                    # link_body.compute_axes() # compute x axis from centroid to the tip

                    ######################################################################
                    # implmenting something that retinas to cl controller here #
                    ######################################################################

                    mean_values_data = link_body.data[link_body.data['link_tag_id'] == 'Mean']
                    mean_row = mean_values_data.iloc[0]


                    retinas_data[link_num] = {
                        'centroid': (mean_row['centroid_x'], mean_row['centroid_y'], mean_row['centroid_z']),
                        'upper_tip': (mean_row['upper_tip_x'], mean_row['upper_tip_y'], mean_row['upper_tip_z']),
                        'bottom_tip': (mean_row['bottom_tip_x'], mean_row['bottom_tip_y'], mean_row['bottom_tip_z'])
                    }
                        
                    # reset dataframe for the next frame -> later u will use this line to collect data   
                    

            frame_resized = cv2.resize(frame, (display_width, display_height))
            cv2.imshow("AprilTags Pose Estimation", frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occured: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

def retinas_thread():
    try:
        main()

    except Exception as e:
        debug_print(f"Error in retinas thread: {e}")

if __name__ == "__main__":
    main()