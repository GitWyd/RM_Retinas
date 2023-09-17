import cv2
import apriltag
import numpy as np
import pyrealsense2 as rs

# Initialize the apriltag detector (same as before)

detector = apriltag.Detector(apriltag.DetectorOptions(
            families='tag36h11',
            border=1,
            nthreads=8,
            quad_decimate=1.0,
            quad_blur=0.2, # a little blurring helps, bettter than 0.0
            refine_edges=True,
            refine_decode=True, # change to true to refine the decoded bits by sampling the pixel neighborhood around the bit
            refine_pose=False, # change to true to refine tag pose estimate
            debug=False,
            quad_contours=True
            ))


# Use your calibration matrix (this is for elpK)
mtx = np.array([
            [9.53e2, 0  , 975],
            [0  , 9.53e2, 550],
            [0  , 0  , 1]
        ])

# Use a null distortion matrix (assuming no lens distortion)
dist = np.zeros(5)

# Create a pipeline for the RealSense camera
pipeline = rs.pipeline()
config = rs.config()

# Enable the streams you want (e.g., depth and color)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# working pair at detection distance of 1.1 m
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
# # config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 8)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)


config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
# config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 8)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 8)


#####################################3


# Create a context object. This object owns the handles to all connected realsense devices
# pipeline = rs.pipeline()
# pipeline.start()

# # Get the active profile and the device
# profile = pipeline.get_active_profile()
# device = profile.get_device()

# # Get and print information of all stream profiles
# for sensor in device.sensors:
#     for profile in sensor.get_stream_profiles():
#         print(profile)

# pipeline.stop()

################################3333

# Start streaming
pipeline.start(config)

def draw(img, rvec, tvec, mtx, dist):
    # Define the length of the coordinate axes
    axis_length = 5

    # Project the origin and the axis points into the image
    origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
    axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)
    imgpts_origin, _ = cv2.projectPoints(origin, rvec, tvec, mtx, dist)
    imgpts_axis, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)

    # Draw the coordinate axes on the image
    img = cv2.line(img, tuple(imgpts_origin[0].ravel().astype(int)), tuple(imgpts_axis[0].ravel().astype(int)), (0, 0, 255), 5)  # X-axis (Red)
    img = cv2.line(img, tuple(imgpts_origin[0].ravel().astype(int)), tuple(imgpts_axis[1].ravel().astype(int)), (0, 255, 0), 5)  # Y-axis (Green)
    img = cv2.line(img, tuple(imgpts_origin[0].ravel().astype(int)), tuple(imgpts_axis[2].ravel().astype(int)), (255, 0, 0), 5)  # Z-axis (Blue)

    return img

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames(10000)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the color image to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Initialize a black image of the same size as your frame
        axes_img = np.zeros_like(color_image)

        # Perform detection on the thresholded image
        result = detector.detect(gray)

        # Draw the detections on the original color frame
        for tag in result:
            # Get the four corners of the tag
            corners = tag.corners.astype(int)

            # Draw the tag boundary
            cv2.polylines(color_image, [corners], True, (255,0,0), 2)

            # Draw tag id
            cv2.putText(color_image, str(tag.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Define the model points (in 3D space) of the tag (assuming it's a 5x5 square)
            model_points = np.float32([[-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0]])

            # Estimate the tag's pose in the camera frame
            ret, rvec, tvec = cv2.solvePnP(model_points, tag.corners.astype(np.float32).reshape(-1, 2), mtx, dist)

            # project 3D points to image plane
            axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,-0.1]]).reshape(-1,3)
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

            axes_img = draw(color_image, rvec, tvec, mtx, dist)

        # Display the resulting frame with detections drawn
        cv2.imshow('AprilTag Detection', color_image)
        cv2.imshow('AprilTag Axes', axes_img)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()





