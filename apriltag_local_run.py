
import cv2
import apriltag
import numpy as np

# ############################## CAMERA INDICE CODE ##############################333
# def find_all_available_cameras(max_cameras_to_check=10):
#     available_cameras = []
#     for i in range(max_cameras_to_check):
#         cap = cv2.VideoCapture(i)
#         if cap is not None and cap.isOpened():
#             available_cameras.append(i)
#             cap.release()
#     return available_cameras


# available_cameras = find_all_available_cameras()
# if available_cameras:
#     print(f"Available cameras are at indices {available_cameras}")
# else:
#     print("No available cameras  found")

#########################################################33

# Initialize the apriltag detector
detector = apriltag.Detector(apriltag.DetectorOptions(
            families='tag36h11',
            # families='tag41h12',
            # families = "tagStandard41h12",
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


cap = cv2.VideoCapture(0)

# newly added
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # Height

# Use your calibration matrix (this is for elpK)
mtx = np.array([
            [9.53e2, 0  , 975],
            [0  , 9.53e2, 550],
            [0  , 0  , 1]
        ])

# Use a null distortion matrix (assuming no lens distortion)
dist = np.zeros(5)

# # Load calibration parameters
# with np.load('calibration_parameters.npz') as X:
#     mtx, dist = X['mtx'], X['dist']


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


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale because apriltag detection requires grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize a black image of the same size as your frame
    axes_img = np.zeros_like(frame)

    # Perform detection on the thresholded image
    result = detector.detect(gray)

    #     # Draw the detections on the original color frame
    for tag in result:
        # Get the four corners of the tag
        corners = tag.corners.astype(int)

        # Draw the tag boundary
        cv2.polylines(frame, [corners], True, (255,0,0), 2)

        # Draw tag id
        cv2.putText(frame, str(tag.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Define the model points (in 3D space) of the tag (assuming it's a 5x5 square)
        model_points = np.float32([[-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0]])

        # Estimate the tag's pose in the camera frame
        ret, rvec, tvec = cv2.solvePnP(model_points, tag.corners.astype(np.float32).reshape(-1, 2), mtx, dist)

        # project 3D points to image plane
        axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,-0.1]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

        axes_img = draw(frame, rvec, tvec, mtx, dist)


        # axes_img = draw(axes_img, corners, imgpts)


    # Display the resulting frame with detections drawn
    cv2.imshow('AprilTag Detection', frame)
    cv2.imshow('AprilTag Axes', axes_img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
