AprilTag Tracking and Pose Estimation with OpenCV

Requirements:

Python 3.x
OpenCV (cv2)
dt_apriltags library
NumPy
Matplotlib

Key Features:

AprilTag Detection: Detects AprilTags in real-time from a webcam feed.
Camera Calibration: Computes the calibration matrix for the camera
Pose Estimation: Computes the 3D pose of each detected tag relative to the camera using OpenCV's solvePnP.
World Coordinate Transformation: Transforms detected tag poses into a global coordinate system, defined by certain "world tags".
Visualization: Displays the video feed with detected tags highlighted, showing tag ID, global pose, and reprojection error.
Reprojection Error Calculation: Calculates and visualizes reprojection errors for all detected tags.

Notes:

The predefined spatial relationships of world tags are also hardcoded in the script. Adjust these if your world tag positions are different.
