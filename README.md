# robot_metabolism_retina

AprilTag Detection and Pose Estimation with OpenCV****

Requirements:
Python 3.x
OpenCV (cv2)
dt_apriltags library
NumPy
Matplotlib

Setup
Ensure you have Python 3 installed.
Install the required libraries:
bash
Copy code
pip install opencv-python dt-apriltags numpy matplotlib

How to Run
Clone this repository:
bash
Copy code
git clone <repository-url>
cd <repository-directory>
Execute the main script:
bash
Copy code
python <script_name>.py
Replace <script_name> with the name of the script file.

When running, a window will display the video feed with detected AprilTags highlighted and annotated with their ID, global pose, and reprojection error.
To exit, press the 'q' key.


Key Features
AprilTag Detection: Detects AprilTags in real-time from a webcam feed.
Pose Estimation: Computes the 3D pose of each detected tag relative to the camera using OpenCV's solvePnP.
World Coordinate Transformation: Transforms detected tag poses into a global coordinate system, defined by certain "world tags".
Visualization: Displays the video feed with detected tags highlighted, showing tag ID, global pose, and reprojection error.
Reprojection Error Calculation: Calculates and visualizes reprojection errors for all detected tags.

Notes
The predefined spatial relationships of world tags are also hardcoded in the script. Adjust these if your world tag positions are different.
