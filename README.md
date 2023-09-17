<<<<<<< HEAD
@ RM_Retinas V2

Inspired by the original Retinas repository, Retinas V2 offers enhanced capabilities for AprilTag tracking and pose estimation using OpenCV.
🔧 Installation

    Clone the repository:

    bash

git clone https://github.com/GitWyd/RM_Retinas.git

Navigate to the cloned directory:

bash

cd RM_Retinas

Install the necessary dependencies:

bash

    pip install -r requirements.txt

🌟 Key Features

    AprilTag Detection: Provides real-time detection of AprilTags from a webcam feed.
    Camera Calibration: Enables calculation of the camera's calibration matrix.
    Pose Estimation: Determines the 3D pose of each detected tag relative to the camera using OpenCV's solvePnP.
    World Coordinate Transformation: Transforms detected tag poses to a global coordinate system, as defined by particular "world tags".
    Visualization: Enhances the video feed by highlighting detected tags, showcasing the tag ID, global pose, and reprojection error.
    Reprojection Error Calculation: Computes and showcases reprojection errors for all detected tags.

🚀 Usage

    Calibrate Your Camera:

    bash

python src/calibration.py

Once calibration is complete, launch the main application:

bash

    python main.py

📌 Notes

    The spatial relationships of the predefined world tags are hardcoded into the script. Kindly modify if your world tag setups are different.

>>>>>>> source-repo/main
