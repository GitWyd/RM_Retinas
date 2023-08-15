import cv2
import apriltag
import numpy as np

# Initialize the apriltag detector
detector = apriltag.Detector()

# Initialize the video capture with OpenCV (0 is the id of the webcam)
cap = cv2.VideoCapture(4)

# Lists to store object points and image points from all the detected AprilTags
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in image plane


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale because apriltag detection requires grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the grayscale frame
    result = detector.detect(gray)

    # Draw the detections on the original color frame
    for tag in result:
        # Get the four corners of the tag as integers
        corners_int = tag.corners.astype(int)
        # print(type(corners_int))
        # print(corners_int.shape)

        # Draw the tag boundary
        cv2.polylines(frame, [corners_int], True, (255, 0, 0), 2)

        # convert the corners to float32 for calibration
        corners_float = tag.corners.astype(np.float32).reshape(-1, 2)

        # Append the 3D object points (assuming an approx. square AprilTag of size 2cm * 2cm)
        objpoints.append(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32))

        # Append the detected corner points
        imgpoints.append(corners_float)

    cv2.imshow('AprilTag Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Perform camera calibration using the collected object and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration parameters
np.savez("calibration_parameters.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Print the camera matrix, distortion coefficients, and transformation vectors
print("Camera Matrix:")
print(mtx)
print("\nDistortion Coefficients:")
print(dist)
print("\nRotation Vectors:")
print(rvecs)
print("\nTranslation Vectors:")
print(tvecs)


