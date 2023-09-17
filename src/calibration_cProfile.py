# calibration script using checkerboard pattern

def main():

    # import standard libraries
    import os

    # import third party libraries
    import cv2
    import numpy as np

    # Checkerboard (refers to the inner corners of the  squares)
    # Therefore, using 8 squares * 11 squares -> (7*10)
    CHECKERBOARD = (7,10)

    # Termination Criteria for refining the detected corners
    # Stop either when epsilon is 
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Create a numpy array which corresponds to the total num of corners and each corner's coordinates
    obj_pts = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    # Use meshgrid to create the coordinates for each (x, y, 0) point in the grid
    # note that z = 0 because it is planar
    obj_pts[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)


    # List to store object points(3D) and image points(2D)
    obj_pts_list = []
    img_pts_list = []

    # save in retina/asset/calibration folder
    save_dir ="assets/calibration"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # Start camera stream
    cap = cv2.VideoCapture(0)

    # ELP 4K
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # Width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  # Height

    # Resizing Variables to display in resolution
    # display_width = 1920
    # display_height = 1080

    print("Press 'spacebar' to capture an image of the checkerboard")
    print("Press 'q' to quit and return calibration results")

    frame_skip = 10
    frame_count = 0

    while True:
        ret, img = cap.read()
        frame_count +=1

        if frame_count % frame_skip != 0:
            continue

        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Search for chessboard corners
        ret, corners = cv2.findChessboardCorners(gray,CHECKERBOARD, None)

        if ret:

            # Draw and display corners
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

        # img_resized = cv2.resize(img, (display_width, display_height))
        # cv2.imshow("Chessboard Image", img_resized)

        cv2.imshow("Chessboard Image", img)

        key = cv2.waitKey(1)

        if (key & 0xFF == 32): # when spacebar is pressed and corners are detected

            if ret:
                # Refine detected checkerboard corners to sub-pixel accuracy
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Save the image
                filename = os.path.join(save_dir, f"calibration_img_{len(img_pts_list)+1}.png")
                cv2.imwrite(filename, img)
                print("Captured Image")

                # Append obj_pts and img_pts to the lists (used later for calibration)
                obj_pts_list.append(obj_pts)
                img_pts_list.append(refined_corners)

        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts_list, img_pts_list, gray.shape[::-1], None, None)

    # # Display results
    # print("\nCamera Matrix:")
    # print(mtx)

    # print("\nDistortion Coefficients:")
    # print(dist)

    # print("\nRotation Vectors:")
    # for rvec in rvecs:
    #     print(rvec)

    # print("\nTranslation Vectors:")
    # for tvec in tvecs:
    #     print(tvec)


    # # At the end of calibration.py
    # np.savez(os.path.join(save_dir, 'calibration_data.npz'), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    # print(f"Calibration data saved to {save_dir}/calibration_data.npz")


if __name__ == "__main__":
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    profiler.dump_stats("profile_results.pstats")
