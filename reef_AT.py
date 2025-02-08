"""
https://robotpy.readthedocs.io/projects/apriltag/en/latest/robotpy_apriltag.html
https://pypi.org/project/photonlibpy/
PORT TO PHOTONVISION LATER: https://docs.photonvision.org/en/latest/docs/programming/photonlib/index.html

Steps:
1. Get the pose of AT
2. Find the 3D offsets of the AT -> reef branches (have a cad model for that)
3. Transform locations of reef branches via depth data -> camera frame 
4. Detect if objects are "fixed" onto those branches

Measurement in inches
"""
import cv2
import numpy as np
from robotpy_apriltag import \
    AprilTagField, AprilTagFieldLayout, AprilTagDetector, \
    AprilTagPoseEstimator
from wpimath.geometry import Transform3d
import json

detector =  AprilTagDetector()
detector.addFamily("tag36h11")


### CAD Offset:
### X = Downwards -> Right
### Y = Upwards -> Right
### Z = Upwards

# Measurement is in Inches

cad_to_branch_offset = {
    "L2-L" : np.array([-6.470, -12.854, 9.00]),
    "L2-R" : np.array([6.468, -12.833, 9.00]),
    "L3-L" : np.array([-6.470, -23.503, 16.457]),
    "L3-R" : np.array([6.468, -23.482, 16.442]),
    "L4-L" : np.array([-6.470, -58.4175, 0.921]),
    "L4-R" : np.array([6.468, -58.4175, 0.876])
}

### Camera AT Coordinate System: 
#   X is LEFT -> Right  [-inf, inf]
#   Y is TOP -> Down    [-inf, inf]
#   Z is DEPTH AWAY     [0, inf]


# Convert to meters:
for branch, offset in cad_to_branch_offset.items():
    for i in range(len(offset)):
        offset[i] *= 0.0254

# Obtain camera calibration data
with open("1280x800v1.json") as PV_config:
    data = json.load(PV_config)

    cameraIntrinsics = data["cameraIntrinsics"]["data"]
    fx = cameraIntrinsics[0]
    fy = cameraIntrinsics[4]
    cx = cameraIntrinsics[2]
    cy = cameraIntrinsics[5]

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)


    width = int(data["resolution"]["width"])
    height = int(data["resolution"]["height"])

    distCoeffsSize = int(data["distCoeffs"]["cols"])
    distCoeffs = np.array(data["distCoeffs"]["data"][0:distCoeffsSize], dtype=np.float32)

# Start Capture and Calibrate Camera
video_path = 4 # or do int 0 for /dev/video0
cap = cv2.VideoCapture(video_path) # /dev/video0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("camera intrinsics: {cx, cy, fx, fy}:", cx, cy, fx, fy)

# Tag Size: 165.1 mm = 0.1651 m
config = AprilTagPoseEstimator.Config(tagSize=0.1651, fx=fx, fy=fy, cx=cx, cy=cy)
estimator = AprilTagPoseEstimator(config)


frame_ct = -1
while cap.isOpened():
    ret, image = cap.read()
    frame_ct += 1

    if not ret:
        break

    # TODO: Undistort image
    image = cv2.undistort(image, K, distCoeffs)   
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    output = detector.detect(grayscale_image)
    for detections in output:
        print("ID", detections.getId(), "at frame count:", frame_ct)
        # Retrieve the corners of the AT detection
        points = []
        for corner in range(0, 4):
            x = detections.getCorner(corner).x
            y = detections.getCorner(corner).y
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=3)

        # Retrieve the center of the AT detection   
        centerX = detections.getCenter().x
        centerY = detections.getCenter().y
        cv2.circle(image, (int(centerX), int(centerY)), 2, color=(0, 255, 255), thickness=3)

        # Get Tag Pose Information
        tag_pose_estimation = AprilTagPoseEstimator.estimate(estimator, detections)
        tag_pose_estimation_matrix = tag_pose_estimation.toMatrix() # 4x4 Affline Transformation
        print(f"x: {tag_pose_estimation.x}, y: {tag_pose_estimation.y}, z: {tag_pose_estimation.z}")

        for offset_idx, offset_3d in cad_to_branch_offset.items():
            # solve camera -> branch via camera -> tag and tag -> branch transformations
            tag_to_reef_homography = np.append(offset_3d, 1.0) # ensures shape is 4x4
            camera_to_reef = np.dot(tag_pose_estimation_matrix, tag_to_reef_homography)
            
            x_cam, y_cam, z_cam, _ = camera_to_reef
            
            # project the 3D point to 2D image coordinates:
            u = (fx * x_cam / z_cam) + cx
            v = (fy * y_cam / z_cam) + cy

            cv2.circle(image, (int(u), int(v)), 5, (0, 255, 255), 2)
            cv2.putText(image, f"{offset_idx}", (int(u), int(v) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    cv2.imshow("frame", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break