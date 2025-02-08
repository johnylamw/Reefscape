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


### Offset:
### X = Downwards -> Right
### Y = Upwards -> Right
### Z = Upwards

# Measurement is in Inches
cad_tag_to_branch_offset = [
    np.array([-4.560, 12.854, 10.103]), # L2:L
    np.array([-6.468, 12.833, 8.986]), # L2:R
    np.array([-11.017, 23.503, 13.831]), #L3:L
    np.array([-17.473, 23.482, 2.620]), #L3:R
    np.array([2.438, 0.000, 6.063]), #L4:L
    np.array([-3.992, 0.000, -5.164]), #L4:R
]

### [X, Y, Z] => [-Z, -X, Y] where CAD => WPI
def convert_to_wpi_coordinates(offsets):
    converted_list = []
    for points in offsets:
        converted_list.append(np.array([-points[2], -points[0], points[1]]))
    return converted_list

wpi_offsets = convert_to_wpi_coordinates(cad_tag_to_branch_offset)

# convert to meters:
wpi_offsets_meters = [points * 0.0254 for points in wpi_offsets]
with open("rearleft.json") as PV_config:
    data = json.load(PV_config)

    cameraIntrinsics = data["cameraIntrinsics"]["data"]
    fx = cameraIntrinsics[0]
    fy = cameraIntrinsics[4]
    cx = cameraIntrinsics[2]
    cy = cameraIntrinsics[5]

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)


    width = data["resolution"]["width"]
    height = data["resolution"]["height"]

    distCoeffs = np.array(data["distCoeffs"]["data"], dtype=np.float32)


# Start Capture and Calibrate Camera
video_path = "1.mp4" # or do int 0 for /dev/video0
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
    #image = cv2.undistort(image, K, distCoeffs)   
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
        print(tag_pose_estimation_matrix)

        for offset_idx, offset_3d in enumerate(wpi_offsets_meters):
            # solve camera -> branch via camera -> tag and tag -> branch transformations
            tag_to_reef_homography = np.append(offset_3d, 1.0) # ensures shape is 4x4
            camera_to_reef = np.dot(tag_pose_estimation_matrix, tag_to_reef_homography)
            
            x_cam, y_cam, z_cam, _ = camera_to_reef
            
            # project the 3D point to 2D image coordinates:
            u = (fx * x_cam / z_cam) + cx
            v = (fy * y_cam / z_cam) + cy

            cv2.circle(image, (int(u), int(v)), 5, (255, 255, 255), 2)
        
        cv2.imshow("detected", image)
        cv2.waitKey(5000)

    cv2.imshow("frame", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break