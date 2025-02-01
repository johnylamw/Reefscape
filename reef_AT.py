"""
https://robotpy.readthedocs.io/projects/apriltag/en/latest/robotpy_apriltag.html
"""
import cv2
import numpy as np
from robotpy_apriltag import AprilTagField, AprilTagFieldLayout, AprilTagDetector

image_path = "./image.png"
image = cv2.imread(image_path)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector =  AprilTagDetector()
detector.addFamily("tag36h11")

output = detector.detect(grayscale_image)
for detections in output:
    print("ID", detections.getId())
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

    # Get Homography Information:
    print(detections.getHomographyMatrix())
cv2.namedWindow("Polygon", cv2.WINDOW_GUI_NORMAL)

while True:
    cv2.imshow("frame", image)
    if cv2.waitKey(1) == ord('q'):
        break;