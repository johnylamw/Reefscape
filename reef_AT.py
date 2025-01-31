"""
https://robotpy.readthedocs.io/projects/apriltag/en/latest/robotpy_apriltag.html
"""
import cv2
from robotpy_apriltag import AprilTagField, AprilTagFieldLayout, AprilTagDetector

image_path = "./image.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector =  AprilTagDetector()
detector.addFamily("tag36h11")

output = detector.detect(image)
print(output)