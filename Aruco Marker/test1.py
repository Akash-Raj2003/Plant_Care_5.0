import cv2
import argparse
import numpy as np
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default = r"C:\Users\kyran\OneDrive\Desktop\Loughborough\Group Project\Aruco Marker", required = True, help = "path to output image containing ArUco Tag")
ap.add_argument("-i", "--id", type = int, required = True, help = "ID of ArUco tag to generate")
ap.add_argument("-t", "--type", type = str, default = "DICT_ARUCO_ORIGINAL", help = "type of ArUco tag to generate")

args = vars(ap.parse_args())

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUco tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])


print("[INFO] generating ArUco tag type '{}' with ID '{}'".format(args["type"], args['id']))

tag = np.zeros((300,300,1), dtype="uint8")

marker = cv2.aruco.generateImageMarker(arucoDict, args["id"], 300, tag, 1)

marker_with_border = cv2.copyMakeBorder(marker, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value = 255)

cv2.imwrite(args["output"], marker_with_border)
cv2.imshow("ArUco Tag", tag)
cv2.waitKey(0)

#python test1.py --id 13 --type DICT_4X4_50 --output DICT_4x4_50_id13.png



