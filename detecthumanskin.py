import imutils
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", help = "path to the optional video")
args = vars(ap.parse_args())

lower = np.array([0,48,80], dtype = "uint8")
upper = np.array([20,255,255], dtype = "uint8")

if not args.get("video", False):
	cam = cv2.VideoCapture(0)

else:
	cam = cv2.VideoCapture(args["video"])

while True:
	(grabbed, frame) = cam.read()

	if args.get("video") and not grabbed:
		break

	frame = imutils.resize(frame, width = 400)
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted,lower,upper)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)

	cv2.imshow("image", np.hstack([frame,skin]))

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cam.release()
cv2.destroyAllWindows()

