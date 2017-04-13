import argparse
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="Classifier_All/allcatdogbody.xml",
	help="path to face detector haar cascade")
args = vars(ap.parse_args())

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
#image = cv2.resize(image, (300, image.shape[0]*300//image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector Haar cascade, then detect faces
# in the input image
detector = cv2.CascadeClassifier(args["cascade"])
rects = detector.detectMultiScale(gray, scaleFactor=1.3,
	minNeighbors=10, minSize=(75, 75))

print(rects[0])

X = []
Y = []
XW = []
YH = []

# loop over the reactnalges and record edge positions
for (i, (x, y, w, h)) in enumerate(rects):
	X.append(x)
	Y.append(y)
	XW.append(x+w)
	YH.append(y+h)

# get minimum and maximum rectangle coordinates
if len(X) != 0:
	xMin = min(X)
	yMin = min(Y)
	xwMax = max(XW)
	yhMax = max(YH)
else:
	height,width = gray.shape
	xMin,yMin,xwMax,yhMax = 0,0,width,height

# draw combined rectangle
cv2.rectangle(image, (xMin, yMin), (xwMax, yhMax), (0, 0, 255), 2)

# get portion of image within large rectangle
cropped_img = gray[yMin:yhMax,xMin:xwMax]

# compute local binary pattern of cropped image and its normalized histogram
lbp = local_binary_pattern(cropped_img, 8, 1, "default")
hist, _ = np.histogram(lbp, 256, density=True)

#print(hist)

# show the detected faces
cv2.imshow("Animal Faces", image)
cv2.imshow("Cropped", cropped_img)
cv2.waitKey(0)