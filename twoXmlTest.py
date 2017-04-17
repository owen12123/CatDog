import argparse
import cv2
import numpy as np
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# set paths of haar-feature based classifiers
xml1 = 'Classifiers/4kdogcascade.xml'
xml2 = 'Classifier_cat_4k/cascade.xml'

# load input image, resize it, convert it to grayscale, and pass a 5x5 gaussian flter over it
image = cv2.imread(args["image"])
imagef = cv2.resize(image, (150, image.shape[0]*150//image.shape[1]))
grayf = cv2.cvtColor(imagef, cv2.COLOR_BGR2GRAY)
grayf = cv2.blur(grayf,(5,5))

# create cascade classifier objects
detector1 = cv2.CascadeClassifier(xml1)
detector2 = cv2.CascadeClassifier(xml2)

# load detectors and store coordinates of detected regions
rects1 = detector1.detectMultiScale(grayf, scaleFactor=1.3, minNeighbors=10, minSize=(20, 20))
rects2 = detector2.detectMultiScale(grayf, scaleFactor=1.3, minNeighbors=10, minSize=(20, 20))

rects = []
if len(rects1)>0 and len(rects2)==0:
	for i in range(0,len(rects1)):
		rects.append(rects1[i])
elif len(rects2)>0 and len(rects1)==0:	
	for i in range(0,len(rects2)):
		rects.append(rects2[i])
elif len(rects1)>0 and len(rects2)>0 : 
	rects.append(rects1[0])
	rects.append(rects2[0])

# initialize lists of rectangle edge positions		
X = []
Y = []
XW = []
YH = []		

# loop over the rectangles and record edge positions
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
# if no region of image was detected, use the whole image
else:
	height,width = gray.shape
	xMin,yMin,xwMax,yhMax = 0,0,width,height	

# draw large rectangle based on extreme coordinates of small rectangles
cv2.rectangle(imagef, (xMin, yMin), (xwMax, yhMax), (0, 0, 255), 2)

# show the detected faces
cv2.imshow("Animal Faces", imagef)
cv2.waitKey(0)

# save image with face detection rectangle
#cv2.imwrite('photo.jpg', imagef)