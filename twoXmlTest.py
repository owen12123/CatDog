import argparse
import cv2
import numpy as np
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
#ap.add_argument("-c", "--cascade",
#	default="Classifiers/cascade.xml",
#	help="path to face detector haar cascade")
args = vars(ap.parse_args())

xml1 = 'Classifiers/4kdogcascade.xml'
xml2 = 'Classifier_cat_4k/cascade.xml'
xml3 = 'Classifier_All/allcatdogbody.xml'

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image2 = np.copy(image)

#face
imagef = cv2.resize(image, (150, image.shape[0]*150//image.shape[1]))
grayf = cv2.cvtColor(imagef, cv2.COLOR_BGR2GRAY)
grayf = cv2.blur(grayf,(5,5))

#body



# load the face detector Haar cascade, then detect faces
# in the input image
detector1 = cv2.CascadeClassifier(xml1)
detector2 = cv2.CascadeClassifier(xml2)
#detector3 = cv2.CascadeClassifier(xml3)
rects1 = detector1.detectMultiScale(grayf, scaleFactor=1.3, minNeighbors=10, minSize=(20, 20))
rects2 = detector2.detectMultiScale(grayf, scaleFactor=1.3, minNeighbors=10, minSize=(20, 20))
#rects3 = detector2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))

#TODO: add add for loops and stuff for when there are multiple rectangles

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
'''
# loop over the faces and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
	cv2.rectangle(imagef, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(imagef, "{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

for (i, (x, y, w, h)) in enumerate(rects3):
	cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(image2, "{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
'''

cv2.rectangle(imagef, (xMin, yMin), (xwMax, yhMax), (0, 0, 255), 2)
# show the detected faces
cv2.imshow("Animal Faces", imagef)
#cv2.imshow("Body Detection", image2)
cv2.waitKey(0)
cv2.imwrite('photo.jpg', imagef)