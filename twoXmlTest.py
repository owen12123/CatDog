import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
#ap.add_argument("-c", "--cascade",
#	default="Classifiers/cascade.xml",
#	help="path to face detector haar cascade")
args = vars(ap.parse_args())

xml1 = 'Classifiers/cascade.xml'
xml2 = 'Classifiers/cascade_8k_cat.xml'

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
#image = cv2.resize(image, (500, image.shape[0]*500//image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector Haar cascade, then detect faces
# in the input image
detector1 = cv2.CascadeClassifier(xml1)
detector2 = cv2.CascadeClassifier(xml2)
rects1 = detector1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))
rects2 = detector2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))

#TODO: add add for loops and stuff for when there are multiple rectangles

rects = []
rects.append(rects1[0])
rects.append(rects2[0])

# loop over the faces and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(image, "{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# show the detected faces
cv2.imshow("Animal Faces", image)
cv2.waitKey(0)
#cv2.imwrite('photo.jpg', image)