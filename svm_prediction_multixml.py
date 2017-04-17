import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
import argparse
from sklearn import svm
from sklearn.externals import joblib
import csv

def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
'''
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to the input image")

ap.add_argument("-c", "--cascade",
	default="Classifier_All/allcatdogbody.xml",
	help="path to face detector haar cascade")
args = vars(ap.parse_args())
'''

xml1 = 'Classifiers/4kdogcascade.xml'
xml2 = 'Classifier_cat_4k/cascade.xml'
xml3 = 'Classifier_cat_ears/cascade.xml'

detector1 = cv2.CascadeClassifier(xml1)
detector2 = cv2.CascadeClassifier(xml2)
detector3 = cv2.CascadeClassifier(xml3)

clf = joblib.load('Training_Models/withEars.pkl')

outputpath = 'withEars.csv'

filenames = []
for root, dirs, files in os.walk('C:/Users/Raymond/Desktop/X_Test'):
    filenames = files 
testpath = 'C:/Users/Raymond/Desktop/X_Test/'

xSamples = np.zeros((len(filenames),256), dtype=np.float64)

outputlist = []
outputlist.append("Image,Label".split(","))

for i in range(0,len(filenames)):
# load the input image and convert it to grayscale
	image = cv2.imread(testpath + filenames[i])
	image = cv2.resize(image, (150, image.shape[0]*150//image.shape[1]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	gray = cv2.blur(gray,(5,5))
	# load the face detector Haar cascade, then detect faces
	# in the input image
	rects1 = detector1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))
	rects2 = detector2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))			
	rects3 = detector3.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))	

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

	if len(rects3) > 0:
		for i in range(0,len(rects3)):
			rects.append(rects3[i])
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
	
	# get portion of image within large rectangle
	cropped_img = gray[yMin:yhMax,xMin:xwMax]		
	# compute local binary pattern of cropped image and its normalized histogram
	lbp = local_binary_pattern(cropped_img, 8, 1, "default")
	hist, _ = np.histogram(lbp, 256, density=True)
	xSamples[i] = hist

predictions = clf.predict(xSamples)

for i in range(0,len(filenames)):
	outputlist.append((filenames[i]+","+predictions[i]).split(","))

csv_writer(outputlist,outputpath)


