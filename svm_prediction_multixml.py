import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
from sklearn import svm
from sklearn.externals import joblib
import csv

# function for writing csv files
def csv_writer(data, path):
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

# load haar-feature based classifiers
xml1 = 'Classifiers/4kdogcascade.xml'
xml2 = 'Classifier_cat_4k/cascade.xml'

# create cascade classifier objects
detector1 = cv2.CascadeClassifier(xml1)
detector2 = cv2.CascadeClassifier(xml2)

# load training model
clf = joblib.load('Training_Models/4kcatdogface.pkl')

# set path for output csv file
outputpath = '4kcatdogface.csv'

# generate list of files from test image folder
filenames = []
for root, dirs, files in os.walk('C:/Users/Raymond/Desktop/X_Test'):
    filenames = files 
testpath = 'C:/Users/Raymond/Desktop/X_Test/'

# initialize array to store LBP histograms
xSamples = np.zeros((len(filenames),256), dtype=np.float64)

# initialize list for storing image names and predictied labels
outputlist = []
outputlist.append("Image,Label".split(","))

# loop through all images in test image folder
for i in range(0,len(filenames)):
	# load input image, resize it, convert it to grayscale, and pass a 5x5 gaussian flter over it
	image = cv2.imread(testpath + filenames[i])
	image = cv2.resize(image, (150, image.shape[0]*150//image.shape[1]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	gray = cv2.blur(gray,(5,5))

	# load detectors and store coordinates of detected regions
	rects1 = detector1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))
	rects2 = detector2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))			

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
	
	# make one large rectangle by using extreme coordinates of smaller rectangles
	# and use that region to compute LBP and the LBP's normalize 
	cropped_img = gray[yMin:yhMax,xMin:xwMax]		
	lbp = local_binary_pattern(cropped_img, 8, 1, "default")
	hist, _ = np.histogram(lbp, 256, density=True)

	# save histogram into xSamples array
	xSamples[i] = hist

# predict labels using LBP histograms
predictions = clf.predict(xSamples)

# put image names and labels into list
for i in range(0,len(filenames)):
	outputlist.append((filenames[i]+","+predictions[i]).split(","))

# write output data into csv file
csv_writer(outputlist,outputpath)


