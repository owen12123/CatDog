import csv
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
from sklearn import svm
from sklearn.externals import joblib

# function for reading csv files to assign labels to training imges
def csv_reader(file_obj):
    labels = []
    reader = csv.reader(file_obj)
    for row in reader:
        labels.append(row)
    return labels

# load haar-feature based classifiers
xml1 = 'Classifiers/4kdogcascade.xml'
xml2 = 'Classifiers/4kcatcascade.xml'

# create cascade classifier objects
detector1 = cv2.CascadeClassifier(xml1)
detector2 = cv2.CascadeClassifier(xml2)

# set path for image label csv file
yPath = 'Y_Train.csv'

# load image names and labels into array
with open(yPath, "r") as file:
    csv = csv_reader(file)

# remove header line from csv file
del csv[0]

# set path for training image folder
samplePath = 'C:/Users/Raymond/Desktop/trainSet'
xPath = samplePath + '/'

# generate list of images in training image folder
filenames = []
for root, dirs, files in os.walk(samplePath):
    filenames = files 

# initialize array for storing LBP histogram of each image
xData = np.zeros((len(filenames),256), dtype=np.float64)

# initialize list for storing image labels
yData = []

# loop through all images in training folder
for i in range(0,len(filenames)):
	curr_path = xPath + filenames[i]

	# load input image, resize it, convert it to grayscale, and pass a 5x5 gaussian flter over it
	image = cv2.imread(curr_path)
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
	# and use that region to compute LBP and the LBP's normalized histogram 
	cropped_img = gray[yMin:yhMax,xMin:xwMax]	
	lbp = local_binary_pattern(cropped_img, 8, 1, "default")
	hist, _ = np.histogram(lbp, 256, density=True)

	# save histogram into xData array
	xData[i] = hist

	# obtain image label using the current image's name
	img_index = int(filenames[i][:-4])
	yData.append(csv[img_index][1])

# create linear SVM object 
clf = svm.LinearSVC()

# fit data 
clf.fit(xData,yData)

# save training model in .pkl format
joblib.dump(clf, 'training_model.pkl')